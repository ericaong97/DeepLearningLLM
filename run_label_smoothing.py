import torch
import torch.nn as nn
from data_utils import get_train_loader, get_val_loader, get_test_loader, set_seed
from baseline_transformer_architecture import create_small_transformer
from modeling_functions import validate_transformer
from optimizer_scheduler import get_optimizer, get_plateau_scheduler, linear_teacher_scheduler
from tokenizers import Tokenizer
from tqdm.auto import tqdm
import json
import matplotlib.pyplot as plt

NUM_EPOCHS = 5

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=1):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(1) - 1))
            mask = target != self.ignore_index
            target = target.masked_fill(~mask, 0)
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            true_dist.masked_fill_(~mask.unsqueeze(1), 0)
        return torch.mean(torch.sum(-true_dist * pred.log_softmax(dim=1), dim=1))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  Lightweight model config
    config = {
        "vocab_size": 20000,
        "dropout": 0.1,
        "d_model": 384,
        "nhead": 6,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "dim_feedforward": 1536
    }

    tokenizer = Tokenizer.from_file("cnn_bpe_tokenizer_20k.json")
    pad_idx = tokenizer.token_to_id("[PAD]")

    model = create_small_transformer(**config).to(device)
    optimizer = get_optimizer(model)
    plateau_scheduler = get_plateau_scheduler(optimizer)
    teacher_scheduler = linear_teacher_scheduler
    criterion = LabelSmoothingLoss(smoothing=0.01, ignore_index=pad_idx)

    #  Reduce batch size for GPU safety
    train_loader = get_train_loader(tokenizer, batch_size=32, num_workers=2)
    val_loader = get_val_loader(tokenizer, batch_size=4, num_workers=0)
    test_loader = get_test_loader(tokenizer, batch_size=4, num_workers=0)

    history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": None,
        "test_accuracy": None,
        "learning_rate": [],
        "teacher_forcing_ratio": [],
    }

    # === Training Loop ===
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        tf_ratio = teacher_scheduler.step()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/15"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            output = model(
                src=input_ids,
                tgt=labels,
                src_key_padding_mask=(attn_mask == 0),
                teacher_forcing_ratio=tf_ratio
            )
            logits = output.view(-1, output.size(-1))
            targets = labels[:, 1:].contiguous().view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Clear cache before validation
        torch.cuda.empty_cache()
        val_loss, _ = validate_transformer(model, val_loader, criterion, tokenizer, device, pad_idx, max_length_generate=40)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(optimizer.param_groups[0]['lr'])
        history["teacher_forcing_ratio"].append(tf_ratio)

        plateau_scheduler.step(val_loss)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # === Final Test Evaluation ===
    torch.cuda.empty_cache()
    model.eval()
    correct = total = 0
    total_test_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            output = model(
                src=input_ids,
                tgt=labels,
                src_key_padding_mask=(attn_mask == 0)
            )
            logits = output.view(-1, output.size(-1))
            targets = labels[:, 1:].contiguous().view(-1)
            loss = criterion(logits, targets)
            total_test_loss += loss.item()

            pred_ids = logits.argmax(dim=-1)
            correct += (pred_ids == targets).sum().item()
            total += targets.numel()

    avg_test_loss = total_test_loss / len(test_loader)
    accuracy = correct / total
    history["test_loss"] = avg_test_loss
    history["test_accuracy"] = accuracy

    print(f"\n Test Loss: {avg_test_loss:.4f} | Test Accuracy: {accuracy:.4f}")

    # === Save history ===
    with open("histories/label_smooth_history.json", "w") as f:
        json.dump(history, f, indent=2)

