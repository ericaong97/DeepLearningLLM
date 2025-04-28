# Import libraries
import torch
import torch.nn as nn
from data_utils import get_train_loader, get_val_loader, get_test_loader, set_seed
from baseline_transformer_architecture import create_small_transformer
from modeling_functions import validate_transformer, generate_with_beam_search, calculate_and_save_rouge
from optimizer_scheduler import get_plateau_scheduler, linear_teacher_scheduler, get_optimizer
from visualization import visualize_training_dynamics, load_history

from tokenizers import Tokenizer
from tqdm.auto import tqdm
import json
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from torch.optim.lr_scheduler import LambdaLR

# === Simple token-level dropout ===
def random_token_dropout(input_ids, dropout_prob=0.1, pad_token_id=1):
    keep_mask = (torch.rand_like(input_ids.float()) > dropout_prob).long()
    return input_ids * keep_mask + pad_token_id * (1 - keep_mask)

# === ROUGE scoring for training/test sets ===
def compute_rouge(reference_ids, prediction_ids):
    refs = [tokenizer.decode(r.tolist(), skip_special_tokens=True) for r in reference_ids]
    preds = [tokenizer.decode(p.tolist(), skip_special_tokens=True) for p in prediction_ids]
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for r, p in zip(refs, preds):
        result = scorer.score(r, p)
        for k in scores:
            scores[k].append(result[k].fmeasure)
    return {k: sum(v)/len(v) if v else 0.0 for k, v in scores.items()}

# === Main code section ===
if __name__ == "__main__":
    torch.cuda.empty_cache()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # == Model Configuration ==
    filename = "data_aug"
    config = {
        "vocab_size": 20000,
        "dropout": 0.1,
        # "max_len": 512,
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "dim_feedforward": 2048
    }

    tokenizer = Tokenizer.from_file("cnn_bpe_tokenizer_20k.json")
    pad_idx = tokenizer.token_to_id("[PAD]")

    model = create_small_transformer(**config).to(device)
    optimizer = get_optimizer(model)
    plateau_scheduler = get_plateau_scheduler(optimizer)
    teacher_scheduler = linear_teacher_scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    train_loader = get_train_loader(tokenizer, batch_size=64, num_workers=4)
    val_loader = get_val_loader(tokenizer, batch_size=64, num_workers=2)
    test_loader = get_test_loader(tokenizer, batch_size=64, num_workers=2)
    
    # What is stored
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_rouge1": [],
        "train_rouge2": [],
        "train_rougeL": [],
        "rouge1": [], # validation rouge scores (named this way for visualization formatting)
        "rouge2": [],
        "rougeL": [],
        "learning_rate": [],
        "teacher_forcing_ratio": [],
        "global_step": 0
    }
    
    start_epoch = 0

    # Warmup scheduler configuration
    warmup_steps = 1 * len(train_loader)  # 1 epochs of batches
    initial_lr = 5e-5
    target_lr = 2e-4
    
    # Scaling from 5e-5 to 2e-4
    warmup_scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda step: (initial_lr + (target_lr - initial_lr) * min(1.0, step / warmup_steps))/ target_lr
    )
    
    for epoch in range(5):
        
        # === Training Phase ===
        model.train()
        total_loss = 0
        predictions, references = [], []
        epoch_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/5"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            
            if epoch < 2:
                tf_ratio = 0.9  # Warmup phase, keep the same
            else:
                tf_ratio = teacher_scheduler.step()

            input_ids_aug = random_token_dropout(input_ids, dropout_prob=0.1, pad_token_id=pad_idx)

            optimizer.zero_grad()
            
            output = model(
                src=input_ids_aug,
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
            
            history['global_step'] += 1
                
            # Learning rate warmup setting
            if history['global_step'] < warmup_steps:
                warmup_scheduler.step()

            with torch.no_grad():
                predictions.append(output.argmax(dim=-1))
                references.append(labels[:, 1:])

        avg_train_loss = total_loss / len(train_loader)
        train_rouge_scores = compute_rouge(torch.cat(references), torch.cat(predictions))

        torch.cuda.empty_cache()
        
        # === Validation Phase ===
        val_loss, val_rouge_scores = validate_transformer(
            model, val_loader, criterion, tokenizer, device, pad_idx, max_length_generate=40)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(optimizer.param_groups[0]['lr'])
        history["teacher_forcing_ratio"].append(tf_ratio)

        for k in ["rouge1", "rouge2", "rougeL"]:
            history[f"train_{k}"].append(train_rouge_scores[k])
            history[f"{k}"].append(val_rouge_scores[f"{k}_mean"])
            
        # Update history and schedulers
        if epoch >= 2:
            plateau_scheduler.step(val_loss)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        for k in ["rouge1", "rouge2", "rougeL"]:
            print(f"  Train {k.upper()}: {train_rouge_scores[k]:.4f}, Val {k.upper()}: {val_rouge_scores[f'{k}_mean']:.4f}")
    
    # Final save
    torch.save(model.state_dict(), f"{filename}_final_model.pt")
    with open(f"histories/{filename}_history.json", 'w') as f:
        json.dump(history, f, indent=4)