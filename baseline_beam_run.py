
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import (
    BPETokenizerWrapper,
    SummarizationDataset,
    collate_fn,
    generate_json_from_cnn_dailymail
)
from transformer_model import TransformerSummarizer
import os, json
from tqdm import tqdm

# === Config ===
tokenizer_path = "cnn_bpe_tokenizer_20k.json"
max_len = 128
batch_size = 64
vocab_size = 20000
num_epochs = 3
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
beam_width = 4
max_decode_len = 60

# === Generate JSON if not present ===
if not os.path.exists("train_data.json") or not os.path.exists("test_data.json"):
    generate_json_from_cnn_dailymail()

# === Load Tokenizer ===
tokenizer = BPETokenizerWrapper(tokenizer_path, max_len=max_len)

# === Load Data ===
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["articles"], data["summaries"]

train_articles, train_summaries = load_json("train_data.json")
test_articles, test_summaries = load_json("test_data.json")

train_dataset = SummarizationDataset(train_articles, train_summaries, tokenizer)
test_dataset = SummarizationDataset(test_articles, test_summaries, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# === Model Setup ===
model = TransformerSummarizer(vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# === Training ===
def train_one_epoch(model, dataloader):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(dataloader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        optimizer.zero_grad()
        logits = model(src, tgt_input)
        loss = criterion(logits.view(-1, vocab_size), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# === Beam Search ===
def beam_search(model, src, sos_token_id, eos_token_id, max_len=60, beam_width=4):
    model.eval()
    src = src.to(device).unsqueeze(0)
    memory = model.pos_encoder(model.embedding_src(src))

    beams = [(torch.tensor([[sos_token_id]], device=device), 0.0)]
    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == eos_token_id:
                new_beams.append((seq, score))
                continue
            tgt_emb = model.pos_encoder(model.embedding_tgt(seq))
            out = model.transformer(memory, tgt_emb)
            logits = model.output_layer(out[:, -1, :])
            probs = torch.log_softmax(logits, dim=-1)
            topk = torch.topk(probs, beam_width, dim=-1)

            for i in range(beam_width):
                token = topk.indices[0, i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, token], dim=1)
                new_score = score + topk.values[0, i].item()
                new_beams.append((new_seq, new_score))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
    return beams[0][0].squeeze()

# === Run ===
for epoch in range(num_epochs):
    loss = train_one_epoch(model, train_loader)
    print(f"Epoch {epoch+1} - Train Loss: {loss:.4f}")

# === Inference (Beam Search) ===
model.eval()
print("\nSample Beam Search Inference:")
with torch.no_grad():
    for src, tgt in test_loader:
        src = src[0]
        pred_ids = beam_search(model, src, tokenizer.sos_token_id, tokenizer.eos_token_id,
                               max_len=max_decode_len, beam_width=beam_width)
        pred_text = tokenizer.decode(pred_ids)
        tgt_text = tokenizer.decode(tgt[0])
        print(f"\nTarget: {tgt_text}\nPredicted: {pred_text}")
        break  # show only 1 example
