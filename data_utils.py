
from tokenizers import Tokenizer
from torch.utils.data import Dataset
import torch
import json
from datasets import load_dataset

class BPETokenizerWrapper:
    def __init__(self, tokenizer_path, max_len=128):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.sos_token_id = self.tokenizer.token_to_id("[SOS]")
        self.eos_token_id = self.tokenizer.token_to_id("[EOS]")

    def encode(self, text):
        ids = self.tokenizer.encode(text).ids
        ids = [self.sos_token_id] + ids[:self.max_len - 2] + [self.eos_token_id]
        return torch.tensor(ids + [self.pad_token_id] * (self.max_len - len(ids)))

    def decode(self, ids):
        ids = ids.tolist()
        if self.eos_token_id in ids:
            ids = ids[:ids.index(self.eos_token_id)]
        return self.tokenizer.decode([i for i in ids if i != self.pad_token_id and i != self.sos_token_id])

    def batch_encode(self, texts):
        return torch.stack([self.encode(t) for t in texts])

class SummarizationDataset(Dataset):
    def __init__(self, articles, summaries, tokenizer):
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        src = self.tokenizer.encode(self.articles[idx])
        tgt = self.tokenizer.encode(self.summaries[idx])
        return src, tgt

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    return torch.stack(src_batch), torch.stack(tgt_batch)

def generate_json_from_cnn_dailymail(train_path="train_data.json", test_path="test_data.json"):
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    train_articles = dataset["train"]["article"] + dataset["validation"]["article"]
    train_summaries = dataset["train"]["highlights"] + dataset["validation"]["highlights"]
    test_articles = dataset["test"]["article"]
    test_summaries = dataset["test"]["highlights"]

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump({"articles": train_articles, "summaries": train_summaries}, f, ensure_ascii=False)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump({"articles": test_articles, "summaries": test_summaries}, f, ensure_ascii=False)

    print(f"Saved {len(train_articles)} training and {len(test_articles)} test samples.")
