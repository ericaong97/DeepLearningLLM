
from tokenizers import Tokenizer
from torch.utils.data import Dataset
import torch
# import json
from datasets import load_dataset

# class BPETokenizerWrapper:
#     def __init__(self, tokenizer_path, max_len=128):
#         self.tokenizer = Tokenizer.from_file(tokenizer_path)
#         self.max_len = max_len
#         self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
#         self.sos_token_id = self.tokenizer.token_to_id("[SOS]")
#         self.eos_token_id = self.tokenizer.token_to_id("[EOS]")

#     def encode(self, text):
#         ids = self.tokenizer.encode(text).ids
#         ids = [self.sos_token_id] + ids[:self.max_len - 2] + [self.eos_token_id]
#         return torch.tensor(ids + [self.pad_token_id] * (self.max_len - len(ids)))

#     def decode(self, ids):
#         ids = ids.tolist()
#         if self.eos_token_id in ids:
#             ids = ids[:ids.index(self.eos_token_id)]
#         return self.tokenizer.decode([i for i in ids if i != self.pad_token_id and i != self.sos_token_id])

#     def batch_encode(self, texts):
#         return torch.stack([self.encode(t) for t in texts])

# class SummarizationDataset(Dataset):
#     def __init__(self, articles, summaries, tokenizer):
#         self.articles = articles
#         self.summaries = summaries
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.articles)

#     def __getitem__(self, idx):
#         src = self.tokenizer.encode(self.articles[idx])
#         tgt = self.tokenizer.encode(self.summaries[idx])
#         return src, tgt

# def collate_fn(batch):
#     src_batch, tgt_batch = zip(*batch)
#     return torch.stack(src_batch), torch.stack(tgt_batch)

# def generate_json_from_cnn_dailymail(train_path="train_data.json", test_path="test_data.json"):
#     dataset = load_dataset("cnn_dailymail", "3.0.0")
#     train_articles = dataset["train"]["article"] + dataset["validation"]["article"]
#     train_summaries = dataset["train"]["highlights"] + dataset["validation"]["highlights"]
#     test_articles = dataset["test"]["article"]
#     test_summaries = dataset["test"]["highlights"]

#     with open(train_path, "w", encoding="utf-8") as f:
#         json.dump({"articles": train_articles, "summaries": train_summaries}, f, ensure_ascii=False)

#     with open(test_path, "w", encoding="utf-8") as f:
#         json.dump({"articles": test_articles, "summaries": test_summaries}, f, ensure_ascii=False)

#     print(f"Saved {len(train_articles)} training and {len(test_articles)} test samples.")

from torch.utils.data import Dataset,DataLoader
import torch
from datasets import load_dataset
import random
import numpy as np

# 0. Set a global seed
def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1. Dataset Processing
class CNNDataset(Dataset):
    def __init__(self, split, tokenizer, max_article_len=400, max_summary_len=40):
        # Data related variables
        self.data = load_dataset("cnn_dailymail", "3.0.0",split=split)
        self.tokenizer = tokenizer
        self.sos_id = tokenizer.token_to_id("[SOS]") 
        self.eos_id = tokenizer.token_to_id("[EOS]")  
        self.pad_id = tokenizer.token_to_id("[PAD]") 
        
        # Verify tokenizer exists
        if None in [self.sos_id, self.eos_id, self.pad_id]:
            raise ValueError("Tokenizer missing required special tokens")
            
        self.max_article_len = max_article_len
        self.max_summary_len = max_summary_len
        
    def _process_text(self, text, is_summary=False):
        # 1. Encode text
        encoding = self.tokenizer.encode(text)
        tokens = encoding.ids
        # 2. Determine max length (only difference for summaries vs articles)
        max_len = self.max_summary_len if is_summary else self.max_article_len
                
        # 3. Apply identical processing logic to both which reserve space for SOS/EOS
        if len(tokens) > max_len - 2: 
            tokens = tokens[:max_len - 2]
        processed = [self.sos_id] + tokens + [self.eos_id]
        
        # 4. Pad to exactly max_len (identical for both)
        padding_needed = max_len - len(processed)
        if padding_needed > 0:
            processed = processed + [self.pad_id] * padding_needed
        
        return processed[:max_len]

    def __getitem__(self, idx):
        article = self._process_text(self.data[idx]["article"])
        summary = self._process_text(self.data[idx]["highlights"], is_summary=True)
        
        return {
            "input_ids": torch.tensor(article,dtype=torch.long),
            "labels": torch.tensor(summary,dtype=torch.long),
            "attention_mask": torch.tensor([int(tok != self.pad_id) for tok in article],dtype=torch.long),
            "decoder_attention_mask": torch.tensor([int(tok != self.pad_id) for tok in summary],dtype=torch.long)}

    def __len__(self):
        return len(self.data)

# 2. Collate function
def collate_fn(batch):
    """Convert list of dicts to dict of tensors (handles existing tensors)"""
    def _prepare_tensor(data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.long)
        return data.clone().detach()

    return {
        "input_ids": torch.stack([_prepare_tensor(x["input_ids"]) for x in batch]),
        "labels": torch.stack([_prepare_tensor(x["labels"]) for x in batch]),
        "attention_mask": torch.stack([_prepare_tensor(x["attention_mask"]) for x in batch])
    }

# 3. Define seed for worker in data loading process
def seed_worker(worker_id):
    """For DataLoader worker reproducibility"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 3. Get Train Loader
def get_train_loader(tokenizer, batch_size=64, num_workers=2, shuffle=True):
    """Create training DataLoader for CNN/DailyMail dataset"""
    train_dataset = CNNDataset(split="train", tokenizer=tokenizer)
    generator = torch.Generator()
    generator.manual_seed(42)
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=collate_fn
    )

# 4. Get Val Loader
def get_val_loader(tokenizer, batch_size=64, num_workers=2):
    """Create validation DataLoader for CNN/DailyMail dataset"""
    val_dataset = CNNDataset(split="validation", tokenizer=tokenizer)
    generator = torch.Generator()
    generator.manual_seed(42)
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=collate_fn
    )
    
# 5. Get Test Loader
def get_test_loader(tokenizer, batch_size=64, num_workers=2):
    """Create validation DataLoader for CNN/DailyMail dataset"""
    test_dataset = CNNDataset(split="test", tokenizer=tokenizer)
    generator = torch.Generator()
    generator.manual_seed(42)
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=collate_fn
    )