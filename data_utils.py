"""
data_utils.py - How to use create data loaders
"""
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

def collate_fn(batch, pad_idx):
    """
    Enhanced collate function with proper padding handling.
    
    Args:
        batch: List of samples (each sample is a dict)
        pad_idx: Padding token ID
    
    Returns:
        Dictionary of padded tensors with consistent shapes
    """
    # Extract all sequences
    input_ids = [torch.as_tensor(sample["input_ids"]) for sample in batch]
    labels = [torch.as_tensor(sample["labels"]) for sample in batch]
    
    # Get max lengths in this batch
    max_input_len = max(len(seq) for seq in input_ids)
    max_label_len = max(len(seq) for seq in labels)
    
    # Pad sequences with proper values
    padded_inputs = torch.full((len(batch), max_input_len), pad_idx, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_label_len), pad_idx, dtype=torch.long)
    
    # Fill tensors with actual data
    for i, (inp, lbl) in enumerate(zip(input_ids, labels)):
        padded_inputs[i, :len(inp)] = inp
        padded_labels[i, :len(lbl)] = lbl
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (padded_inputs != pad_idx).float()
    
    return {
        "input_ids": padded_inputs,
        "labels": padded_labels,
        "attention_mask": attention_mask
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
        collate_fn=lambda b: collate_fn(b,pad_idx=1)
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
        collate_fn=lambda b: collate_fn(b,pad_idx=1)
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
        collate_fn=lambda b: collate_fn(b,pad_idx=1)
    )