# optimizer_and_scheduler.py
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

# 1. Optimizer setup
def get_optimizer(model):
    return AdamW(
        model.parameters(),
        lr=2e-4,
        eps=1e-6,
        weight_decay=0.01
    )
    
# 2. Plateau scheduler
def get_plateau_scheduler(optimizer):
    return ReduceLROnPlateau(optimizer, mode='min',
                            patience=1,
                            factor=0.5,
                            min_lr=1e-6)