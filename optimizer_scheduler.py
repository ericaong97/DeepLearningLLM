# optimizer_and_scheduler.py
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
                            patience=2,
                            factor=0.5,
                            threshold=0.01, 
                            threshold_mode='rel',
                            min_lr=1e-5)    
    
# 3. Teacher Forcing Ratio Class
class TeacherForcingScheduler:
    def __init__(self, initial_ratio=0.9, min_ratio=0.1, 
                decay_type='exp', decay_steps=4487*8, 
                decay_rate=0.9998, staircase=False):
        """
        Args:
            decay_type: 'exp' (exponential) or 'linear'
            decay_steps: Steps to decay from initial→min_ratio
            staircase: If True, decay at discrete intervals
        """
        self.initial_ratio = initial_ratio
        self.min_ratio = min_ratio
        self.decay_type = decay_type
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self._step = 0  # Critical for resuming
        
    def step(self):
        """Call this EVERY batch update"""
        self._step += 1
        
        if self.decay_type == 'linear':
            ratio = self.initial_ratio - (self.initial_ratio - self.min_ratio) * (min(1.0, self._step / self.decay_steps))
        else:  # exponential
            if self.staircase:
                ratio = self.initial_ratio * (self.decay_rate ** (self._step // self.decay_steps))
            else:
                ratio = self.initial_ratio * (self.decay_rate**self._step)
        
        self.current_ratio = max(ratio, self.min_ratio)
        return self.current_ratio
    
    def state_dict(self):
        return {k: v for k, v in self.__dict__.items()}
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


# 4. Exponential decay teacher forcing
exp_teacher_scheduler = TeacherForcingScheduler(
    initial_ratio=0.9,
    min_ratio=0.1,
    decay_rate=0.9998,  # Reaches 0.1 after ~2 epochs
    decay_steps=4487*8)

# 5. Linear decay teacher forcing
linear_teacher_scheduler = TeacherForcingScheduler(
    decay_type='linear',
    decay_steps=4487*6  # Full decay after 8 epochs
)