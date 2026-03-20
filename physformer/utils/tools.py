"""
General utility functions for PhysFormer.
"""

import torch
import numpy as np
import random


def set_seed(seed=2024):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, epoch, args):
    """Decay learning rate with cosine annealing."""
    if hasattr(args, 'lradj') and args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        if epoch in lr_adjust:
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
