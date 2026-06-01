import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SelfAttentionDistillation(nn.Module):
    """Self-attention distillation via 1D conv + max pooling (Informer)."""

    def __init__(self, d_model: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model, out_channels=d_model,
            kernel_size=kernel_size, padding=kernel_size // 2,
        )
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, d_model = x.size()
        if seq_len <= 2:
            x = x.transpose(1, 2)
            x = self.conv(x)
            x = self.activation(x)
            return x.transpose(1, 2)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.activation(x)
        if seq_len % self.pool.stride != 0:
            pad_len = self.pool.stride - (seq_len % self.pool.stride)
            x = F.pad(x, (0, pad_len), mode='constant', value=0)
        x = self.pool(x)
        return x.transpose(1, 2)
