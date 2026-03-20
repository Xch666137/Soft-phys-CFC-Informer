"""
位置编码实现

Informer中的位置编码模块，为序列添加位置信息。
支持绝对位置编码和RoPE旋转位置编码。
"""

import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    位置编码
    
    使用正弦和余弦函数为序列中的每个位置添加唯一的位置信息。
    基于"Attention Is All You Need"中的实现。
    
    Args:
        d_model: 模型维度
        max_len: 最大序列长度
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        self.d_model = d_model
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # 注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播

        Args:
            x: 输入张量，支持多种格式：
                - [seq_len, batch_size, d_model]
                - [batch_size, seq_len, d_model]

        Returns:
            添加位置编码后的张量，保持输入格式
        """
        if x.dim() == 3:  # [batch_size, seq_len, d_model]
            seq_len = x.size(1)
            pe = self.pe[:seq_len, :]  # [seq_len, 1, d_model]
            pe = pe.transpose(0, 1)  # [1, seq_len, d_model]
            return x + pe.to(x.device)
        elif x.dim() == 2:  # [seq_len, d_model]
            seq_len = x.size(0)
            return x + self.pe[:seq_len, :].squeeze(0).to(x.device)
        else:
            raise ValueError(f"不支持的输入维度: {x.dim()}")


class RoPEPositionalEncoding(nn.Module):
    """
    旋转位置编码（Rotary Positional Encoding）
    
    通过旋转矩阵将位置信息注入到向量表示中，保持相对位置信息的内积性质。
    相比绝对位置编码，RoPE具有更好的长度外推能力和位置编码表达能力。
    
    Args:
        d_model: 模型维度（必须是偶数）
        max_len: 最大序列长度
        base: 旋转频率基数（默认10000）
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, base: int = 10000):
        super().__init__()
        
        self.d_model = d_model
        self.base = base
        self.max_len = max_len

        assert d_model % 2 == 0, "d_model必须是偶数"
        
        # 预计算旋转矩阵的频率
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

        # 预计算正弦和余弦缓存
        self._update_cos_sin_cache(max_len)

    def _update_cos_sin_cache(self, seq_len: int):
        """更新正弦和余弦缓存"""
        self.max_cached_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)

        # 计算频率
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)

        # 分离实部和虚部
        emb = freqs
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """
        应用旋转位置编码

        Args:
            x: 输入张量 [batch_size, n_heads, seq_len, dim] 或 [batch_size, seq_len, dim]
            offset: 位置偏移量(用于增量解码)

        Returns:
            旋转后的张量
        """
        seq_len = x.size(-2)

        # 确保预计算的长度足够
        if seq_len + offset > self.max_cached_len:
            self._update_cos_sin_cache(seq_len + offset + 128)  # 额外缓存一些位置

        # 获取对应的正弦和余弦值
        cos = self.cos_cached[:, :, offset:offset+seq_len, :]
        sin = self.sin_cached[:, :, offset:offset+seq_len, :]

        # 应用旋转
        if x.dim() == 4:  # [batch_size, n_heads, seq_len, dim]
            x1, x2 = x[..., :self.d_model//2], x[..., self.d_model//2:]
            rotated_x = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        elif x.dim() == 3:  # [batch_size, seq_len, dim]
            x1, x2 = x[..., :self.d_model//2], x[..., self.d_model//2:]
            rotated_x = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        else:
            raise ValueError(f"不支持的输入维度: {x.dim()}")

        return rotated_x