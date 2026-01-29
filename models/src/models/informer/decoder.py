"""
解码器实现

Informer中的解码器模块，包括自注意力、交叉注意力和前馈网络。
保持原始功能不变，仅做模块化分解。
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from .attention import ProbAttention, FullAttention
from .encoder import FeedForward, AttentionLayer


class DecoderLayer(nn.Module):
    """
    解码器层，支持RoPE
    
    单个解码器层，包含自注意力、交叉注意力和前馈网络。
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 use_rope: bool = False, rope_base: int = 10000):
        super().__init__()
        self.self_attention = AttentionLayer(
            ProbAttention(d_model, n_heads, dropout, use_rope=use_rope, rope_base=rope_base),
            d_model, n_heads, dropout
        )
        self.cross_attention = AttentionLayer(
            FullAttention(d_model, n_heads, dropout, use_rope=use_rope, rope_base=rope_base),
            d_model, n_heads, dropout
        )
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x: Tensor, memory: Tensor, 
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                rope_offset: int = 0) -> Tensor:
        # 自注意力
        x = self.self_attention(x, x, x, tgt_mask, rope_offset=rope_offset)
        
        # 交叉注意力
        x = self.cross_attention(x, memory, memory, memory_mask)
        
        # 前馈网络
        x = self.feed_forward(x)
        
        return x


class Decoder(nn.Module):
    """
    解码器
    
    多层解码器，堆叠多个解码器层。
    """
    
    def __init__(self, num_layers: int, d_model: int, n_heads: int, d_ff: int, 
                 dropout: float = 0.1, use_rope: bool = False, rope_base: int = 10000):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model, n_heads, d_ff, dropout,
                use_rope=use_rope,
                rope_base=rope_base
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                rope_offset: int = 0) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask, rope_offset=rope_offset)
        return self.norm(x)