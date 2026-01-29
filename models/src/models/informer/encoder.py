"""
编码器实现

Informer中的编码器模块，包括编码器层和多头注意力。
添加自注意力蒸馏机制来减少序列长度和计算复杂度。
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from .attention import ProbAttention
from .distillation import SelfAttentionDistillation


class FeedForward(nn.Module):
    """
    前馈网络
    
    标准的前馈网络，包含两个线性层和ReLU激活函数。
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.dropout(self.net(x)))

    def _init_weights(self):
        """
        前馈网络专门初始化 (GELU 适配版)
        """
        # 第一层 (d_model -> d_ff):
        # 使用 Kaiming Normal (He Init)。
        # 虽然是 GELU，但 nonlinearity='relu' 是最接近的近似，能保持方差稳定。
        nn.init.kaiming_normal_(
            self.net[0].weight,
            mode='fan_in',
            nonlinearity='relu'
        )
        if self.net[0].bias is not None:
            nn.init.zeros_(self.net[0].bias)

        # 第二层 (d_ff -> d_model):
        # 使用 Xavier Normal (Glorot Init)。
        # 这一层是线性投影回 d_model，没有非线性激活紧随其后，Gain=1.0 即可。
        nn.init.xavier_normal_(
            self.net[3].weight,
            gain=1.0
        )
        if self.net[3].bias is not None:
            nn.init.zeros_(self.net[3].bias)


class AttentionLayer(nn.Module):
    """
    注意力层
    
    包装注意力机制和层归一化。
    """
    
    def __init__(self, attention: nn.Module, d_model: int, n_heads: int, 
                 dropout: float = 0.1):
        super().__init__()
        self.attention = attention
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, 
                mask: Optional[Tensor] = None, rope_offset: int = 0) -> Tensor:
        if hasattr(self.attention, 'forward') and 'rope_offset' in self.attention.forward.__code__.co_varnames:
            outputs = self.attention(query, key, value, mask, rope_offset)
        else:
            outputs = self.attention(query, key, value, mask)

        if isinstance(outputs, tuple):
            attended = outputs[0]  # 只取 output, 丢弃 attn_weights
        else:
            attended = outputs

        # 残差连接 + 归一化
        return self.norm(query + self.dropout(attended))


class EncoderLayer(nn.Module):
    """
    编码器层，支持RoPE
    
    单个编码器层，包含自注意力和前馈网络。
    可选添加自注意力蒸馏机制。
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, attn_cls,
                 dropout: float = 0.1, use_distillation: bool = False,
                 use_rope: bool = False, rope_base: int = 10000):
        super().__init__()
        self.use_distillation = use_distillation
        self.self_attention = AttentionLayer(
            attn_cls(d_model, n_heads, dropout, use_rope=use_rope, rope_base=rope_base),
            d_model, n_heads, dropout
        )
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        if use_distillation:
            self.distillation = SelfAttentionDistillation(d_model)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.self_attention(x, x, x, mask)
        x = self.feed_forward(x)
        
        # 应用蒸馏（如果启用）
        if self.use_distillation:
            x = self.distillation(x)
            
        return x


class Encoder(nn.Module):
    """
    编码器
    
    多层编码器，堆叠多个编码器层。
    支持在特定层间添加自注意力蒸馏来减少序列长度。
    """
    
    def __init__(self, num_layers: int, d_model: int, n_heads: int, d_ff: int,
                 attn_cls, dropout: float = 0.1,
                 use_distillation: bool = True,
                 use_rope: bool = False, rope_base: int = 10000):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model, n_heads, d_ff,
                attn_cls=attn_cls,
                dropout=dropout,
                use_distillation=use_distillation,
                use_rope=use_rope,
                rope_base=rope_base
            )
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)