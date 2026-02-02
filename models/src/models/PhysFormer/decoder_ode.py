import torch
import torch.nn as nn
from ..informer.attention import FullAttention
from ..informer.encoder import AttentionLayer
from .cfc import CfcBlock  # 复用你已有的 CfcBlock


class PhysDecoderLayer(nn.Module):
    """
    Phys-ODE Decoder Layer
    结构：CfcBlock (Self-Dynamics) -> Cross-Attention (External Forcing) -> FFN (Flow Field)
    """

    def __init__(self, d_model, n_heads, d_ff, d_phys=64, dropout=0.1, stride=1):
        super(PhysDecoderLayer, self).__init__()

        # 1. Self-Dynamics: 替代 Masked Self-Attention
        # 使用 CfcBlock 模拟系统的惯性与内部演化
        # stride=1 保证输出的时间分辨率不丢失
        self.self_dynamics = CfcBlock(
            d_model=d_model,
            d_ff=d_ff,
            d_phys=d_phys,
            dropout=dropout,
            stride=stride
        )

        # 2. External Forcing: Cross-Attention
        # 从 Encoder (环境/历史) 获取驱动力
        self.external_forcing = AttentionLayer(
            FullAttention(d_model, n_heads, dropout=dropout, output_attention=False),
            d_model, n_heads, dropout
        )

        # 3. Vector Field Update: FFN
        # 模拟 ODE 的欧拉积分步
        self.vector_field = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Norm Layers
        # CfcBlock 内部已有 Norm，但为了连接 CrossAttn，这里再加一层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # x: [Batch, Pred_Len, D]

        # Step A: Self-Dynamics (Internal Evolution)
        # 用 Cfc 模拟连续演化。因为是 ODE，天然 Causal，不需要 tgt_mask
        x_evolved = self.self_dynamics(x)
        x = x + self.dropout(x_evolved)
        x = self.norm1(x)

        # Step B: External Forcing (Environment Injection)
        # memory 是 Encoder 的输出 (双流融合后的特征)
        x_forcing = self.external_forcing(
            x, memory, memory, memory_mask
        )
        x = x + x_forcing
        x = self.norm2(x)

        # Step C: Euler Integration Step (FFN)
        delta_x = self.vector_field(x)
        x = x + delta_x
        x = self.norm3(x)

        return x


class PhysDecoder(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, d_ff, d_phys=64, dropout=0.1, stride=1):
        super(PhysDecoder, self).__init__()
        self.layers = nn.ModuleList([
            PhysDecoderLayer(
                d_model, n_heads, d_ff, d_phys, dropout, stride
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.norm(x)