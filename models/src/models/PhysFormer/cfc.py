import torch
import torch.nn as nn
import torch.jit as jit
import torch.nn.functional as F  # 新增引用
from typing import List, Tuple


# --- 核心优化：将循环逻辑编译为 TorchScript ---
# (这部分 cfc_rnn_scan 函数保持不变，不用动)
@jit.script
def cfc_rnn_scan(
        x_ff1_seq: torch.Tensor,
        x_ta_seq: torch.Tensor,
        x_tb_seq: torch.Tensor,
        h_init: torch.Tensor,
        w_h_ff1: torch.Tensor, b_h_ff1: torch.Tensor,
        w_h_ta: torch.Tensor, b_h_ta: torch.Tensor,
        w_h_tb: torch.Tensor, b_h_tb: torch.Tensor,
        timespan: float
) -> torch.Tensor:
    h_state = h_init
    output_list: List[torch.Tensor] = []
    seq_len = x_ff1_seq.size(1)

    for t in range(seq_len):
        x_ff1 = x_ff1_seq[:, t, :]
        x_ta = x_ta_seq[:, t, :]
        x_tb = x_tb_seq[:, t, :]

        h_ff1 = torch.matmul(h_state, w_h_ff1.t()) + b_h_ff1
        h_ta = torch.matmul(h_state, w_h_ta.t()) + b_h_ta
        h_tb = torch.matmul(h_state, w_h_tb.t()) + b_h_tb

        ff1 = torch.tanh(x_ff1 + h_ff1)
        t_a = x_ta + h_ta
        t_b = x_tb + h_tb

        gate = torch.sigmoid(t_a * timespan + t_b)
        h_state = ff1 * (1.0 - gate) + h_state * gate

        output_list.append(h_state)

    return torch.stack(output_list, dim=1)


class CfcCell(nn.Module):
    # (保持不变)
    def __init__(self, input_size, hidden_size):
        super(CfcCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.backbone = nn.Linear(input_size + hidden_size, hidden_size)
        self.time_a = nn.Linear(input_size + hidden_size, hidden_size)
        self.time_b = nn.Linear(input_size + hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, h_prev, timespan):
        cat_input = torch.cat([input, h_prev], dim=-1)
        ff1 = self.tanh(self.backbone(cat_input))
        t_a = self.time_a(cat_input)
        t_b = self.time_b(cat_input)
        gate = self.sigmoid(t_a * timespan + t_b)
        h_new = ff1 * (1.0 - gate) + h_prev * gate
        return h_new


class CfcBlock(nn.Module):
    """
    CFC Block V2:
    集成 Stride(时间降维) + Bottleneck(特征降维) + Gated Fusion(门控融合)
    """

    def __init__(self, d_model, d_ff, d_phys=32, dropout=0.1, stride=4):
        # 新增 d_phys 参数，默认设为 64 或 32，远小于 d_model(512)
        super(CfcBlock, self).__init__()
        self.d_model = d_model
        self.d_phys = d_phys  # 物理核心状态维度
        self.stride = stride

        # --- 1. 降维投影 (Bottleneck Down) ---
        # 将高维统计特征压缩为低维物理状态 [512 -> 64]
        self.down_project = nn.Linear(d_model, d_phys)

        # --- 2. CfC 核心层 (在低维 d_phys 上运行) ---
        # 这里的 input_size 和 hidden_size 都是 d_phys
        self.x_backbone = nn.Linear(d_phys, d_phys)
        self.h_backbone = nn.Linear(d_phys, d_phys)

        self.x_time_a = nn.Linear(d_phys, d_phys)
        self.h_time_a = nn.Linear(d_phys, d_phys)

        self.x_time_b = nn.Linear(d_phys, d_phys)
        self.h_time_b = nn.Linear(d_phys, d_phys)

        # --- 3. 升维投影 (Bottleneck Up) ---
        # 将物理演化结果恢复回高维 [64 -> 512]
        self.up_project = nn.Linear(d_phys, d_model)

        # --- 4. 自适应门控 (Adaptive Gating) ---
        # 决定多少信息来自物理层，多少来自原始统计特征
        # 输入是 [Original(512) + Physics(512)] -> 输出 [Mask(512)]
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # 初始化gate的权重，让其初始为0(little tips)
        nn.init.constant_(self.gate_net[0].bias, -3.0)

    def forward(self, x):
        # x: [Batch, Seq_Len, D_model] (例如: B, 672, 512)
        b, s, d = x.shape

        # 保存原始输入用于残差/门控
        residual = x

        # --- Step 1: 降维 (Feature Compression) ---
        # [B, S, 512] -> [B, S, 64]
        # 这一步极大减少了后续 ODE 的参数量
        x_phys = self.down_project(x)

        # --- Step 2: 时间步长降采样 (Time Downsampling - 你的逻辑) ---
        if self.stride > 1:
            # [B, S, 64] -> [B, S/4, 64]
            x_input = x_phys[:, ::self.stride, :]
            effective_timespan = float(self.stride)
        else:
            x_input = x_phys
            effective_timespan = 1.0

        # --- Step 3: CfC 预计算 ---
        # 计算量从 (512^2 * S) 降低到了 (64^2 * S/4) -> 理论加速约 256倍
        x_ff1_seq = self.x_backbone(x_input)
        x_ta_seq = self.x_time_a(x_input)
        x_tb_seq = self.x_time_b(x_input)

        # 初始化隐藏状态 (注意维度是 d_phys)
        h_init = torch.zeros(b, self.d_phys, device=x.device, dtype=x.dtype)

        # --- Step 4: JIT 循环 ---
        output_short = cfc_rnn_scan(
            x_ff1_seq, x_ta_seq, x_tb_seq,
            h_init,
            self.h_backbone.weight, self.h_backbone.bias,
            self.h_time_a.weight, self.h_time_a.bias,
            self.h_time_b.weight, self.h_time_b.bias,
            effective_timespan
        )

        # --- Step 5: 上采样恢复 (Upsample) ---
        if self.stride > 1:
            output_short = output_short.permute(0, 2, 1)  # [B, 64, S_short]
            output_phys_low = F.interpolate(
                output_short, size=s, mode='linear', align_corners=False
            )  # [B, 64, S]
            output_phys_low = output_phys_low.permute(0, 2, 1)  # [B, S, 64]
        else:
            output_phys_low = output_short

        # --- Step 6: 升维 (Feature Expansion) ---
        # [B, S, 64] -> [B, S, 512]
        x_phys_evolved = self.up_project(output_phys_low)

        # --- Step 7: 门控融合 (Gated Fusion) ---
        # 这一步是关键：不要直接替换，而是融合
        concat_feat = torch.cat([residual, x_phys_evolved], dim=-1)
        gate = self.gate_net(concat_feat)  # [B, S, 512] 值为 0~1

        # 融合公式: Gate * Physics + (1-Gate) * Original
        # 这样模型可以自己学会：什么时候该听物理的，什么时候该听统计的
        final_out = gate * x_phys_evolved + (1.0 - gate) * residual

        return self.norm(self.dropout(final_out))