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
    CFC Block V3 (PhysFormer Optimized):
    集成 Stride(时间降维) + Bottleneck(特征降维) + Transposed Upsample(可学习上采样)
    """

    def __init__(self, d_model, d_ff, d_phys=32, dropout=0.1, stride=2):
        super(CfcBlock, self).__init__()
        self.d_model = d_model
        self.d_phys = d_phys
        self.stride = stride

        # --- 1. 降维投影 (Bottleneck Down) ---
        # [512 -> 64]
        self.down_project = nn.Linear(d_model, d_phys)

        # --- 2. CfC 核心层 (在低维 d_phys 上运行) ---
        self.x_backbone = nn.Linear(d_phys, d_phys)
        self.h_backbone = nn.Linear(d_phys, d_phys)
        self.x_time_a = nn.Linear(d_phys, d_phys)
        self.h_time_a = nn.Linear(d_phys, d_phys)
        self.x_time_b = nn.Linear(d_phys, d_phys)
        self.h_time_b = nn.Linear(d_phys, d_phys)

        # --- [关键修改] 3. 可学习上采样 (Learnable Upsampling) ---
        if self.stride > 1:
            # Transposed Conv1d: [In_C, Out_C, Kernel, Stride]
            # Kernel=Stride 且 Stride=Stride 意味着无重叠的“块状”放大，就像铺瓷砖
            # 这样能最好地还原时序结构
            self.upsample_layer = nn.ConvTranspose1d(
                in_channels=d_phys,
                out_channels=d_phys,
                kernel_size=stride,
                stride=stride,
                padding=0,
                output_padding=0
            )
            # 初始化建议：Xavier Uniform 有助于打破初始对称性
            nn.init.xavier_uniform_(self.upsample_layer.weight)
            if self.upsample_layer.bias is not None:
                nn.init.constant_(self.upsample_layer.bias, 0)

        # --- 4. 升维投影 (Bottleneck Up) ---
        # [64 -> 512]
        self.up_project = nn.Linear(d_phys, d_model)

        # --- 5. 自适应门控 (Adaptive Gating) ---
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # 初始化 gate 偏置，使其初始倾向于保留原始特征 (Bias < 0)
        nn.init.constant_(self.gate_net[0].bias, -2.0)

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        b, s, d = x.shape
        residual = x

        # 1. 特征降维 [B, S, 512] -> [B, S, 64]
        x_phys = self.down_project(x)

        # 2. 时间下采样 (Stride)
        if self.stride > 1:
            # 简单的切片下采样
            x_input = x_phys[:, ::self.stride, :]
            effective_timespan = float(self.stride)
        else:
            x_input = x_phys
            effective_timespan = 1.0

        # 3. CfC 预计算
        x_ff1_seq = self.x_backbone(x_input)
        x_ta_seq = self.x_time_a(x_input)
        x_tb_seq = self.x_time_b(x_input)

        h_init = torch.zeros(b, self.d_phys, device=x.device, dtype=x.dtype)

        # 4. JIT 循环求解 ODE
        # output_short: [Batch, S_short, d_phys]
        output_short = cfc_rnn_scan(
            x_ff1_seq, x_ta_seq, x_tb_seq,
            h_init,
            self.h_backbone.weight, self.h_backbone.bias,
            self.h_time_a.weight, self.h_time_a.bias,
            self.h_time_b.weight, self.h_time_b.bias,
            effective_timespan
        )

        # --- [关键修改] 5. 转置卷积上采样 ---
        if self.stride > 1:
            # ConvTranspose1d 需要输入维度为 [Batch, Channels, Length]
            # Permute: [B, S_short, C] -> [B, C, S_short]
            feat_in = output_short.permute(0, 2, 1)

            # 执行上采样 -> [B, C, S_upsampled]
            feat_out = self.upsample_layer(feat_in)

            # Permute back: [B, S_upsampled, C]
            output_phys_upsampled = feat_out.permute(0, 2, 1)

            # 安全截断/补全：处理整除带来的长度微小差异
            # 比如 seq_len=97, stride=2, down=48, up=96 -> 少1个
            # 或者 seq_len=96, stride=2, down=48, up=96 -> 刚好
            curr_len = output_phys_upsampled.shape[1]
            if curr_len > s:
                # 如果长了，截断
                output_phys_low = output_phys_upsampled[:, :s, :]
            elif curr_len < s:
                # 如果短了，补零 (极少情况)
                pad_len = s - curr_len
                output_phys_low = F.pad(output_phys_upsampled, (0, 0, 0, pad_len))
            else:
                output_phys_low = output_phys_upsampled
        else:
            output_phys_low = output_short

        # 6. 特征升维 [B, S, 64] -> [B, S, 512]
        x_phys_evolved = self.up_project(output_phys_low)

        # 7. 门控融合
        concat_feat = torch.cat([residual, x_phys_evolved], dim=-1)
        gate = self.gate_net(concat_feat)
        final_out = gate * x_phys_evolved + (1.0 - gate) * residual

        return self.norm(self.dropout(final_out))