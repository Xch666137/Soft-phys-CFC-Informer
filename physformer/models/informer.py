"""
Informer模型核心实现

基于"Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"论文的完整实现。
此模块保持原始功能不变，所有组件已模块化分解。

论文要点：
- ProbSparse Attention: 复杂度从O(n²)降至O(n log n)
- 教师强制机制: 结合历史标签和预测部分
- 长序列预测: 支持超长输入序列
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from ..layers.attention import ProbAttention, FullAttention
from ..layers.encoder import Encoder
from ..layers.decoder import Decoder
from ..layers.embedding import DataEmbedding
from ..layers.mask import TriangularCausalMask


class Informer(nn.Module):
    """
    Informer 模型主体

    参数:
        configs: 配置对象 (Namespace), 包含 enc_in, d_model, n_heads 等所有超参数
    """
    
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.time_feat_dim = getattr(configs, "time_feat_dim", 10)

        # 根据 attn 参数决定使用哪一个类
        AttnClass = ProbAttention if configs.attn == 'prob' else FullAttention

        # --- 1. Embedding 层 ---
        # 编码器嵌入: Value + Position + Temporal
        self.enc_embedding = DataEmbedding(
            c_in=configs.enc_in,
            d_model=configs.d_model,
            embed_type=configs.embed,
            freq=configs.freq,
            dropout=configs.dropout,
            time_enc_in=self.time_feat_dim
        )
        # 解码器嵌入
        self.dec_embedding = DataEmbedding(
            c_in=configs.dec_in,
            d_model=configs.d_model,
            embed_type=configs.embed,
            freq=configs.freq,
            dropout=configs.dropout,
            time_enc_in=self.time_feat_dim
        )

        # --- 2. Encoder ---
        self.encoder = Encoder(
            num_layers=configs.e_layers,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            attn_cls=AttnClass,
            dropout=configs.dropout,
            use_distillation=configs.distil,  # 你的 encoder.py 需支持此参数
            use_rope=configs.use_rope,  # 你的 attention.py 需支持此参数
            rope_base=configs.rope_base
        )

        # --- 3. Decoder ---
        self.decoder = Decoder(
            num_layers=configs.d_layers,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            use_rope=configs.use_rope,
            rope_base=configs.rope_base
        )

        # --- 4.0 Output Projection ---
        # 将 d_model 维度映射回 c_out
        self.output_projection = nn.Linear(configs.d_model, configs.c_out)

        # # --- 4.1 Multi-Head Output Projection ---
        # # 输出层解耦，将单一线性映射拆分成多头(load、pv、wind)
        # # 假设 c_out=3 (顺序: Load, PV, Wind)，如果不是3，请根据实际情况调整
        #
        # # Head A: Load (负荷)
        # # 负荷相对平滑，使用 GELU 激活
        # self.head_load = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),
        #     nn.GELU(),
        #     nn.Linear(d_model // 2, 1)
        # )
        #
        # # Head B: PV (光伏)
        # # 物理约束: 光伏输出恒 >= 0
        # # 最后一层加 ReLU，从结构上解决"夜间负值"问题
        # self.head_pv = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),
        #     nn.ReLU(),
        #     nn.Linear(d_model // 2, 1),
        #     nn.ReLU()  # <--- 物理硬约束
        # )
        #
        # # Head C: Wind (风电)
        # # 风电波动大，加入 Dropout 增加鲁棒性
        # self.head_wind = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model // 2, 1)
        # )

        # 初始化权重
        self._init_weights()


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None) -> Tensor:
        """
        前向传播

        Args:
            x_enc: Encoder 输入数据 [Batch, Seq_Len, Enc_In]
            x_mark_enc: Encoder 时间特征 [Batch, Seq_Len, 5]
            x_dec: Decoder 输入数据 [Batch, Label_Len + Pred_Len, Dec_In]
            x_mark_dec: Decoder 时间特征 [Batch, Label_Len + Pred_Len, 5]
            enc_self_mask: Encoder 自注意力掩码 (通常为 None)
            dec_self_mask: Decoder 自注意力掩码 (通常需生成因果掩码)
            dec_enc_mask: Decoder-Encoder 交叉注意力掩码 (通常为 None)
        """
        # 1. Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        # 2. Encoder
        enc_out = self.encoder(enc_out, mask=enc_self_mask)

        # 3. Decoder
        # 自动生成因果掩码 (如果未提供)
        if dec_self_mask is None:
            # 动态获取当前的 batch_size 和 序列长度
            # 这样无论你的 label_len/pred_len 怎么变，维度永远是对的
            B, L_dec, _ = dec_out.shape

            # 生成因果掩码 [B, 1, L, L]
            # 1 代表保留，0 代表遮挡
            dec_self_mask = TriangularCausalMask(B, L_dec, device=dec_out.device).mask

        # 调用 Decoder
        # 注意：rope_offset 通常用于增量推理，训练时默认为 0
        dec_out = self.decoder(
            dec_out, enc_out,
            tgt_mask=dec_self_mask,
            memory_mask=dec_enc_mask,
            rope_offset=0
        )

        # 4. Projection
        output = self.output_projection(dec_out)

        # 4.1 Multi-Head Projection
        # 分别预测三个特征
        # out_load = self.head_load(dec_out)  # [B, L, 1]
        # out_pv = self.head_pv(dec_out)  # [B, L, 1]
        # out_wind = self.head_wind(dec_out)  # [B, L, 1]

        # 拼接回 [batch, seq_len, 3]
        # output = torch.cat([out_load, out_pv, out_wind], dim=-1)

        # 5. Output Formatting
        # 只需要返回预测的部分 (最后 pred_len 长度)
        if self.output_attention:
            return output[:, -self.pred_len:, :], None
        else:
            return output[:, -self.pred_len:, :]

    def _init_weights(self):
        """
        融合优化后的初始化策略：
        1. 区分 Linear 和 Conv1d
        2. 区分输出层和内部层 (gain控制)
        3. 显式初始化 Bias 为 0
        """
        for name, module in self.named_modules():
            # 1. 处理 Linear 层 和 Conv1d 层 (Embedding 和 Distillation 用到了 Conv1d)
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                # 区分输出层和普通层
                # 注意：self.output_projection 的 name 里包含 'output'
                if 'output' in name or 'projection' in name or 'head' in name:
                    gain = 1.0
                else:
                    # 使用较小的 gain 有助于深层 Transformer 的稳定性
                    gain = 0.5

                    # 使用 Xavier Normal 初始化权重
                if hasattr(module, 'weight') and module.weight is not None:
                    # 对于 Conv1d，xavier_normal 依然有效
                    nn.init.xavier_normal_(module.weight, gain=gain)

                # 显式将 Bias 置为 0
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)

            # 2. 特殊处理 LayerNorm (可选，但 PyTorch 默认就是 1 和 0，通常不需要动)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

            # 3. 如果你的位置编码是可学习的 (Learnable)，需要初始化
            # 如果是 Fixed (Sin/Cos)，则不需要这一步，因为它是 buffer 不是 parameter
            if hasattr(self, 'enc_embedding') and \
                    hasattr(self.enc_embedding, 'position_embedding') and \
                    hasattr(self.enc_embedding.position_embedding, 'pe'):
                # 仅当 pe 是 nn.Parameter 时执行，如果是 buffer 则跳过
                pe = self.enc_embedding.position_embedding.pe
                if isinstance(pe, nn.Parameter):
                    nn.init.normal_(pe, mean=0.0, std=0.02)
