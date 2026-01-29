import torch
import torch.nn as nn
import torch.nn.functional as F
from ..informer.attention import ProbAttention, FullAttention
from ..informer.encoder import Encoder
from ..informer.decoder import Decoder
from ..informer.Embedding import DataEmbedding
from .cfc import CfcBlock


class PhysFormer(nn.Module):
    """
    PhysFormer Architecture (V5)

    结构特点：
    1. Physics Enhanced: 保留 CFC 物理层处理非线性动力学。
    2. Clean Output: 只输出 [Load, PV, Wind] 3个核心变量，移除噪声干扰。

    融合特性：
    1. CFC Physics Adapter: 输入端物理动力学特征提取
    2. Multi-Head Output: 输出端针对 Load/PV/Wind 的独立解耦预测
    """

    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0'), use_rope=False, rope_base=10000):

        super(PhysFormer, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.output_attention = output_attention

        # --- 1. Embedding 层 ---
        self.enc_embedding = DataEmbedding(
            c_in=enc_in,
            d_model=d_model,
            embed_type=embed,
            freq=freq,
            dropout=dropout
        )

        self.dec_embedding = DataEmbedding(
            c_in=dec_in,
            d_model=d_model,
            embed_type=embed,
            freq=freq,
            dropout=dropout
        )

        # --- 2. Encoder (Transformer Stream) ---
        # 根据 attn 参数决定使用哪一个类
        Attn = ProbAttention if attn == 'prob' else FullAttention

        self.encoder = Encoder(
            num_layers=e_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            attn_cls=Attn,
            dropout=dropout,
            use_distillation=distil,
            use_rope=use_rope,
            rope_base=rope_base
        )

        # --- 3. 物理动力学注入层 (CFC Stream) ---
        # stride=1 保证物理层捕捉最精细的动力学特征
        self.physics_adapter = CfcBlock(
            d_model, d_ff,
            d_phys=64,
            dropout=dropout,
            stride=2
        )


        # --- 4. Encoder (Transformer Stream) ---
        # 严格使用 uploaded: decoder.py 中的 Decoder 类
        self.decoder = Decoder(
            num_layers=d_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            use_rope=use_rope,
            rope_base=rope_base
        )


        # --- 5. Fusion Layer (融合层) ---
        # 将 Attention 的特征和 Physics 的特征融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # --- 6. Multi-Head Output ---
        # Head A: Load
        self.head_load = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # Head B: PV
        self.head_pv = nn.Sequential(
            nn.Linear(d_model, d_model),  # input dim change
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # Head C: Wind
        self.head_wind = nn.Sequential(
            nn.Linear(d_model, d_model),  # input dim change
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_mask=None, dec_mask=None):

        # --- Step 1: Embedding (共享) [B, S, D] ---
        enc_base = self.enc_embedding(x_enc, x_mark_enc)

        # --- Step 2: 双流并行处理 (Dual-Stream) ---
        # Path A: 统计流 (Informer Encoder)
        # 负责提取长程依赖、周期性模式
        enc_out_stat = self.encoder(enc_base, mask=enc_mask)

        # Path B: 物理流 (Optimized CFC)
        # 负责提取连续时间动力学、平滑突变
        # 因为 CfC 内部有 Gate，它会自动决定保留多少原始信息，所以直接传 enc_base 没问题
        enc_out_phys = self.physics_adapter(enc_base)


        # --- Step 3: 特征融合 (Fusion) ---
        # 将 "统计特征" 和 "物理特征" 结合
        # [B, S, D] + [B, S, D] -> [B, S, 2D] -> [B, S, D]
        if enc_out_stat.shape[1] != enc_out_phys.shape[1]:
            enc_out_phys_aligned = enc_out_phys.permute(0, 2, 1)
            enc_out_phys_aligned = F.interpolate(
                enc_out_phys_aligned,
                size=enc_out_stat.shape[1],
                mode='linear',
                align_corners=False
            )
            enc_out_phys_aligned = enc_out_phys_aligned.permute(0, 2, 1)
        else:
            enc_out_phys_aligned = enc_out_phys

        combined_enc = torch.cat([enc_out_stat, enc_out_phys_aligned], dim=-1)
        enc_out_fused = self.fusion_layer(combined_enc)

        # --- Step 4: Decoder ---
        dec_in = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_in,
            enc_out_fused,
            tgt_mask=dec_mask,
            memory_mask=None
        )

        # dec_out shape: [Batch, Pred_Len, d_model]

        # --- Step 5: Flatten Projection (Time Domain Mapping) ---
        # 分别通过三个独立的 MLP 头
        out_load = self.head_load(dec_out)  # [B, P, 1]
        out_pv = self.head_pv(dec_out)  # [B, P, 1]
        out_wind = self.head_wind(dec_out)  # [B, P, 1]

        # 拼接输出
        output = torch.cat([out_load, out_pv, out_wind], dim=-1)  # [B, P, 3]
        output = output[:, -self.pred_len:, :]  # [Batch, 192, 3] -> [Batch, 96, 3]

        if self.output_attention:
            return output, enc_out_phys
        return output

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 使用 Kaiming Normal 适配 GELU
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)