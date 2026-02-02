import torch
import torch.nn as nn
import torch.nn.functional as F
from ..informer.attention import ProbAttention, FullAttention
from ..informer.encoder import Encoder
# from ..informer.decoder import Decoder
from .decoder_ode import PhysDecoder
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
                 output_attention=False, distil=True, mix=True, d_phys=64, stride=1,
                 device=torch.device('cuda:0'), use_rope=False, rope_base=10000):

        super(PhysFormer, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.output_attention = output_attention

        # --- 1. 双流 Embedding 层 ---

        # A. 统计流 Embedding (Stat Stream)
        # 输入: [Load, PV, Wind] -> c_in=3
        # 作用: 包含 Token(Value) + Position + Temporal(Time)
        self.stat_embedding = DataEmbedding(
            c_in=3,  # <--- 固定为3 (Load, PV, Wind)
            d_model=d_model,
            embed_type=embed,
            freq=freq,
            dropout=dropout
        )

        # B. 物理流 Embedding (Phys Stream)
        # 输入: [Temp, Irr, Speed, ΔLoad, ΔPV, ΔWind] -> c_in=6
        # 作用: 包含 Token(Value) + Position + Temporal(Time)
        # 注：给物理流加上时间嵌入也有助于它学习日照规律
        self.phys_embedding = DataEmbedding(
            c_in=6,  # <--- 固定为6 (3 Weather + 3 Diff)
            d_model=d_model,
            embed_type=embed,
            freq=freq,
            dropout=dropout
        )

        # C. Decoder Embedding (保持不变)
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
            d_phys=d_phys,
            dropout=dropout,
            stride=stride
        )


        # --- 4. Decoder (Transformer Stream) ---
        # self.decoder = Decoder(
        #     num_layers=d_layers,
        #     d_model=d_model,
        #     n_heads=n_heads,
        #     d_ff=d_ff,
        #     dropout=dropout,
        #     use_rope=use_rope,
        #     rope_base=rope_base
        # )

        # --- 4. Decoder (ODE Stream) ---
        self.decoder = PhysDecoder(
            num_layers=d_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            d_phys=d_phys,  # 确保这是一个合理的物理隐层维度，建议 64 或 32
            dropout=dropout,
            stride=1  # Decoder 必须保持逐点分辨率，不要下采样
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

    def forward(self, x_stat, x_phys, x_mark_enc, x_dec, x_mark_dec,
                enc_mask=None, dec_mask=None):
        """
        x_stat: [Batch, Seq, 3]  (Load, PV, Wind)
        x_phys: [Batch, Seq, 6]  (Weather + Diff)
        x_mark_enc: [Batch, Seq, 4/8] (Time Features)
        """

        # --- Step 1: Embedding (解耦) ---

        # 统计流: 加上了位置编码和时间特征 [B, S, 3] -> [B, S, D]
        enc_out_stat = self.stat_embedding(x_stat, x_mark_enc)

        # 物理流: 同样加上位置和时间 [B, S, 6] -> [B, S, D]
        # (时间特征 x_mark_enc 是共享的，这很合理)
        enc_out_phys_base = self.phys_embedding(x_phys, x_mark_enc)

        # --- Step 2: 双流并行处理 (Dual-Stream) ---
        # Path A: 统计流 (Informer Encoder)
        # 负责提取长程依赖、周期性模式
        enc_out_stat = self.encoder(enc_out_stat, mask=enc_mask)

        # Path B: 物理流 (Optimized CFC)
        # 负责提取连续时间动力学
        enc_out_phys = self.physics_adapter(enc_out_phys_base)

        # --- Step 3: 特征融合 (Fusion) ---
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
        # 注意：ODE Decoder 不需要 dec_mask (sequence masking)，因为它内部是递归的
        # 但 Cross-Attention 可能需要 memory_mask (通常不需要，除非 Encoder 有 padding)
        dec_in = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_in,
            enc_out_fused,
            tgt_mask=None,  # 显式传入 None，强调不再需要 Attention Mask
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