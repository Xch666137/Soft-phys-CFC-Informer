import torch
import torch.nn as nn
from ..informer.attention import ProbAttention, FullAttention
from ..informer.encoder import Encoder
from ..informer.Embedding import DataEmbedding
from .Causal_coupling import PhysicsGuidedCausalCoupling
from .physical_layer import ExplicitPhysicalMapping
from .Flatten_head import FlattenHead


class PhysFormer(nn.Module):
    """
    PhysFormer-Quantile (Encoder-Only Version)

    结构：
    1. Input Embedding (Stat + Phys)
    2. Dual-Stream Encoder (Stat Transformer + Explicit Physics)
    3. Causal Coupling Fusion (History Fusion)
    4. Flatten Head Projection (History -> Future Features)
    5. Future Physics Guidance (Gray-box Constraint)
    6. Quantile Output Heads (Residual Learning)
    """

    def __init__(self, enc_in, seq_len, pred_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.1, attn='prob', embed='fixed', freq='h',
                 activation='gelu', use_rope=False, rope_base=10000,
                 weather_mean=None, weather_std=None,
                 target_mean=None, target_std=None, distil=False,
                 device=torch.device('cuda:0')):

        super(PhysFormer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        # --- 1. 双流 Embedding 层 ---

        # A. 统计流 Embedding (Stat Stream)
        self.stat_embedding = DataEmbedding(
            c_in=enc_in,
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


        # --- 2. 显式物理映射层 ---
        # 负责计算 "理论基准"
        self.phys_layer = ExplicitPhysicalMapping(
            d_model=d_model,
            weather_dim=3,
            weather_mean=weather_mean,
            weather_std=weather_std,
            target_mean=target_mean,
            target_std=target_std
        )

        # --- 3. Causal Coupling (因果融合层) ---
        self.causal_coupling = PhysicsGuidedCausalCoupling(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )

        # --- 4. Flatten Head (Encoder -> Future Projection) ---
        self.flatten_head = FlattenHead(seq_len, d_model, pred_len, dropout)

        # 6. Advanced Multi-Head Output
        # 天气特征融合层
        # 使用GLU门控模块将 Flatten 出来的历史特征与未来天气特征更好地混合
        self.weather_fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.weather_fusion_proj = nn.Linear(d_model * 2, d_model)

        # 共享投影：提取天气对所有功率的共性影响
        self.shared_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Head A: Load (相对简单，浅层网络)
        # 输入: [B, P, D] -> 输出: [B, P, 1]
        self.head_load = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # Head B: PV (波动剧烈，深层网络)
        self.head_pv = nn.Sequential(
            nn.Linear(d_model, d_model),  # 保持维度，增加容量
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),  # 逐步压缩
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # Head C: Wind (混沌特性，深层网络)
        self.head_wind = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # 8. 权重初始化
        self.apply(self._init_weights)
        self._init_gating_mechanisms()

    def forward(self, x_stat, x_weather_hist, x_weather_future, x_mark_enc, alpha=0.5, return_gates=False):
        """
        Args:
            x_stat: [B, Seq, 3] 历史观测值 (Load, PV, Wind)
            x_weather_hist: [B, Seq, 3] 历史天气
            x_weather_future: [B, Pred, 3] 未来天气预报 (必不可少!)
            x_mark_enc: [B, Seq, 4] 时间特征

        Returns:
            output: [B, Pred, 3] 点预测结果
            gates_info: (Optional) 监控信息
        """

        # === Step 1: History Encoding ===
        # 早期融合：将历史功率与历史天气拼接
        # x_enc_input: [B, Seq, 6]
        x_enc_input = torch.cat([x_stat, x_weather_hist], dim=-1)

        enc_out_stat = self.stat_embedding(x_enc_input, x_mark_enc)
        enc_out_stat = self.encoder(enc_out_stat)  # [B, Seq, D]

        # === Step 2: History Physics ===
        phys_feat_hist, _ = self.phys_layer(x_weather_hist)  # [B, Seq, D]

        # === Step 3: Fusion (Physics-Guided Attention) ===
        if return_gates:
            enc_out_fused, reg_loss, gates_info = self.causal_coupling(
                enc_out_stat, phys_feat_hist, x_weather_hist, alpha=alpha, return_gates=True
            )
        else:
            enc_out_fused, reg_loss = self.causal_coupling(
                enc_out_stat, phys_feat_hist, x_weather_hist, alpha=alpha, return_gates=False
            )
            gates_info = None

        # === Step 4: Projection to Future ===
        # Flatten Head 将时间步映射到未来 [B, Seq, D] -> [B, Pred, D]
        # 此时 future_feat 仅包含历史信息的投影
        future_feat_history = self.flatten_head(enc_out_fused)  # [B, Pred, D]

        # === Step 5: Inject Future Weather (Fixing Residual Blind Spot) [MODIFIED] ===

        # A. 计算未来的理论物理值 (theory_future) 和 天气隐层特征 (weather_feat_future)
        # phys_layer 返回的第一个值是 projected_feat [B, Pred, D]
        weather_feat_future, theory_future = self.phys_layer(x_weather_future)

        # B. 门控融合(GLU)将未来天气特征注入到残差网络输入中
        # 这样 Residual Head 就能看到"未来是否阴天/大风"
        # 使用残差连接方式融合：History_Projection + Future_Weather_Context
        cat_feat = torch.cat([future_feat_history, weather_feat_future], dim=-1)
        fusion_gate = self.weather_fusion_gate(cat_feat)
        future_feat_combined = fusion_gate * self.weather_fusion_proj(cat_feat) + future_feat_history

        # 经过共享投影层
        future_feat = self.shared_projection(future_feat_combined)

        # === Step 6: Residual Prediction (使用差异化 Head) ===
        res_load = self.head_load(future_feat)  # [B, Pred, 1]
        res_pv = self.head_pv(future_feat)
        res_wind = self.head_wind(future_feat)

        # === Step 7: Physics Guidance & Combination ===
        theory_load = theory_future[:, :, 0].unsqueeze(-1)
        theory_pv = theory_future[:, :, 1].unsqueeze(-1)
        theory_wind = theory_future[:, :, 2].unsqueeze(-1)

        # Final = Theory + Residual
        final_load = theory_load + res_load
        final_pv = theory_pv + res_pv
        final_wind = theory_wind + res_wind

        output = torch.cat([final_load, final_pv, final_wind], dim=-1)

        if return_gates:
            return output, reg_loss, gates_info
        else:
            return output, reg_loss

    def _init_weights(self, m):
        """通用的权重初始化策略"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_gating_mechanisms(self):
        """针对 Gate 的特殊初始化，防止饱和"""
        coupling = self.causal_coupling
        gate_learners = [
            coupling.gate_learner_pv,
            coupling.gate_learner_wind,
            coupling.gate_learner_load
        ]

        for learner in gate_learners:
            last_linear = learner[-2]  # 假设结构最后是 Linear -> Sigmoid
            if isinstance(last_linear, nn.Linear):
                nn.init.normal_(last_linear.weight, mean=0, std=0.001)
                nn.init.constant_(last_linear.bias, 0)