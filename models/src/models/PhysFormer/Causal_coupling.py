import torch
import torch.nn as nn

from models.src.utils.losses import GateResponseRegularization


class PhysicsGuidedCausalCoupling(nn.Module):
    """
    PhysFormer 核心组件：物理引导的因果耦合模块 (最终版)
    融合了：
    1. 显式天气嵌入 (Explicit Weather Embedding)
    2. 课程学习机制 (Curriculum Learning): 从硬物理规则逐渐过渡到软学习
    3. 时序平滑约束 (Temporal Smoothing)
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1, smooth_kernel=3):
        super().__init__()

        self.d_model = d_model

        # ============================================================
        # Part 1: Query端子空间投影 (保留原版思想)
        # ============================================================
        # 让 Load/PV/Wind 从不同角度解析统计流 (Stat Stream)
        self.query_proj_load = nn.Linear(d_model, d_model)
        self.query_proj_pv = nn.Linear(d_model, d_model)
        self.query_proj_wind = nn.Linear(d_model, d_model)

        # 物理流 (Phys Stream) 的投影
        self.phys_proj_k = nn.Linear(d_model, d_model)  # Key
        self.phys_proj_v = nn.Linear(d_model, d_model)  # Value

        # ============================================================
        # Part 2: 显式天气嵌入 (用于增强 Query)
        # ============================================================
        # 将 [Temp, Irradiance, WindSpeed] 映射为高维 Condition
        self.weather_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)  # 映射到与特征同维度
        )

        # 天气信息注入门控 (Weather Gate)
        self.weather_gate_learner = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
        # 专门用于融合前的特征变换
        self.weather_transform = nn.Linear(d_model, d_model)

        # ============================================================
        # Part 3: 因果注意力 (Causal Attention) - 坚决保留！
        # ============================================================
        # Q = Stat + Weather (带着环境上下文去检索)
        # K, V = Phys
        self.attn_load = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.attn_pv = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.attn_wind = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

        # ============================================================
        # Part 4: 门控与保护 (Gate & Protection)
        # ============================================================
        # 注意力检索出的特征，需要经过 Gate 过滤才能融合
        # 输入: Stat_Subspace + Phys_Projected + Weather_Condition
        # 输出: Gate (0~1)
        # 为 PV 和 Wind 分别设计独立的 Gate Learner

        # PV Gate Learner (增加波动率特征输入)
        self.gate_learner_pv = nn.Sequential(
            nn.Linear(d_model * 3 + 1, d_model // 2),  # +1 for volatility feature
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )

        # Wind Gate Learner (增加波动率特征输入)
        self.gate_learner_wind = nn.Sequential(
            nn.Linear(d_model * 3 + 1, d_model // 2),  # +1 for volatility feature
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )

        # Load Gate Learner (增加波动率特征输入)
        self.gate_learner_load = nn.Sequential(
            nn.Linear(d_model * 3 + 1, d_model // 2),  # +1 for volatility feature
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )

        # 使用 Depthwise Conv 保证 Gate 不会突变，保留长程平滑性
        self.smoother = nn.Conv1d(
            d_model, d_model,
            kernel_size=smooth_kernel,
            padding=smooth_kernel // 2,
            groups=d_model  # 独立平滑每个通道
        )

        # ============================================================
        # Part 5: 融合与归纳偏置
        # ============================================================
        self.fusion_proj = nn.Linear(d_model * 3, d_model)

        self.net_balance_inductor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # ============================================================
        # Part 6: 输出与辅助
        # ============================================================
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # ===== 可学习的物理阈值 =====
        # 使用 logit 空间，通过 sigmoid 映射到合理范围
        self.irr_threshold_logit = nn.Parameter(torch.tensor(0.94))
        self.wind_threshold_logit = nn.Parameter(torch.tensor(-1.0))

        # 可学习的斜率
        self.irr_slope_log = nn.Parameter(torch.tensor(2.3))  # exp(2.3) ≈ 10
        self.wind_slope_log = nn.Parameter(torch.tensor(1.6))  # exp(1.6) ≈ 5

        self.gate_response_reg = GateResponseRegularization(weight=0.01)

        # self.register_buffer('step_counter', torch.tensor(0.0))
        # self.decay_steps = 2281 * 20  # ≈ 45,620 batches，epoch 20 时才完成内部软化

        # 监控用
        self.register_buffer('irr_threshold_ema', torch.tensor(-1.0))
        self.register_buffer('wind_threshold_ema', torch.tensor(-1.0))
        self.register_buffer('info_gain_ema', torch.tensor(1.0))

    def get_current_thresholds(self):
        """辅助函数：计算当前的实际阈值和斜率"""
        irr_threshold = -2.0 + 2.5 * torch.sigmoid(self.irr_threshold_logit)
        wind_threshold = -2.0 + 2.5 * torch.sigmoid(self.wind_threshold_logit)

        irr_slope = torch.exp(torch.clamp(self.irr_slope_log, 0, 4.6))
        wind_slope = torch.exp(torch.clamp(self.wind_slope_log, 0, 4.6))

        return irr_threshold, wind_threshold, irr_slope, wind_slope

    def get_hard_prior(self, x_weather):
        """生成物理掩码"""
        irr_threshold, wind_threshold, irr_slope, wind_slope = self.get_current_thresholds()

        # 假设 x_weather 是 StandardScaler 归一化的数据
        # 提取特征 [B, S, 1]
        irradiance = x_weather[:, :, 1].unsqueeze(-1)
        wind_speed = x_weather[:, :, 2].unsqueeze(-1)

        # 软截断 (Sigmoid Soft Gating)
        prior_pv = torch.sigmoid((irradiance - irr_threshold) * irr_slope)
        prior_wind = torch.sigmoid((wind_speed - wind_threshold) * wind_slope)
        prior_load = torch.ones_like(prior_pv)

        # 监控更新
        if self.training:
            with torch.no_grad():
                self.irr_threshold_ema = 0.99 * self.irr_threshold_ema + 0.01 * irr_threshold
                self.wind_threshold_ema = 0.99 * self.wind_threshold_ema + 0.01 * wind_threshold

        return prior_load, prior_pv, prior_wind

    def forward(self, stat_feat, phys_feat, x_weather, alpha=0.5, return_gates=False):
        """
        Args:
            stat_feat: [B, S, D] - 统计流 (来自Informer，包含长程依赖)
            phys_feat: [B, S, D] - 物理流 (来自CFC，包含动力学)
            return_gates: bool - 是否返回门控值用于可视化

        Returns:
            output: [B, S, D] - 耦合后的特征
            gates: dict (可选) - 三个门控的平均激活值
        """
        B, S, D = stat_feat.shape

        # ===== 将 Curriculum Learning 的 alpha 计算放在最优先 =====
        if self.training:
            # 冻结机制：前50%不更新阈值梯度，强制模型适应初始物理规则
            if alpha > 0.5:
                with torch.no_grad():
                    prior_load, prior_pv, prior_wind = self.get_hard_prior(x_weather)
            else:
                prior_load, prior_pv, prior_wind = self.get_hard_prior(x_weather)
        else:
            alpha = 0.1  # 测试时保留10%安全先验
            prior_load, prior_pv, prior_wind = self.get_hard_prior(x_weather)


        # ===== 计算历史天气波动率特征 =====
        # x_weather: [B, S, 3] (温度、辐照度、风速)
        # 计算时间维度方差，然后取均值得到每个样本的波动率
        volatility = torch.var(x_weather, dim=1).mean(dim=-1)  # [B]
        # 扩展为 [B, S, 1] 以匹配序列长度
        volatility_expanded = volatility.unsqueeze(-1).unsqueeze(-1).expand(-1, S, 1)  # [B, S, 1]

        # ===== Step 1: 准备 Q, K, V (引入退火门控) =====
        c_weather = self.weather_proj(x_weather)  # [B, S, D]

        # 计算软门控 (模型自己学的)
        weather_gate_soft = self.weather_gate_learner(torch.cat([stat_feat, c_weather], dim=-1))

        # [核心机制] 课程学习融合：初期强制开门(1.0)，后期信任模型(soft)
        weather_gate = alpha * 1.0 + (1 - alpha) * weather_gate_soft

        # 变换并注入
        gated_weather = weather_gate * self.weather_transform(c_weather)

        # Q = Stat_Proj + Weather_Proj
        # 这样 Stat 流在做 Attention 时，就知道"现在是晚上"，不会去强行匹配光伏特征
        q_load = self.query_proj_load(stat_feat) + gated_weather
        q_pv = self.query_proj_pv(stat_feat) + gated_weather
        q_wind = self.query_proj_wind(stat_feat) + gated_weather

        k_phys = self.phys_proj_k(phys_feat)
        v_phys = self.phys_proj_v(phys_feat)

        # ===== Step 2: 执行因果注意力 (Causal Attention) =====
        # Stat 主动去检索 Phys 中的有用信息
        # attn_out: [B, S, D]

        # ===== Step 2: 执行因果注意力 (Causal Attention) =====
        # 传入 attn_mask，并删除 is_causal=True (避免报错)
        attn_load, _ = self.attn_load(q_load, k_phys, v_phys)
        attn_pv, _ = self.attn_pv(q_pv, k_phys, v_phys)
        attn_wind, _ = self.attn_wind(q_wind, k_phys, v_phys)

        # ===== Step 3: 计算 Gate (决定检索结果的可信度) =====
        # 即使 Attention 检索到了东西，我们也需要 Gate 来决定是否通过
        # 这一步引入了 Curriculum Learning 和 Hard Prior

        # ===== Step 3: 计算 Gate (决定检索结果的可信度) =====
        gate_input_load = torch.cat([stat_feat, attn_load, c_weather, volatility_expanded], dim=-1)
        gate_input_pv = torch.cat([stat_feat, attn_pv, c_weather, volatility_expanded], dim=-1)
        gate_input_wind = torch.cat([stat_feat, attn_wind, c_weather, volatility_expanded], dim=-1)

        # 神经网络输出的软控制系数，经过 Sigmoid 范围是 [0, 1]
        gate_load_soft = self.gate_learner_load(gate_input_load)
        gate_pv_soft = self.gate_learner_pv(gate_input_pv)
        gate_wind_soft = self.gate_learner_wind(gate_input_wind)

        # -------------------------------------------------------------------
        # 严格物理门控 (Strict Physical Gating)
        # prior 决定绝对物理上限 (夜间严格为0)，soft_gate 决定日间的微调(0~1)
        # [PERFORMANCE FIX] prior_pv 本身是由 x_weather.unsqueeze(-1) 得来，所以自身已经是 [B, S, 1]
        # 直接使用乘法触发隐式广播机制，避免无谓的显存分配。
        # -------------------------------------------------------------------
        gate_pv = prior_pv * gate_pv_soft
        gate_wind = prior_wind * gate_wind_soft
        gate_load = gate_load_soft

        # 时序平滑 (不变)
        def apply_smoothing(g, max_val=1.5):
            smoothed = self.smoother(g.transpose(1, 2)).transpose(1, 2)
            return torch.clamp(smoothed, 0.0, max_val)

        gate_pv = apply_smoothing(gate_pv, max_val=1.5)
        gate_wind = apply_smoothing(gate_wind, max_val=1.5)
        gate_load = apply_smoothing(gate_load, max_val=1.0)

        # ===== Step 4: 物理特征融合 =====
        # 使用 Gate 对 Attention 的结果进行加权
        # 这是真正的 "Gated Attention"
        phys_load_final = gate_load * attn_load
        phys_pv_final = gate_pv * attn_pv
        phys_wind_final = gate_wind * attn_wind

        # ===== Step 5: 归纳偏置与输出 (Part 5 & 6) =====
        # 融合三路物理特征
        phys_fused = self.fusion_proj(
            torch.cat([phys_load_final, phys_pv_final, phys_wind_final], dim=-1)
        )

        # 净负荷平衡归纳
        phys_balanced = self.net_balance_inductor(phys_fused)

        # 残差连接
        output = self.norm(stat_feat + self.dropout(phys_balanced))

        # 监控
        if self.training:
            with torch.no_grad():
                info_gain = torch.norm(phys_balanced) / (torch.norm(stat_feat) + 1e-8)
                self.info_gain_ema = 0.99 * self.info_gain_ema + 0.01 * info_gain

        # ===== [关键修复] 计算阈值正则化 Loss =====
        # 我们将其放入 gates 字典返回，在外部 loop 中加到 total loss
        reg_loss = 0.0
        pv_corr, wind_corr = 0.0, 0.0
        if self.training:
            # 原有的阈值正则化
            irr_t, wind_t, _, _ = self.get_current_thresholds()
            reg_loss += 0.01 * ((irr_t - (-0.2)) ** 2 + (wind_t - (-1.0)) ** 2)

            loss_corr_pv, pv_corr = self.gate_response_reg(
                gate_pv_soft, prior_pv
            )
            loss_corr_wind, wind_corr = self.gate_response_reg(
                gate_wind_soft, prior_wind
            )
            reg_loss += (loss_corr_pv + loss_corr_wind)

        if return_gates:
            gate_pv_seq = gate_pv.mean(dim=-1).detach().cpu().numpy()
            gate_wind_seq = gate_wind.mean(dim=-1).detach().cpu().numpy()
            irr_seq = x_weather[:, :, 1].detach().cpu().numpy()
            speed_seq = x_weather[:, :, 2].detach().cpu().numpy()

            gates = {
                'load': gate_load.mean().item(),
                'pv': gate_pv.mean().item(),
                'wind': gate_wind.mean().item(),
                'weather_gate': weather_gate.mean().item(),
                'info_gain': self.info_gain_ema.item(),
                'gate_reg_loss': reg_loss,  # 返回给外部
                'pv_corr': pv_corr,  # 用于日志监控
                'wind_corr': wind_corr,  # 用于日志监控
                'pv_seq_batch': gate_pv_seq,
                'wind_seq_batch': gate_wind_seq,
                'irr_seq_batch': irr_seq,
                'speed_seq_batch': speed_seq,
                'volatility': volatility.mean().item()
            }
            return output, reg_loss, gates

        return output, reg_loss