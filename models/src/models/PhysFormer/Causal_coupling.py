import torch
import torch.nn as nn


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

        # PV Gate Learner
        self.gate_learner_pv = nn.Sequential(
            nn.Linear(d_model * 3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )

        # Wind Gate Learner
        self.gate_learner_wind = nn.Sequential(
            nn.Linear(d_model * 3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )

        # Load Gate Learner (负荷通常一直存在，Gate 机制略有不同，主要看天气影响)
        self.gate_learner_load = nn.Sequential(
            nn.Linear(d_model * 3, d_model // 2),
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
        self.irr_threshold_logit = nn.Parameter(torch.tensor(-0.5))
        self.wind_threshold_logit = nn.Parameter(torch.tensor(-1.0))

        # 可学习的斜率
        self.irr_slope_log = nn.Parameter(torch.tensor(2.3))  # exp(2.3) ≈ 10
        self.wind_slope_log = nn.Parameter(torch.tensor(1.6))  # exp(1.6) ≈ 5

        self.register_buffer('step_counter', torch.tensor(0.0))
        self.decay_steps = 5500.0   # batch 1100 * epoch 5

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

    def forward(self, stat_feat, phys_feat, x_weather, return_gates=False):
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

        # ===== Step 1: 准备 Q, K, V =====
        # 关键修改：Q 必须包含 Weather 信息！
        c_weather = self.weather_proj(x_weather)  # [B, S, D]

        # Q = Stat_Proj + Weather_Proj
        # 这样 Stat 流在做 Attention 时，就知道"现在是晚上"，不会去强行匹配光伏特征
        q_load = self.query_proj_load(stat_feat) + c_weather
        q_pv = self.query_proj_pv(stat_feat) + c_weather
        q_wind = self.query_proj_wind(stat_feat) + c_weather

        k_phys = self.phys_proj_k(phys_feat)
        v_phys = self.phys_proj_v(phys_feat)

        # ===== Step 2: 执行因果注意力 (Causal Attention) =====
        # Stat 主动去检索 Phys 中的有用信息
        # attn_out: [B, S, D]
        # 手动生成因果掩码 (Causal Mask)
        B, S, D = q_load.shape
        # 生成一个上三角矩阵，对角线以上为 -inf (表示未来不可见)
        # shape: [S, S]
        causal_mask = torch.triu(torch.ones(S, S) * float('-inf'), diagonal=1).to(q_load.device)

        # ===== Step 2: 执行因果注意力 (Causal Attention) =====
        # 传入 attn_mask，并删除 is_causal=True (避免报错)
        attn_load, _ = self.attn_load(q_load, k_phys, v_phys, attn_mask=causal_mask)
        attn_pv, _ = self.attn_pv(q_pv, k_phys, v_phys, attn_mask=causal_mask)
        attn_wind, _ = self.attn_wind(q_wind, k_phys, v_phys, attn_mask=causal_mask)

        # ===== Step 3: 计算 Gate (决定检索结果的可信度) =====
        # 即使 Attention 检索到了东西，我们也需要 Gate 来决定是否通过
        # 这一步引入了 Curriculum Learning 和 Hard Prior

        # 计算 Soft Gate
        gate_load_soft = self.gate_learner_load(torch.cat([stat_feat, attn_load, c_weather], dim=-1))
        gate_pv_soft = self.gate_learner_pv(torch.cat([stat_feat, attn_pv, c_weather], dim=-1))
        gate_wind_soft = self.gate_learner_wind(torch.cat([stat_feat, attn_wind, c_weather], dim=-1))

        # 应用 Curriculum Learning (硬规则引导)
        if self.training:
            self.step_counter += 1
            progress = torch.clamp(self.step_counter / self.decay_steps, 0.0, 1.0)
            alpha = 1.0 - progress  # 线性衰减

            # [关键] 冻结机制：前50%不更新阈值梯度，强制模型适应初始物理规则
            if progress < 0.5:
                with torch.no_grad():
                    prior_load, prior_pv, prior_wind = self.get_hard_prior(x_weather)
            else:
                prior_load, prior_pv, prior_wind = self.get_hard_prior(x_weather)
        else:
            alpha = 0.1  # 测试时保留10%安全先验
            prior_load, prior_pv, prior_wind = self.get_hard_prior(x_weather)

        prior_pv = prior_pv.repeat(1, 1, D)
        prior_wind = prior_wind.repeat(1, 1, D)
        prior_load = prior_load.repeat(1, 1, D)

        gate_pv = alpha * prior_pv + (1 - alpha) * gate_pv_soft
        gate_wind = alpha * prior_wind + (1 - alpha) * gate_wind_soft
        gate_load = gate_load_soft  # Load通常不需要硬先验

        # 时序平滑（修正版）
        def apply_smoothing(g):
            smoothed = self.smoother(g.transpose(1, 2)).transpose(1, 2)
            return torch.clamp(smoothed, 0.0, 1.0)  # 去掉二次sigmoid

        gate_pv = apply_smoothing(gate_pv)
        gate_wind = apply_smoothing(gate_wind)
        gate_load = apply_smoothing(gate_load)

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
        if self.training:
            irr_t, wind_t, _, _ = self.get_current_thresholds()
            # 鼓励阈值不要偏离 -1.0 (Z-score) 太远
            reg_loss = 0.01 * ((irr_t - (-1.0)) ** 2 + (wind_t - (-1.0)) ** 2)

        if return_gates:
            # 1. 提取 Gate 序列 (Batch, Seq)
            gate_pv_seq = gate_pv.mean(dim=-1).detach().cpu().numpy()
            gate_wind_seq = gate_wind.mean(dim=-1).detach().cpu().numpy()

            # 2. 提取物理环境驱动力序列 (Batch, Seq)
            # x_weather shape: [Batch, Seq, 3] -> (Temp, Irr, Speed)
            irr_seq = x_weather[:, :, 1].detach().cpu().numpy()  # 索引1: 辐照度
            speed_seq = x_weather[:, :, 2].detach().cpu().numpy()  # 索引2: 风速

            gates = {
                'load': gate_load.mean().item(),
                'pv': gate_pv.mean().item(),
                'wind': gate_wind.mean().item(),
                'info_gain': self.info_gain_ema.item(),
                'irr_thresh': self.irr_threshold_ema.item(),  # 记录到日志
                'wind_thresh': self.wind_threshold_ema.item(),
                'gate_reg_loss': reg_loss,          # 返回 Loss 供外部使用
                'pv_seq_batch': gate_pv_seq,
                'wind_seq_batch': gate_wind_seq,

                'irr_seq_batch': irr_seq,
                'speed_seq_batch': speed_seq  # <--- 把风速也传出去
            }
            return output, gates

        return output