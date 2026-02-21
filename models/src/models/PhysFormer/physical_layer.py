import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ExplicitPhysicalMapping(nn.Module):
    """
    显式物理映射层 (Explicit Physical Mapping Layer)

    将天气特征通过三组物理方程直接映射为理论功率基准值：
        - PV理论输出    : 基于辐照度-温度效率模型
        - 风机理论输出  : 基于风功率曲线模型（三段式）
        - 负荷理论基准  : 基于温度舒适度模型

    ============================================================
    所有可学习物理参数的物理含义、量纲与合理值范围
    ============================================================

    === PV 组件参数 ===

    pv_efficiency (nn.Parameter, 经 softplus 激活)
        物理含义 : 光伏组件综合光电转换效率 η_pv
        量纲     : 无量纲 (归一化辐照度→归一化功率 的比例系数)
        物理公式 : P_pv = G × η_pv × (1 - β_T × (T - 25))
                   其中 G 为辐照度，T 为组件温度
        典型范围 : 0.15 ~ 0.25 (单晶硅约0.20，多晶硅约0.17)
        初始值   : 由训练集统计量自动校准 (target_mean[PV] / irr_mean)
        激活函数 : softplus(·)，确保恒正

    pv_temp_coef (nn.Parameter, 经 softplus(·)×0.01 激活)
        物理含义 : 温度功率系数 β_T，表示组件温度每升高 1°C 时效率的相对下降比例
        量纲     : /°C  (每摄氏度的效率衰减率)
        物理公式 : 效率修正因子 = 1 - β_T × (T_cell - T_STC)
                   T_STC = 25°C (标准测试条件温度)
        典型范围 : 0.003 ~ 0.005 /°C
                   单晶硅约 0.004 /°C，非晶硅约 0.002 /°C
        注意事项 : 参数存储的是 softplus 激活前的原始值，
                   实际使用值 = softplus(pv_temp_coef) * 0.01
                   即存储参数约在 0.3~0.5 附近时，激活后约 0.003~0.005
        先验目标 : 0.004 /°C

    === 风机组件参数（三段式功率曲线） ===

    ┌─────────────────────────────────────────────┐
    │  P_wind                                     │
    │    │          额定功率────────────┐          │
    │    │         /                   │ 停机     │
    │    │        /                    │──────    │
    │    │       /                     │          │
    │    │──────/                      │          │
    │    └──────┬────────┬─────────────┬──── v    │
    │         cut_in   rated        cut_out       │
    └─────────────────────────────────────────────┘

    风机工作区间：cut_in ≤ v < cut_out
    在此区间内：P ∝ sigmoid(5 × (v_norm - 0.5))，平滑近似风功率曲线

    【重要设计】三个参数采用"累积增量 + softplus"结构以保证严格物理顺序：
        cut_in_abs  = softplus(p_cut_in)
        rated_abs   = cut_in_abs  + softplus(p_rated_delta)
        cut_out_abs = rated_abs   + softplus(p_cut_out_delta)
    这在数学上永远满足: 0 < cut_in < rated < cut_out

    wind_cut_in (nn.Parameter → softplus → 绝对值)
        物理含义 : 风机启动风速（切入风速），低于此值风机不运行
        量纲     : m/s
        典型范围 : 2.5 ~ 4.5 m/s，常用值 3.0 ~ 3.5 m/s
        先验目标 : 3.5 m/s
        初始参数 : inv_softplus(3.5) ≈ 3.45（使初始激活值≈目标值）

    wind_rated_delta (nn.Parameter → softplus → 相对cut_in的增量)
        物理含义 : rated_abs - cut_in_abs，即额定风速超出启动风速的增量
        量纲     : m/s（增量）
        额定风速典型范围 : 10 ~ 15 m/s，常用值 12 m/s
        先验目标(增量): 12.0 - 3.5 = 8.5 m/s
        初始参数 : inv_softplus(8.5) ≈ 8.50

    wind_cut_out_delta (nn.Parameter → softplus → 相对rated的增量)
        物理含义 : cut_out_abs - rated_abs，即停机风速超出额定风速的增量
        量纲     : m/s（增量）
        停机风速典型范围 : 20 ~ 30 m/s，常用值 25 m/s
        先验目标(增量): 25.0 - 12.0 = 13.0 m/s
        初始参数 : inv_softplus(13.0) ≈ 12.99

    wind_scale (nn.Parameter, 经 softplus 激活)
        物理含义 : 风机额定输出功率的归一化比例系数
        量纲     : 无量纲（与归一化后的功率量纲一致）
        物理公式 : P_wind = wind_scale × sigmoid(5 × (v_norm - 0.5)) × is_running
        典型范围 : 视数据集归一化方式而定，通常 0.5 ~ 2.0
        初始值   : 由训练集统计量自动校准 (2.0 × target_mean[Wind])
        激活函数 : softplus(·)，确保恒正

    === 负荷组件参数 ===

    load_base (nn.Parameter, 无激活函数)
        物理含义 : 基础负荷，即与温度无关的刚性用电需求（归一化值）
        量纲     : 与归一化后的负荷量纲一致（对应真实单位 MW）
        物理公式 : P_load = load_base + l_sens × (T - T_comfort)²
        典型范围 : 0.3 ~ 0.8（归一化值），对应典型园区基础负荷
        初始值   : 由训练集 target_mean[Load] 自动校准
        注意事项 : 无激活约束，理论上可为负，但先验正则化会将其拉回合理范围

    temp_comfort (nn.Parameter, 无激活函数)
        物理含义 : 热舒适基准温度，在此温度附近负荷最低（二次曲线最低点）
        量纲     : °C
        物理含义 : 人体舒适区间约 18~26°C，供暖/制冷切换点约 18~22°C
        典型范围 : 18.0 ~ 24.0 °C
        初始值   : 20.0 °C
        注意事项 : 无激活约束，依赖先验正则化约束范围

    load_temp_sens (nn.Parameter, 经 softplus 激活)
        物理含义 : 温度负荷敏感度，温度每偏离舒适温度 1°C 时负荷的增量（归一化值）
        量纲     : [归一化负荷单位] / °C²
        物理公式 : 温度引起的负荷增量 = l_sens × (T - T_comfort)²
        典型范围 : 0.01 ~ 0.10（归一化值）
        初始值   : inv_softplus(0.05)，使激活后约 0.05
        激活函数 : softplus(·)，确保恒正（温度越偏离舒适点，负荷越高）
    """

    def __init__(self, d_model, weather_dim=3, learnable_params=True,
                 weather_mean=None, weather_std=None,
                 target_mean=None, target_std=None):
        super().__init__()
        self.d_model = d_model

        # 天气数据列索引（与 PhysFormerDataset 保持一致）
        # x_weather: [..., 0]=温度(°C), [..., 1]=辐照度(W/m²归一化), [..., 2]=风速(m/s归一化)
        self.idx_temp = 0
        self.idx_irr  = 1
        self.idx_wind = 2

        # -----------------------------------------------------------
        # 辅助函数: Inverse Softplus
        # softplus(x) = log(1 + exp(x))，单调递增且值域为(0,+∞)
        # inv_softplus(y) = log(exp(y) - 1)，用于根据目标激活值反推初始参数值
        # -----------------------------------------------------------
        def inv_softplus(x):
            if x < 1e-6:
                x = 1e-6
            return np.log(np.exp(x) - 1.0)

        # -----------------------------------------------------------
        # 根据训练集统计量自动校准初始值
        # -----------------------------------------------------------
        init_pv_scale   = 0.8
        init_wind_scale = 1.0
        init_load_base  = 0.5

        if target_mean is not None and weather_mean is not None:
            irr_mean        = weather_mean[self.idx_irr] if abs(weather_mean[self.idx_irr]) > 1e-5 else 1.0
            init_pv_scale   = target_mean[1] / irr_mean                 # PV均值 / 辐照度均值
            init_wind_scale = target_mean[2] + 2.0 * target_std[2]      # 95th百分位
            init_load_base  = target_mean[0]                            # 负荷均值作为基础负荷
            print(
                f">>> [PhysLayer Init] Auto-calibrated: "
                f"PV_scale={init_pv_scale:.4f}, Wind_scale={init_wind_scale:.4f}, "
                f"Load_base={init_load_base:.4f}"
            )

        # -----------------------------------------------------------
        # === 1. PV 参数组 ===
        # -----------------------------------------------------------

        # pv_efficiency: η_pv，光电转换效率
        # 存储值：softplus 激活前的原始参数
        # 激活后：softplus(pv_efficiency)，量纲无量纲，典型值 0.15~0.25
        self.pv_efficiency = nn.Parameter(
            torch.tensor(float(inv_softplus(max(init_pv_scale, 1e-5))))
        )

        # pv_temp_coef: β_T，温度功率系数（存储值）
        # 激活后：softplus(pv_temp_coef) * 0.01，量纲 /°C，典型值 0.003~0.005
        # 初始存储值对应激活后约 0.004/°C:
        #   softplus(x) * 0.01 = 0.004  →  softplus(x) = 0.4  →  x = inv_softplus(0.4) ≈ 0.337
        self.pv_temp_coef = nn.Parameter(
            torch.tensor(float(inv_softplus(0.4)))  # → softplus(0.4)*0.01 ≈ 0.004 /°C
        )

        # 数据估算的综合先验系数
        self._target_pv_eff = float(max(init_pv_scale, 1e-5))

        # -----------------------------------------------------------
        # === 2. 风机参数组 ===
        # 采用"累积增量 + softplus"确保严格物理顺序: cut_in < rated < cut_out
        #
        # 参数语义（存储值）：
        #   wind_cut_in      : 绝对值的 softplus 前参数，激活后=切入风速绝对值
        #   wind_rated_delta : 增量的 softplus 前参数，激活后=额定风速相对切入风速的增量
        #   wind_cut_out_delta: 增量的 softplus 前参数，激活后=停机风速相对额定风速的增量
        #
        # 初始化目标：cut_in≈3.5, rated≈12.0, cut_out≈25.0
        #   → 增量：Δrated=8.5, Δcut_out=13.0
        # -----------------------------------------------------------

        # wind_cut_in: softplus 后 = 切入风速 (m/s)，目标初值 3.5 m/s
        self.wind_cut_in = nn.Parameter(
            torch.tensor(float(inv_softplus(3.5)))   # → softplus(·) ≈ 3.5 m/s
        )

        # wind_rated_delta: softplus 后 = (额定风速 - 切入风速) (m/s)，目标 8.5 m/s
        self.wind_rated_delta = nn.Parameter(
            torch.tensor(float(inv_softplus(8.5)))   # → softplus(·) ≈ 8.5 m/s
        )

        # wind_cut_out_delta: softplus 后 = (停机风速 - 额定风速) (m/s)，目标 13.0 m/s
        self.wind_cut_out_delta = nn.Parameter(
            torch.tensor(float(inv_softplus(13.0)))  # → softplus(·) ≈ 13.0 m/s
        )

        # wind_scale: 风机额定输出归一化比例系数，量纲无量纲，激活后恒正
        self.wind_scale = nn.Parameter(
            torch.tensor(float(inv_softplus(max(init_wind_scale, 0.1))))
        )

        # -----------------------------------------------------------
        # === 3. 负荷参数组 ===
        # -----------------------------------------------------------

        # load_base: 基础负荷（归一化值），量纲与归一化负荷一致
        # 无激活函数，依赖先验正则化约束其在合理范围内
        self.load_base = nn.Parameter(torch.tensor(float(init_load_base)))

        # 保存初始化时的负荷基值作为先验目标（正则化时使用）
        self._target_load_base = float(init_load_base)

        # temp_comfort: 热舒适基准温度 (°C)，二次负荷模型的最低点
        # 无激活函数，先验正则化约束在 18~24°C 附近
        self.temp_comfort = nn.Parameter(torch.tensor(20.0))

        # load_temp_sens: 温度负荷敏感度（softplus后），量纲 [归一化负荷] / °C²
        # 激活后恒正，确保"偏离舒适温度→负荷升高"的物理单调性
        # 数据估算值约 0.005
        self.load_temp_sens = nn.Parameter(
            torch.tensor(float(inv_softplus(0.005)))
        )

        # -----------------------------------------------------------
        # 输出投影：将3维物理输出映射到 d_model 维特征空间
        # 输入：[P_load_norm, P_pv_norm, P_wind_norm]（归一化物理量）
        # 输出：d_model 维特征向量
        # -----------------------------------------------------------
        self.out_projection = nn.Linear(3, d_model)

        # 冻结参数选项（用于消融实验，固定物理先验不学习）
        if not learnable_params:
            for p in self.parameters():
                p.requires_grad = False

        # 注册天气统计量 buffer（用于反归一化天气输入）
        if weather_mean is not None:
            self.register_buffer('weather_mean', torch.tensor(weather_mean).float())
            self.register_buffer('weather_std',  torch.tensor(weather_std).float())
            self.use_denormalization = True
        else:
            self.use_denormalization = False

        # 注册目标量统计量 buffer（用于归一化物理输出）
        if target_mean is not None:
            self.register_buffer('target_mean', torch.tensor(target_mean).float())
            self.register_buffer('target_std',  torch.tensor(target_std).float())

    # ------------------------------------------------------------------
    # 辅助方法：计算当前风速阈值的绝对值（供 forward 和 prior_loss 共用）
    # ------------------------------------------------------------------
    def _get_wind_thresholds(self):
        """
        通过累积增量结构计算风机三段阈值的绝对值（m/s）。

        返回:
            cut_in  (Tensor, scalar): 切入风速，恒 > 0
            rated   (Tensor, scalar): 额定风速，恒 > cut_in
            cut_out (Tensor, scalar): 停机风速，恒 > rated

        数学保证:
            cut_in  = softplus(p_cut_in)                    > 0
            rated   = cut_in + softplus(p_rated_delta)      > cut_in
            cut_out = rated  + softplus(p_cut_out_delta)    > rated
        """
        cut_in  = F.softplus(self.wind_cut_in)
        rated   = cut_in  + F.softplus(self.wind_rated_delta)
        cut_out = rated   + F.softplus(self.wind_cut_out_delta)
        return cut_in, rated, cut_out

    def forward(self, x_weather_normalized):
        """
        前向传播：天气特征 → 理论物理功率基准

        Args:
            x_weather_normalized: [B, T, 3]，归一化后的天气序列
                [..., 0] = 温度 (StandardScaler 归一化)
                [..., 1] = 辐照度 (StandardScaler 归一化)
                [..., 2] = 风速 (StandardScaler 归一化)

        Returns:
            projected_feat: [B, T, d_model]，投影到模型特征空间的物理特征
            phys_out_norm:  [B, T, 3]，归一化后的理论功率 [Load, PV, Wind]
        """
        # --- Step 1: 反归一化，还原真实物理量 ---
        if self.use_denormalization:
            # weather_std/mean shape: [3]，广播到 [B, T, 3]
            x_weather_raw = x_weather_normalized * self.weather_std + self.weather_mean
        else:
            x_weather_raw = x_weather_normalized

        # 提取各天气变量，shape: [B, T]
        temp       = x_weather_raw[:, :, self.idx_temp]            # 温度 (°C)
        irr        = F.relu(x_weather_raw[:, :, self.idx_irr])     # 辐照度 (W/m²归一化)，截断负值
        wind_speed = F.relu(x_weather_raw[:, :, self.idx_wind])    # 风速 (m/s归一化)，截断负值

        # ==================================================
        # A. PV 理论功率
        # 公式: P_pv = G × η_pv × (1 - β_T × (T - 25))
        # ==================================================

        # η_pv: 转换效率，量纲无量纲，典型值 0.15~0.25
        eta  = F.softplus(self.pv_efficiency)

        # β_T: 温度系数，量纲 /°C，典型值 0.003~0.005
        # 存储参数 → softplus → ×0.01 得到最终值
        beta = F.softplus(self.pv_temp_coef) * 0.01

        # 温度效率修正因子: clamp 防止物理异常
        # min=0.0: 极高温时效率不应为负
        # max=1.5: 低温时效率提升最多不超过标准值的1.5倍
        temp_loss = torch.clamp(1.0 - beta * (temp - 25.0), min=0.0, max=1.5)

        # PV 理论输出（与辐照度量纲一致的归一化功率）
        p_pv_theory = irr * eta * temp_loss

        # ==================================================
        # B. 风机理论功率（三段式功率曲线）
        # 工作区间: cut_in ≤ v < cut_out
        # 功率曲线: P ∝ sigmoid(5 × (v_norm - 0.5))
        # ==================================================

        # 获取三个阈值的绝对值（严格顺序保证）
        cut_in, rated, cut_out = self._get_wind_thresholds()
        # cut_in: 切入风速 (m/s归一化，与wind_speed同量纲)
        # rated : 额定风速 (m/s归一化)
        # cut_out: 停机风速 (m/s归一化)

        # 归一化风速（相对于切入→额定区间）
        w_norm = (wind_speed - cut_in) / (rated - cut_in + 1e-5)
        w_scale = F.softplus(self.wind_scale)
        p_curve = w_scale * torch.sigmoid(5.0 * (w_norm - 0.5))

        # 使用陡峭的 Sigmoid 替代布尔掩码，确保梯度回传
        steepness = 20.0
        # 当 wind_speed > cut_in 时逼近 1，否则逼近 0
        turn_on = torch.sigmoid(steepness * (wind_speed - cut_in))
        # 当 wind_speed < cut_out 时逼近 1，否则逼近 0
        turn_off = torch.sigmoid(steepness * (cut_out - wind_speed))

        is_running_soft = turn_on * turn_off

        # 应用软运行状态掩码
        p_wind_theory = p_curve * is_running_soft

        # ==================================================
        # C. 负荷理论基准
        # 公式: P_load = load_base + l_sens × (T - T_comfort)²
        # 二次型：温度偏离舒适点越远，负荷越高（制冷/供暖增加）
        # ==================================================

        # l_sens: 温度敏感度，量纲 [归一化负荷]/°C²，softplus 保证恒正
        l_sens        = F.softplus(self.load_temp_sens)
        p_load_theory = self.load_base + l_sens * (temp - self.temp_comfort).pow(2)

        # ==================================================
        # D. 组合输出
        # ==================================================

        # 理论功率（真实量纲/归一化量纲，与天气量纲一致）
        # phys_out_real shape: [B, T, 3]，顺序为 [Load, PV, Wind]
        phys_out_real = torch.stack([p_load_theory, p_pv_theory, p_wind_theory], dim=-1)

        # 归一化到与标签相同的量纲（便于与统计流输出叠加）
        if hasattr(self, 'target_mean'):
            phys_out_norm = (phys_out_real - self.target_mean) / (self.target_std + 1e-5)
        else:
            phys_out_norm = phys_out_real                             # 未注册统计量时原样输出

        # 投影到 d_model 特征空间，供后续 Causal Coupling 使用
        projected_feat = self.out_projection(phys_out_norm)           # [B, T, d_model]

        return projected_feat, phys_out_norm

    def get_physics_prior_loss(self, prior_weight=0.1):
        """
        物理参数先验正则化损失。

        原理：对可学习物理参数施加 MSE 正则化，将其约束在物理常识范围附近，
        防止梯度将参数推至物理上不合理的区域（如 PV 效率 > 1，温度系数趋于0等）。

        先验目标值选取依据（基于工程经验）：
            pv_efficiency    → 0.20 (单晶硅平均效率)
            pv_temp_coef     → 0.004 /°C (激活后值，Si 组件标准值)
            wind_cut_in      → 3.5 m/s (典型切入风速)
            wind_rated (abs) → 12.0 m/s (典型额定风速)
            wind_cut_out(abs)→ 25.0 m/s (典型停机风速)
            load_base        → 初始化时的校准值（训练集负荷均值）
            temp_comfort     → 20.0 °C (人体舒适温度)

        Args:
            prior_weight (float): 正则化强度，建议范围 0.01 ~ 0.2

        Returns:
            prior_loss (Tensor): 标量正则化损失
            prior_dict (dict):   各项先验损失的监控字典
        """
        # --- 获取当前参数的激活后值（与 forward 保持完全一致）---

        # PV 参数（激活后真实物理值）
        pv_eff      = F.softplus(self.pv_efficiency)
        # pv_temp_coef 激活后值 = softplus(·) * 0.01，量纲 /°C
        pv_temp_coef = F.softplus(self.pv_temp_coef) * 0.01

        # 风机参数：使用共享方法确保与 forward 中完全一致
        cut_in, rated, cut_out = self._get_wind_thresholds()
        wind_scale = F.softplus(self.wind_scale)

        # 负荷参数
        load_base      = self.load_base
        load_temp_sens = F.softplus(self.load_temp_sens)
        temp_comfort   = self.temp_comfort

        # --- 先验目标值 ---

        dev = pv_eff.device  # 统一设备

        # PV 效率先验: 使用数据估算的综合系数
        target_pv_eff = torch.tensor(
            self._target_pv_eff if hasattr(self, '_target_pv_eff') else 0.0012,
            device=dev
        )

        # PV 温度系数先验: 0.004 /°C
        # 注意：这里直接对比激活后的值（softplus * 0.01），目标同样使用激活后量纲
        # 勿再乘 0.01（原始代码中曾对 0.0045 额外乘 0.01 造成双重缩放 bug）
        target_pv_temp_coef = torch.tensor(0.004, device=dev)

        # 风机阈值先验（绝对值，m/s归一化）
        target_cut_in  = torch.tensor(3.5,  device=dev)  # 切入风速
        target_rated   = torch.tensor(12.0, device=dev)  # 额定风速（绝对值）
        target_cut_out = torch.tensor(25.0, device=dev)  # 停机风速（绝对值）

        # 负荷基准先验：使用初始化时的校准值
        target_load_base    = torch.tensor(
            self._target_load_base if hasattr(self, '_target_load_base') else load_base.item(),
            device=dev
        )

        # 热舒适温度先验: 20.0 °C
        target_temp_comfort = torch.tensor(20.0, device=dev)

        # --- 各项 MSE 损失 ---

        loss_pv_eff       = F.mse_loss(pv_eff,      target_pv_eff)
        # pv_temp_coef: 直接对比激活后值（量纲 /°C），无双重缩放
        loss_pv_temp_coef = F.mse_loss(pv_temp_coef, target_pv_temp_coef)

        # 风机阈值：对比绝对值（由累积结构计算，保证物理顺序）
        loss_wind_cut_in  = F.mse_loss(cut_in,  target_cut_in)
        loss_wind_rated   = F.mse_loss(rated,   target_rated)
        loss_wind_cut_out = F.mse_loss(cut_out, target_cut_out)

        loss_load_base    = F.mse_loss(load_base,    target_load_base)
        loss_temp_comfort = F.mse_loss(temp_comfort, target_temp_comfort)

        # --- 加权求和 ---
        prior_loss = prior_weight * (
            loss_pv_eff
            + loss_pv_temp_coef
            + loss_wind_cut_in
            + loss_wind_rated
            + loss_wind_cut_out
            + loss_load_base
            + 0.5 * loss_temp_comfort    # 舒适温度约束适当放宽（不同气候差异较大）
        )

        # --- 监控字典（记录激活后的真实物理值及各项先验损失）---
        prior_dict = {
            # 激活后的真实物理参数值（用于 TensorBoard 监控）
            'actual_pv_eff':         pv_eff.item(),
            'actual_pv_temp_coef':   pv_temp_coef.item(),
            'actual_wind_cut_in':    cut_in.item(),
            'actual_wind_rated':     rated.item(),
            'actual_wind_cut_out':   cut_out.item(),
            'actual_wind_scale':     wind_scale.item(),
            'actual_load_base':      load_base.item(),
            'actual_temp_comfort':   temp_comfort.item(),
            'actual_load_temp_sens': load_temp_sens.item(),
            # 各项先验 MSE 损失
            'prior_pv_eff':          loss_pv_eff.item(),
            'prior_pv_temp_coef':    loss_pv_temp_coef.item(),
            'prior_wind_cut_in':     loss_wind_cut_in.item(),
            'prior_wind_rated':      loss_wind_rated.item(),
            'prior_wind_cut_out':    loss_wind_cut_out.item(),
            'prior_load_base':       loss_load_base.item(),
            'prior_temp_comfort':    loss_temp_comfort.item(),
            'prior_total':           prior_loss.item(),
        }

        return prior_loss, prior_dict

    def get_current_physics_params(self):
        """
        便捷方法：返回当前所有物理参数的激活后真实值（用于日志/调试）。

        Returns:
            dict: 参数名 → 激活后真实值（Python float）
        """
        cut_in, rated, cut_out = self._get_wind_thresholds()
        with torch.no_grad():
            return {
                'pv_efficiency [dimensionless]':    F.softplus(self.pv_efficiency).item(),
                'pv_temp_coef [/°C]':               (F.softplus(self.pv_temp_coef) * 0.01).item(),
                'wind_cut_in [m/s normalized]':     cut_in.item(),
                'wind_rated [m/s normalized]':      rated.item(),
                'wind_cut_out [m/s normalized]':    cut_out.item(),
                'wind_scale [dimensionless]':        F.softplus(self.wind_scale).item(),
                'load_base [normalized MW]':         self.load_base.item(),
                'temp_comfort [°C]':                 self.temp_comfort.item(),
                'load_temp_sens [norm_MW/°C²]':      F.softplus(self.load_temp_sens).item(),
            }