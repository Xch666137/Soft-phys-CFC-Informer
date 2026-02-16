import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ExplicitPhysicalMapping(nn.Module):
    """
    Explicit Physical Mapping Layer (显式物理映射层) - Smart Init 版
    """

    def __init__(self, d_model, weather_dim=3, learnable_params=True,
                 weather_mean=None, weather_std=None,
                 target_mean=None, target_std=None):
        super().__init__()
        self.d_model = d_model

        # 数据索引
        self.idx_temp = 0
        self.idx_irr = 1
        self.idx_wind = 2

        # -----------------------------------------------------------
        # 辅助函数: Inverse Softplus
        # -----------------------------------------------------------
        def inv_softplus(x):
            if x < 1e-6: x = 1e-6
            return np.log(np.exp(x) - 1.0)

        # -----------------------------------------------------------
        # 参数初始化
        # -----------------------------------------------------------
        init_pv_scale = 0.8
        init_wind_scale = 1.0
        init_load_base = 0.5

        # 根据统计量自动校准初始值
        if target_mean is not None and weather_mean is not None:
            irr_mean = weather_mean[self.idx_irr] if weather_mean[self.idx_irr] > 1e-5 else 1.0
            init_pv_scale = target_mean[1] / irr_mean
            init_wind_scale = 2.0 * target_mean[2]
            init_load_base = target_mean[0]
            print(
                f">>> [PhysLayer Init] Auto-calibrated: PV={init_pv_scale:.4f}, Wind={init_wind_scale:.4f}, Load={init_load_base:.4f}")

        # -----------------------------------------------------------
        # 定义参数 (注意处理 Softplus 的逆变换)
        # -----------------------------------------------------------

        # === 1. PV Parameters ===
        self.pv_efficiency = nn.Parameter(torch.tensor(float(inv_softplus(init_pv_scale))))
        self.pv_temp_coef = nn.Parameter(torch.tensor(0.004))

        # === 2. Wind Parameters ===
        self.wind_scale = nn.Parameter(torch.tensor(float(inv_softplus(init_wind_scale))))
        self.wind_cut_in = nn.Parameter(torch.tensor(3.0))
        self.wind_rated = nn.Parameter(torch.tensor(12.0))
        # [新增] 切出风速参数
        self.wind_cut_out = nn.Parameter(torch.tensor(25.0))

        # === 3. Load Parameters ===
        self.load_base = nn.Parameter(torch.tensor(float(init_load_base)))
        self.temp_comfort = nn.Parameter(torch.tensor(20.0))

        # 负荷敏感度初始化
        # 原代码使用了 -6.0，导致 softplus 后接近 0。改为 0.05 的逆变换。
        init_load_sens = 0.05
        self.load_temp_sens = nn.Parameter(torch.tensor(float(inv_softplus(init_load_sens))))

        # 映射层
        self.out_projection = nn.Linear(3, d_model)

        # 冻结参数逻辑
        if not learnable_params:
            for p in self.parameters():
                p.requires_grad = False

        # 注册统计量用于反归一化
        if weather_mean is not None:
            self.register_buffer('weather_mean', torch.tensor(weather_mean).float())
            self.register_buffer('weather_std', torch.tensor(weather_std).float())
            self.use_denormalization = True
        else:
            self.use_denormalization = False

        if target_mean is not None:
            self.register_buffer('target_mean', torch.tensor(target_mean).float())
            self.register_buffer('target_std', torch.tensor(target_std).float())

    def forward(self, x_weather_normalized):
        # 1. 反归一化得到真实天气值
        if self.use_denormalization:
            x_weather_raw = x_weather_normalized * self.weather_std + self.weather_mean
        else:
            x_weather_raw = x_weather_normalized

        temp = x_weather_raw[:, :, self.idx_temp]
        irr = F.relu(x_weather_raw[:, :, self.idx_irr])
        wind_speed = F.relu(x_weather_raw[:, :, self.idx_wind])

        # ==========================================
        # A. PV Theory
        # ==========================================
        eta = F.softplus(self.pv_efficiency)
        beta = F.softplus(self.pv_temp_coef) * 0.01

        # [修复] 增加 Clamp 防止数值溢出或物理异常
        # 允许低温导致效率提升(>1.0)，但设定上限(如1.5倍)
        temp_loss = torch.clamp(1.0 - beta * (temp - 25.0), min=0.0, max=1.5)

        p_pv_theory = irr * eta * temp_loss

        # ==========================================
        # B. Wind Theory
        # ==========================================
        w_scale = F.softplus(self.wind_scale)

        # [新增] 切出风速逻辑
        # 只有在 (cut_in <= wind < cut_out) 区间内才运行
        is_running = (wind_speed >= self.wind_cut_in) & (wind_speed < self.wind_cut_out)

        w_norm = (wind_speed - self.wind_cut_in) / (self.wind_rated - self.wind_cut_in + 1e-5)
        p_curve = w_scale * torch.sigmoid(5 * (w_norm - 0.5))

        # 应用运行状态掩码
        p_wind_theory = p_curve * is_running.float()

        # ==========================================
        # C. Load Theory
        # ==========================================
        l_sens = F.softplus(self.load_temp_sens)
        p_load_theory = self.load_base + l_sens * (temp - self.temp_comfort).pow(2)

        # ==========================================
        # D. Output
        # ==========================================
        # 这里的 phys_out_real 是真实 MW 值
        phys_out_real = torch.stack([p_load_theory, p_pv_theory, p_wind_theory], dim=-1)

        # 必须归一化后才能返回给 Model 叠加
        if hasattr(self, 'target_mean'):
            phys_out_norm = (phys_out_real - self.target_mean) / (self.target_std + 1e-5)
        else:
            phys_out_norm = phys_out_real

        projected_feat = self.out_projection(phys_out_norm)

        return projected_feat, phys_out_norm