import torch
import torch.nn as nn


class VPPDomainLoss(nn.Module):
    def __init__(self, device,
                 means=None, stds=None, ramp_limits=None,
                 alpha_bound=0.0,       # 物理边界权重 (建议 1.0)
                 alpha_ramp=0.0,        # 爬坡惩罚权重 (建议 0.1 - 仅作安全网)
                 alpha_energy=0.0,      # 能量一致权重 (建议 0.5 - 1.0)
                 alpha_deriv=0.0):      # [新增] 导数匹配权重 (核心！建议 1.0 - 2.0)
        super().__init__()
        self.base_loss = nn.L1Loss()

        # 动态权重（可在训练中修改）
        self.alpha_bound = alpha_bound
        self.alpha_ramp = alpha_ramp
        self.alpha_energy = alpha_energy
        self.alpha_deriv = alpha_deriv

        # --- 初始化物理参数 (基于你的数据集统计) ---
        # Load, PV, Wind
        if means is None: means = [2.8488, 0.1437, 0.5791]
        if stds is None: stds = [0.8474, 0.2448, 0.5341]
        if ramp_limits is None: ramp_limits = [1.0, 0.25, 0.35]  # MW/15min

        self.register_buffer('means', torch.tensor(means, device=device).view(1, 1, -1))
        self.register_buffer('stds', torch.tensor(stds, device=device).view(1, 1, -1))
        self.register_buffer('ramp_limits', torch.tensor(ramp_limits, device=device).view(1, 1, -1))

    def forward(self, pred, true):
        """
        pred, true: 均为归一化后的 Z-Score [Batch, Seq_Len, 3]
        """
        # 1. 基础精度 (L1 Loss) - 保证数值准确性
        loss_val = self.base_loss(pred, true)

        # --- 反归一化 (恢复物理量 MW) ---
        pred_phys = pred * self.stds + self.means
        true_phys = true * self.stds + self.means  # 真实值也恢复，用于能量计算

        # --- 2. 导数匹配损失 (Derivative Matching) ---
        # 这一项专门对抗"平均化"。它强迫预测曲线的"走势/斜率"必须和真实曲线一样。
        # 计算相邻时刻差分 (近似导数)
        pred_diff = pred_phys[:, 1:, :] - pred_phys[:, :-1, :]
        true_diff = true_phys[:, 1:, :] - true_phys[:, :-1, :]

        # 我们希望 pred_diff 无限接近 true_diff (L1 范数)
        # 这比单纯惩罚幅度更能抓住"相位"和"形状"
        loss_deriv = torch.mean(torch.abs(pred_diff - true_diff))

        # --- 3. 物理边界约束 (Boundary) ---
        # 惩罚负值 (保持不变，这是硬物理铁律)
        loss_bound = torch.relu(-pred_phys).mean()

        # --- 4. 爬坡率"安全"约束 (Ramp Constraint) ---
        # 注意：这里我们稍微降级它的作用，只把它当作"熔断器"。
        # 即：只要没超过物理极限 ramp_limits，我们就不惩罚它的波动（哪怕波动很大），
        # 而是交给 loss_deriv 去引导它拟合真实的波动。
        # 只有当预测值波动 超过了 物理极限，才进行惩罚。
        diff_abs_pred = torch.abs(pred_diff)
        loss_ramp = torch.relu(diff_abs_pred - self.ramp_limits).mean()

        # --- 5. 能量一致性约束 (Energy) ---
        total_pred = torch.sum(pred_phys, dim=1)
        total_true = torch.sum(true_phys, dim=1)
        loss_energy = torch.mean(torch.abs(total_pred - total_true)) / pred.shape[1]

        # --- 聚合 ---
        # 建议 alpha_deriv 给大一点 (如 1.0 或 2.0)，因为它直接提升波形拟合度
        total_loss = loss_val + \
                     (self.alpha_bound * loss_bound) + \
                     (self.alpha_ramp * loss_ramp) + \
                     (self.alpha_energy * loss_energy) + \
                     (self.alpha_deriv * loss_deriv)

        # 返回 loss 和 监控指标
        return total_loss, {
            'mae': loss_val.item(),
            'bound': loss_bound.item(),
            'ramp': loss_ramp.item(),
            'energy': loss_energy.item(),
            'deriv': loss_deriv.item()
        }