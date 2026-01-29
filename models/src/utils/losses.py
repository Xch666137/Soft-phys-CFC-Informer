import torch
import torch.nn as nn


class VPPDomainLoss(nn.Module):
    def __init__(self, device,
                 means=None, stds=None, ramp_limits=None,
                 alpha_bound=0.0, alpha_ramp=0.0, alpha_energy=0.0):
        super().__init__()
        self.base_loss = nn.L1Loss()

        # 动态权重（可在训练中修改）
        self.alpha_bound = alpha_bound
        self.alpha_ramp = alpha_ramp
        self.alpha_energy = alpha_energy

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

        # 2. 物理边界约束 (Boundary) - 惩罚负值
        # 物理铁律: 功率 >= 0
        loss_bound = torch.relu(-pred_phys).mean()

        # 3. 爬坡率约束 (Ramp Rate) - 惩罚剧烈抖动
        # 计算差分 |P_t - P_{t-1}|
        diff_phys = torch.abs(pred_phys[:, 1:, :] - pred_phys[:, :-1, :])
        # Deadband机制: 只惩罚超过 limit 的部分
        loss_ramp = torch.relu(diff_phys - self.ramp_limits).mean()

        # 4. 能量一致性约束 (Energy Consistency) - 惩罚总电量偏差
        # 对时间维度求和，对比总能量
        total_pred = torch.sum(pred_phys, dim=1)
        total_true = torch.sum(true_phys, dim=1)
        loss_energy = torch.mean(torch.abs(total_pred - total_true)) / pred.shape[1]

        # --- 聚合 ---
        total_loss = loss_val + \
                     (self.alpha_bound * loss_bound) + \
                     (self.alpha_ramp * loss_ramp) + \
                     (self.alpha_energy * loss_energy)

        # 返回 loss 和 监控指标
        return total_loss, {
            'mae': loss_val.item(),
            'bound': loss_bound.item(),
            'ramp': loss_ramp.item(),
            'energy': loss_energy.item()
        }