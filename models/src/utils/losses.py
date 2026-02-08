import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysAwareVPPLoss(nn.Module):
    def __init__(self, device, means, stds, ramp_limits):
        super().__init__()
        # 1. 基础工具
        self.mse = nn.MSELoss(reduction='none')
        self.mae = nn.L1Loss(reduction='none')

        # beta=0.2 意味着误差 < 0.2 时用平方(细抠细节)，误差 > 0.2 时用线性(抗干扰)
        # 考虑到输入是归一化后的 Z-Score，0.2~0.5 是比较合理的范围
        self.huber = nn.SmoothL1Loss(reduction='none', beta=0.2)

        # 2. 物理统计量 (反归一化用)
        self.register_buffer('means', torch.tensor(means, device=device).view(1, 1, -1))
        self.register_buffer('stds', torch.tensor(stds, device=device).view(1, 1, -1))
        self.register_buffer('ramp_limits', torch.tensor(ramp_limits, device=device).view(1, 1, -1))

        # 3. 自动权重 (6项任务)
        # Base, Net, Deriv, Energy, Direction, Constraints
        self.log_vars = nn.Parameter(torch.zeros(6, device=device))

    def forward(self, pred, true):
        """
        pred, true: [Batch, Seq, 3] (Normalized Z-Score)
        Features: 0:Load, 1:PV, 2:Wind
        """
        # --- A. 数据准备 ---
        # 反归一化到物理域 (MW)
        pred_mw = pred * self.stds + self.means
        true_mw = true * self.stds + self.means

        # 计算差分 (近似导数)
        pred_diff = pred_mw[:, 1:, :] - pred_mw[:, :-1, :]
        true_diff = true_mw[:, 1:, :] - true_mw[:, :-1, :]

        # --- B. 各项 Loss 计算 ---

        # 1. 基础精度 (Base Loss)
        # 使用 MSE 惩罚峰值误差，Load 权重 1.0, PV/Wind 权重 1.2 (稍微侧重源端)
        base_weights = torch.tensor([1.0, 1.2, 1.2], device=pred.device).view(1, 1, 3)
        loss_base = torch.mean(self.huber(pred, true) * base_weights)

        # --- 计算纯物理 MAE (用于早停监控) ---
        # 这一项不参与梯度计算（或者参与也没关系），主要是为了给 Vali 看
        loss_mae = torch.mean(torch.abs(pred_mw - true_mw))

        # 2. 净负荷一致性 (Net Load - VPP Core)
        # 净负荷 = Load - PV - Wind
        pred_net = pred_mw[..., 0] - pred_mw[..., 1] - pred_mw[..., 2]
        true_net = true_mw[..., 0] - true_mw[..., 1] - true_mw[..., 2]
        loss_net = self.mae(pred_net, true_net).mean()

        # 3. 增强型导数 Loss (Weighted Derivative)
        # 对 PV(1)/Wind(2) 的高频波动给予 3 倍关注
        diff_weights = torch.tensor([1.0, 3.0, 3.0], device=pred.device).view(1, 1, 3)
        loss_deriv = torch.mean(torch.abs(pred_diff - true_diff) * diff_weights)

        # 4. 能量/总量一致性 (Energy Integral)
        # 解决 ODE 累积漂移，保证充放电总量正确
        # 按 Batch 平均，对 Sequence 求和
        energy_error = torch.sum(pred_mw, dim=1) - torch.sum(true_mw, dim=1)
        loss_energy = (torch.mean(torch.abs(energy_error)) / pred.shape[1])

        # 5. 方向一致性 (Directional Consistency - 针对 PhysFormer 优化)
        # 解决相位滞后问题：如果真实值在涨，预测值在跌，给重罚
        # relu( - pred_diff * true_diff ) -> 只有异号时才有值
        loss_dir = torch.mean(torch.relu(-1.0 * pred_diff * true_diff)) * 10.0

        # 6. 硬物理约束集合 (Hard Constraints)
        # a. 负值惩罚
        loss_bound = torch.relu(-pred_mw).mean()
        # b. [修正遗漏] 爬坡越限惩罚 (只惩罚超过 limit 的部分)
        loss_ramp = torch.relu(torch.abs(pred_diff) - self.ramp_limits).mean() * 1.0
        # c. [新增] PV 夜间噪声惩罚 (Load=0, PV=1, Wind=2)
        true_pv = true_mw[:, :, 1]
        night_mask = (true_pv < 0.01).float()  # 真实PV为0时
        loss_night = (pred_mw[:, :, 1] ** 2 * night_mask).mean()

        loss_constraints = (loss_bound + loss_ramp + loss_night) * 10.0

        # --- C. 自动权重聚合 (Kendall & Gal) ---
        # 列表顺序对应 self.log_vars 的索引
        task_losses = [loss_base, loss_net, loss_deriv, loss_energy, loss_dir, loss_constraints]

        total_loss = 0
        for i, loss_item in enumerate(task_losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss_item + 0.5 * self.log_vars[i]

        ws = torch.exp(-self.log_vars).detach().cpu().numpy()

        return total_loss, {
            'base': loss_base.item(),
            'mae': loss_mae.item(),  # <--- 【关键修复】必须补上这一行
            'net': loss_net.item(),
            'deriv': loss_deriv.item(),
            'energy': loss_energy.item(),
            'dir': loss_dir.item(),
            'cons': loss_constraints.item(),

            # --- 全量权重监控 ---
            # 权重变化：如果某项权重(exp(-var))变大，说明模型认为这项容易学且重要
            'w_base': ws[0],
            'w_net': ws[1],
            'w_deriv': ws[2],
            'w_energy': ws[3],
            'w_dir': ws[4],
            'w_cons': ws[5]
        }