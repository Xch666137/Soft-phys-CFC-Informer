import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysAwareBaseLoss(nn.Module):
    """
    物理感知基础计算层 (无状态，仅负责计算原始Loss)
    """

    def __init__(self, device, means, stds, ramp_limits, quantiles=None):
        super().__init__()
        self.device = device
        self.register_buffer('means', torch.tensor(means).float().to(device))
        self.register_buffer('stds', torch.tensor(stds).float().to(device))
        self.register_buffer('ramp_limits', torch.tensor(ramp_limits).float().to(device))

    def get_raw_components(self, pred, true):
        """
        计算各项原始损失值
        Args:
            pred: [B, T, 3] 归一化后的预测值
            true: [B, T, 3] 归一化后的真实值
        """
        # 1. 主损失 (MSE for Point Prediction)
        loss_main = F.mse_loss(pred, true)

        # 2. 反归一化 (还原为真实物理量 MW)用于计算物理Loss
        pred_real = pred * self.stds + self.means
        true_real = true * self.stds + self.means

        # 3. 物理一致性 (MW单位)

        # A. Net Load Consistency: Load - PV - Wind
        # 预测的净负荷应该接近真实的净负荷
        net_pred = pred_real[..., 0] - pred_real[..., 1] - pred_real[..., 2]
        net_true = true_real[..., 0] - true_real[..., 1] - true_real[..., 2]
        loss_net = F.l1_loss(net_pred, net_true)

        # B. Derivative (Shape/Smoothness)
        # 一阶差分（变化率）应该相似
        diff_pred = pred_real[:, 1:] - pred_real[:, :-1]
        diff_true = true_real[:, 1:] - true_real[:, :-1]
        loss_deriv = F.l1_loss(diff_pred, diff_true)

        # C. Energy Integral (Daily sum)
        # 总电量应该守恒
        loss_energy = F.l1_loss(torch.sum(pred_real, dim=1), torch.sum(true_real, dim=1))

        # D. Direction (1 - CosSim)
        # 变化趋势方向应该一致
        eps = 1e-8
        diff_pred_norm = diff_pred / (torch.norm(diff_pred, dim=-1, keepdim=True) + eps)
        diff_true_norm = diff_true / (torch.norm(diff_true, dim=-1, keepdim=True) + eps)
        # 计算余弦相似度
        cos_sim = (diff_pred_norm * diff_true_norm).sum(dim=-1)
        loss_dir = torch.mean(1.0 - cos_sim)

        # E. BVR (Boundary Violation Ratio) - Negative Power Penalty
        # 功率不能为负 (ReLU惩罚负值)
        loss_bvr = torch.mean(F.relu(-pred_real))

        # F. RVR (Ramp Violation Ratio)
        # 爬坡不能超过物理极限
        # diff_pred 是 [B, T-1, 3], ramp_limits 是 [3]
        ramp_violation = F.relu(torch.abs(diff_pred) - self.ramp_limits)
        loss_rvr = torch.mean(ramp_violation)

        # Monitor MAE (作为直观指标)
        loss_mae = F.l1_loss(pred_real, true_real)

        return {
            'main': loss_main,
            'mae': loss_mae,
            'net': loss_net,
            'deriv': loss_deriv,
            'energy': loss_energy,
            'dir': loss_dir,
            'bvr': loss_bvr,
            'rvr': loss_rvr
        }


class PhysLoss(nn.Module):
    """
    动态幅度平衡物理损失 (Dynamic Magnitude Balancing)

    原理：
    计算 L_main (MSE) 与 L_phys 的比例 (Scale)，
    确保物理损失经过缩放后，与主损失处于同一数量级。
    """

    def __init__(self, base_loss_module: PhysAwareBaseLoss, device,
                 warmup_batches=50, ema_decay=0.9):
        super().__init__()
        self.base_loss = base_loss_module
        self.device = device
        self.warmup_batches = warmup_batches
        self.ema_decay = ema_decay

        # 内部相对权重 (Fixed Heuristics)
        # 经验法则：守恒定律(Net)最重要，其次是总量(Energy)，最后是形状细节
        self.sub_weights = {
            'net': 1.0,
            'energy': 0.5,
            'deriv': 0.5,
            'dir': 0.1,
            'bvr': 2.0,  # 边界违规惩罚重一点
            'rvr': 2.0
        }

        # 状态量：用于记录 Scale 的 EMA
        self.register_buffer('scale_ema', torch.tensor(1.0))
        self.register_buffer('batch_count', torch.tensor(0))

    def forward(self, pred, true, curriculum_ratio=0.0):
        """
        Args:
            pred: [B, T, 3]
            true: [B, T, 3]
            curriculum_ratio: 0.0 ~ 1.0, 控制物理约束的介入程度
        """
        c = self.base_loss.get_raw_components(pred, true)

        loss_main = c['main']   # MSE

        # 1. 计算加权物理总损失 (原始量级)
        loss_phys_soft = (
                self.sub_weights['net'] * c['net'] +
                self.sub_weights['energy'] * c['energy'] +
                self.sub_weights['deriv'] * c['deriv'] +
                self.sub_weights['dir'] * c['dir']
        )

        loss_phys_hard = (
                self.sub_weights['bvr'] * c['bvr'] +
                self.sub_weights['rvr'] * c['rvr']
        )

        loss_phys_total_raw = loss_phys_soft + loss_phys_hard

        # 2. 更新 Scale EMA (仅训练时)
        if self.training:
            with torch.no_grad():
                if loss_phys_total_raw > 1e-6:
                    # 我们希望 Scale * Phys ≈ Main
                    current_scale = loss_main.detach() / loss_phys_total_raw.detach()
                    # 限制突变
                    current_scale = torch.clamp(current_scale, 0.01, 100.0)
                else:
                    current_scale = self.scale_ema

                self.batch_count += 1
                if self.batch_count <= self.warmup_batches:
                    self.scale_ema = current_scale
                else:
                    self.scale_ema = self.ema_decay * self.scale_ema + (1 - self.ema_decay) * current_scale

        # 3. 组合损失
        # 策略：Scale 将 Phys 拉到和 Main 同一水平线，然后 Ratio 决定你要加多少
        # 当 Ratio=1.0 时，物理 Loss 和 数据 Loss 贡献 1:1

        # Soft: 全程参与，但受 Ratio 影响 (Min 0.1)
        # 让模型一开始就有一点点物理感知，防止跑太偏
        lambda_soft = self.scale_ema * (0.1 + 0.9 * curriculum_ratio)

        # Hard: 仅在课程后期介入
        lambda_hard = self.scale_ema * curriculum_ratio

        total_loss = loss_main + lambda_soft * loss_phys_soft + lambda_hard * loss_phys_hard

        # 4. 构建日志
        debug_dict = {
            'mse': c['main'].item(),
            'mae': c['mae'].item(),
            'net': c['net'].item(),
            'deriv': c['deriv'].item(),
            'energy': c['energy'].item(),
            'dir': c['dir'].item(),
            'bvr': c['bvr'].item(),
            'rvr': c['rvr'].item(),
            'scale': self.scale_ema.item(),
            'ratio': curriculum_ratio
        }

        return total_loss, debug_dict