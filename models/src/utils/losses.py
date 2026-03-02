import torch
import torch.nn as nn
import torch.nn.functional as F


class GateResponseRegularization(nn.Module):
    """
    门控响应度动态正则化
    强制网络的 soft gate 输出必须与物理先验在时序上保持高度正相关
    """
    def __init__(self, weight=0.05):
        super().__init__()
        self.weight = weight

    def forward(self, gate_soft, prior):
        # 1. 沿通道维度 D 取均值，得到时序曲线 [B, S]
        gate_curve = gate_soft.mean(dim=-1)
        prior_curve = prior.mean(dim=-1)

        # 2. 时序中心化 (减去均值)
        gate_centered = gate_curve - gate_curve.mean(dim=-1, keepdim=True)
        prior_centered = prior_curve - prior_curve.mean(dim=-1, keepdim=True)

        # 3. 计算皮尔逊相关系数 (Pearson Correlation)
        # BUGFIX: torch.norm 在输入全0时切线斜率为无穷大，会导致反向传播产生 NaN
        # 注意：由于外部已强制转换为 fp32 计算 Loss，此处 eps 恢复为 1e-8
        eps = 1e-8
        numerator = (gate_centered * prior_centered).sum(dim=-1)
        
        norm_gate = torch.sqrt(torch.sum(gate_centered**2, dim=-1) + eps)
        norm_prior = torch.sqrt(torch.sum(prior_centered**2, dim=-1) + eps)
        
        # 即使 eps 保护了 sqrt 不为 0，防止极端异常，我们在分母里额外加一个强保护 eps
        denominator = norm_gate * norm_prior + eps
        
        correlation = numerator / denominator  # [B]

        # 4. 动态掩码：只惩罚那些物理先验本身就有剧烈波动的场景
        # 比如夜间 prior 全是 0，方差为 0，就不应该惩罚相关性
        prior_variance = prior_centered.var(dim=-1)  # [B]
        active_mask = (prior_variance > 1e-3).float()

        # 5. 计算 Loss：强迫相关性逼近 1
        loss = self.weight * (active_mask * (1.0 - correlation)).mean()

        # 返回 loss 和监控用的平均相关系数
        # 使用更大的 eps 防止除以0
        avg_corr = (active_mask * correlation).sum() / (active_mask.sum() + eps)
        return loss, avg_corr.item()


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
        loss_main = F.mse_loss(pred, true)

        pred_real = pred * self.stds + self.means
        true_real = true * self.stds + self.means

        net_pred = pred_real[..., 0] - pred_real[..., 1] - pred_real[..., 2]
        net_true = true_real[..., 0] - true_real[..., 1] - true_real[..., 2]
        loss_net = F.l1_loss(net_pred, net_true)

        diff_pred = pred_real[:, 1:] - pred_real[:, :-1]
        diff_true = true_real[:, 1:] - true_real[:, :-1]
        loss_deriv = F.l1_loss(diff_pred, diff_true)

        pred_energy = torch.mean(pred_real, dim=1)
        true_energy = torch.mean(true_real, dim=1)
        loss_energy = F.l1_loss(pred_energy, true_energy)

        # BUGFIX: 避免使用 torch.norm 产生 NaN 梯度，改用基于 eps 的安全求模
        # 注意：由于外部已强制使用 fp32 计算，此处 eps 恢复为标准的 1e-8
        eps = 1e-8
        norm_diff_pred = torch.sqrt(torch.sum(diff_pred**2, dim=-1, keepdim=True) + eps)
        norm_diff_true = torch.sqrt(torch.sum(diff_true**2, dim=-1, keepdim=True) + eps)
        
        diff_pred_norm = diff_pred / norm_diff_pred
        diff_true_norm = diff_true / norm_diff_true
        cos_sim = (diff_pred_norm * diff_true_norm).sum(dim=-1)
        loss_dir = torch.mean(1.0 - cos_sim)

        # 将线性惩罚改为二次惩罚 (Quadratic Penalty)
        # BUGFIX: 避免 F.relu(0) 在二次求导下产生的 Subgradient NaN
        eps = 1e-8
        # bvr: lower bound violation (pred < 0) -> F.relu(-pred) -> smoothed
        bvr_raw = -pred_real
        loss_bvr = torch.mean((F.relu(bvr_raw) + eps) ** 2)

        # rvr: ramp violation (|diff| > limit) -> smoothed abs
        safe_abs_diff = torch.sqrt(diff_pred**2 + eps)
        ramp_violation = F.relu(safe_abs_diff - self.ramp_limits)
        loss_rvr = torch.mean((ramp_violation + eps) ** 2)

        loss_mae = F.l1_loss(pred_real, true_real)

        return {
            'main': loss_main, 'mae': loss_mae,
            'net': loss_net, 'deriv': loss_deriv,
            'energy': loss_energy, 'dir': loss_dir,
            'bvr': loss_bvr, 'rvr': loss_rvr
        }


class PhysLoss(nn.Module):
    """
    动态幅度平衡物理损失 v2 (Scale-Curriculum 解耦版)
    """

    def __init__(self, base_loss_module, device,
                 warmup_batches=50, ema_decay=0.9,
                 scale_clamp_min=0.1, scale_clamp_max=10.0,
                 scale_active_threshold=0.05):
        super().__init__()
        self.base_loss = base_loss_module
        self.device = device
        self.warmup_batches = warmup_batches
        self.ema_decay = ema_decay
        self.scale_clamp_min = scale_clamp_min
        self.scale_clamp_max = scale_clamp_max
        self.scale_active_threshold = scale_active_threshold

        self.sub_weights = {
            'net': 1.0, 'energy': 0.5, 'deriv': 0.3,
            'dir': 0.1, 'bvr': 2.0, 'rvr': 2.0
        }

        self.register_buffer('scale_ema', torch.tensor(1.0))
        self.register_buffer('batch_count', torch.tensor(0))

    def _compute_phys_ref(self, c):
        """原始参考物理Loss（不含curriculum权重），用于scale校准"""
        return (
            self.sub_weights['net']    * c['net']    +
            self.sub_weights['energy'] * c['energy'] +
            self.sub_weights['deriv']  * c['deriv']  +
            self.sub_weights['dir']    * c['dir']    +
            self.sub_weights['bvr']    * c['bvr']    +
            self.sub_weights['rvr']    * c['rvr']
        )

    def forward(self, pred, true, curriculum_weights=0.0):
        c = self.base_loss.get_raw_components(pred, true)
        loss_main = c['main']

        required_keys = ['net', 'energy', 'deriv', 'dir', 'bvr', 'rvr']
        if isinstance(curriculum_weights, (int, float)):
            ratio = float(curriculum_weights)
            curriculum_weights = {k: ratio for k in required_keys}
        else:
            for k in required_keys:
                if k not in curriculum_weights:
                    curriculum_weights[k] = 0.0

        avg_curriculum = sum(curriculum_weights.values()) / len(curriculum_weights)

        loss_phys_soft = (
            self.sub_weights['net']    * curriculum_weights['net']    * c['net']    +
            self.sub_weights['energy'] * curriculum_weights['energy'] * c['energy'] +
            self.sub_weights['deriv']  * curriculum_weights['deriv']  * c['deriv']  +
            self.sub_weights['dir']    * curriculum_weights['dir']    * c['dir']
        )
        loss_phys_hard = (
            self.sub_weights['bvr'] * curriculum_weights['bvr'] * c['bvr'] +
            self.sub_weights['rvr'] * curriculum_weights['rvr'] * c['rvr']
        )
        loss_phys_total = loss_phys_soft + loss_phys_hard

        if self.training:
            with torch.no_grad():
                self.batch_count += 1

                if self.batch_count <= self.warmup_batches:
                    # warmup阶段锁定scale=1.0
                    self.scale_ema = torch.tensor(1.0, device=self.device)

                elif avg_curriculum < self.scale_active_threshold:
                    # 物理约束几乎不活跃，冻结scale更新防止噪声污染EMA
                    pass

                else:
                    # [核心修复] 基于无curriculum权重的原始参考量计算scale
                    loss_phys_ref = self._compute_phys_ref(c)
                    # BUGFIX: 由于在外部已强制转换至 fp32 (float32)，此处阈值恢复为 1e-8 的安全下限
                    eps_scale = 1e-8
                    if loss_phys_ref > eps_scale:
                        # 使用真实物理量纲的 MAE (c['mae']) 来对比物理 Loss，实现量纲对齐！
                        current_scale = c['mae'].detach() / (loss_phys_ref.detach() + eps_scale)
                        current_scale = torch.clamp(
                            current_scale,
                            self.scale_clamp_min,
                            self.scale_clamp_max
                        )
                        self.scale_ema = (
                            self.ema_decay * self.scale_ema +
                            (1 - self.ema_decay) * current_scale
                        )

        total_loss = loss_main + self.scale_ema * loss_phys_total

        debug_dict = {
            'mse':    c['main'].item(),
            'mae':    c['mae'].item(),
            'net':    c['net'].item(),
            'deriv':  c['deriv'].item(),
            'energy': c['energy'].item(),
            'dir':    c['dir'].item(),
            'bvr':    c['bvr'].item(),
            'rvr':    c['rvr'].item(),
            'scale':  self.scale_ema.item(),
            'effective_phys_weight': self.scale_ema.item() * avg_curriculum,
        }
        for k in required_keys:
            debug_dict[f'curriculum_{k}'] = curriculum_weights[k]

        return total_loss, debug_dict