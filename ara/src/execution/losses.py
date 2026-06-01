"""
PhysFormer loss function stub — curriculum training (V5).

Canonical implementation: physformer/utils/losses.py
"""

import torch
import torch.nn as nn
from typing import Optional


class PhysFormerLoss(nn.Module):
    """Combined aggregate MSE + component MAE with curriculum schedule."""

    def __init__(
        self,
        component_weight_max: float = 0.10,   # Phase 1 weight
        component_weight_min: float = 0.005,   # Phase 2 floor
        phase1_epochs: int = 15,               # Physics warmup
        phase2_epochs: int = 25,               # Annealing
        # Phase 3 removed in V5.5 (dead end)
    ):
        super().__init__()
        self.cw_max = component_weight_max
        self.cw_min = component_weight_min
        self.e1 = phase1_epochs
        self.e2 = phase1_epochs + phase2_epochs
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def curriculum_weight(self, epoch: int) -> float:
        """Piecewise linear curriculum schedule."""
        if epoch < self.e1:
            return self.cw_max
        elif epoch < self.e2:
            frac = (epoch - self.e1) / (self.e2 - self.e1)
            return self.cw_max + (self.cw_min - self.cw_max) * frac
        else:
            return self.cw_min

    def forward(
        self,
        pred_net: torch.Tensor,
        true_net: torch.Tensor,
        pred_components: torch.Tensor,  # (B, L_out, 5)
        true_components: torch.Tensor,  # (B, L_out, 5)
        epoch: int,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
          total_loss: scalar
          metrics: dict with {loss_net, loss_comp, comp_weight, ...}
        """
        loss_net = self.mse(pred_net, true_net)
        loss_comp = self.mae(pred_components, true_components)  # MAE in kW space
        lam = self.curriculum_weight(epoch)
        total = loss_net + lam * loss_comp

        metrics = {
            "loss_net": loss_net.item(),
            "loss_comp": loss_comp.item(),
            "comp_weight": lam,
            "total": total.item(),
        }
        return total, metrics
