"""
PhysFormer forward pass stub — component-consistent residual (V5).

This is a reference stub. The canonical implementation lives in:
  physformer/models/physformer.py
  physformer/models/conditioning.py
  physformer/models/temporal_decoder.py

Only the novel contribution (component-consistent forward pass) is shown.
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class PhysFormerV5(nn.Module):
    """PhysFormer with per-component residual correction (V5)."""

    def forward(
        self,
        x_net: torch.Tensor,       # (B, L_in, 1)  historical net injection
        x_phys: torch.Tensor,      # (B, L_in, F_phys) physics features
        y_mark: torch.Tensor,      # (B, L_in+L_out, F_time) time marks
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          pred_net:         (B, L_out, 1)  aggregated net forecast
          pred_components:  (B, L_out, 5)  per-component predictions
          theory_components: (B, L_out, 5) theory-only estimates
        """
        # --- Encode with FiLM conditioning ---
        h_enc = self.encoder(x_net, x_phys, y_mark[:, :self.L_in])  # (B, L_in, d_model)

        # --- Theory branches (physically-driven, per-component) ---
        pv_theory = self._pv_theory(x_phys)         # P = η·G·(1+αΔT)
        wind_theory = self._wind_theory(x_phys)      # P ∝ v³ (learned cubic)
        batt_theory = self._batt_theory(x_phys)      # SOC recurrence
        load_theory = self._load_theory(y_mark)      # calendar embedding

        theory = torch.stack([load_theory, pv_theory, wind_theory, batt_theory], dim=-1)

        # --- Decode with time conditioning ---
        y_future = y_mark[:, self.L_in:, :]          # future time marks only
        h_dec = self.temporal_decoder(h_enc, self.time_proj(y_future))  # (B, L_out, d_dec)

        # --- Per-component residual heads (key V5 contribution) ---
        r_load  = self.residual_head_load(h_dec)      # (B, L_out, 1)
        r_pv    = self.residual_head_pv(h_dec)
        r_wind  = self.residual_head_wind(h_dec)
        r_batt  = self.residual_head_batt(h_dec)
        r_soc   = self.residual_head_soc(h_dec)       # Battery SOC residual

        # --- Component predictions (theory + residual) ---
        load_pred  = load_theory + r_load
        pv_pred    = pv_theory + r_pv
        wind_pred  = wind_theory + r_wind
        batt_pred  = batt_theory + r_batt

        # --- Aggregate via power balance identity ---
        # net = load - pv - wind + batt
        pred_net = load_pred - pv_pred - wind_pred + batt_pred

        pred_components = torch.stack(
            [load_pred, pv_pred, wind_pred, batt_pred, r_soc], dim=-1
        )
        theory_components = torch.stack(
            [load_theory, pv_theory, wind_theory, batt_theory,
             torch.zeros_like(r_soc)],  # SOC theory tracked separately
            dim=-1
        )

        return pred_net, pred_components, theory_components
