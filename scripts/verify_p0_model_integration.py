"""Integration test for P0-1: import PhysFormer model, run forward pass,
verify real-unit power balance correctness.

Run: python scripts/verify_p0_model_integration.py
"""

import sys
import torch
import numpy as np

# ---------------------------------------------------------------------------
# 1. Import the fixed model
# ---------------------------------------------------------------------------
from physformer.models.physformer.model import PhysFormer

print("Model import: OK")

# ---------------------------------------------------------------------------
# 2. Create a minimal PhysFormer instance
# ---------------------------------------------------------------------------
# Realistic scaler params (order-of-magnitude VPP data)
aux_mean = [2.5, 0.5, 0.3, 0.0, 5.0]  # load, pv, wind, batt_p, batt_soc
aux_std  = [1.0, 0.3, 0.2, 0.5, 3.0]
target_mean = [2.5]
target_std  = [1.2]
weather_mean = [15.0, 400.0, 4.0]
weather_std  = [10.0, 300.0, 3.0]
state_mean = [0.0, 5.0]
state_std  = [0.3, 3.0]

model = PhysFormer(
    enc_in=6,
    seq_len=96,
    pred_len=24,
    d_model=64,
    n_heads=4,
    e_layers=1,
    d_ff=128,
    dropout=0.0,
    aux_mean=aux_mean,
    aux_std=aux_std,
    target_mean=target_mean,
    target_std=target_std,
    weather_mean=weather_mean,
    weather_std=weather_std,
    state_mean=state_mean,
    state_std=state_std,
    time_feat_dim=8,
    load_gru_hidden=32,
)
model.eval()
print("Model creation: OK")

# ---------------------------------------------------------------------------
# 3. Create synthetic input tensors
# ---------------------------------------------------------------------------
B, S, P = 2, 96, 24

x_net_hist = torch.randn(B, S, 1)
x_weather_hist = torch.randn(B, S, 3)
x_battery_hist = torch.randn(B, S, 2)
x_weather_future = torch.randn(B, P, 3)
x_mark_enc = torch.randn(B, S, 8)
y_mark = torch.randn(B, P, 8)

# ---------------------------------------------------------------------------
# 4. Run forward pass
# ---------------------------------------------------------------------------
with torch.no_grad():
    output = model(x_net_hist, x_weather_hist, x_battery_hist, x_weather_future,
                   x_mark_enc, y_mark, portfolio_ids=None)

pred_net = output["pred_net"]
theory_net = output["theory_net"]
component_residual = output["component_residual"]
physics_states = output["physics_states"]

print(f"pred_net shape:        {pred_net.shape}")
print(f"theory_net shape:      {theory_net.shape}")
print(f"component_residual:    {component_residual.shape}")

# ---------------------------------------------------------------------------
# 5. Verify: pred_net_real ≈ theory_net_real + residual_net_real
# ---------------------------------------------------------------------------
# Denorm pred_net to real MW
pred_net_real = pred_net * model.target_std.view(1, 1, -1) + model.target_mean.view(1, 1, -1)

# Denorm theory_net to real MW
theory_net_real_from_model = theory_net * model.target_std.view(1, 1, -1) + model.target_mean.view(1, 1, -1)

# Reconstruct from components
component_real = physics_states["component_theory_real"]
component_residual_real = component_residual * model.aux_std.view(1, 1, -1)
component_pred_real = component_real + component_residual_real
pred_net_real_reconstructed = (
    component_pred_real[..., 0:1]
    - component_pred_real[..., 1:2]
    - component_pred_real[..., 2:3]
    + component_pred_real[..., 3:4]
)

# Residual in real MW
residual_net_real = (
    component_residual_real[..., 0:1]
    - component_residual_real[..., 1:2]
    - component_residual_real[..., 2:3]
    + component_residual_real[..., 3:4]
)

# Theory net in real MW from components
theory_net_real_from_components = (
    component_real[..., 0:1]
    - component_real[..., 1:2]
    - component_real[..., 2:3]
    + component_real[..., 3:4]
)

# Check 1: pred_net_real matches component reconstruction
error_1 = (pred_net_real - pred_net_real_reconstructed).abs().mean().item()
print(f"\nCheck 1 - pred_net_real vs component reconstruction: MAE = {error_1:.2e}")
print(f"  PASS: {error_1 < 1e-5}")

# Check 2: pred_net_real = theory_net_real + residual_net_real
error_2 = (pred_net_real - (theory_net_real_from_components + residual_net_real)).abs().mean().item()
print(f"Check 2 - pred = theory + residual (real MW):          MAE = {error_2:.2e}")
print(f"  PASS: {error_2 < 1e-5}")

# Check 3: no NaN
has_nan = torch.isnan(pred_net).any().item()
print(f"Check 3 - No NaN in pred_net: {not has_nan}")

# Check 4: theory_net_real from model matches theory_net_real from components
error_4 = (theory_net_real_from_model - theory_net_real_from_components).abs().mean().item()
print(f"Check 4 - theory_net from model vs components:          MAE = {error_4:.2e}")

# ---------------------------------------------------------------------------
# 6. Also test no_phys_stream=True
# ---------------------------------------------------------------------------
model.no_phys_stream = True
with torch.no_grad():
    output_np = model(x_net_hist, x_weather_hist, x_battery_hist, x_weather_future,
                      x_mark_enc, y_mark, portfolio_ids=None)
has_nan_np = torch.isnan(output_np["pred_net"]).any().item()
print(f"\nCheck 5 - no_phys_stream mode: no NaN = {not has_nan_np}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
all_pass = (
    error_1 < 1e-5
    and error_2 < 1e-5
    and not has_nan
    and not has_nan_np
)
print(f"ALL CHECKS PASS: {all_pass}")
