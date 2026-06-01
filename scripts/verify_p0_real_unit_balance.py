"""TDD verification for P0-1: Real-unit power balance fix.

Demonstrates that the current _build_component_net mixes per-component
z-score spaces, producing a distorted pred_net that does NOT equal the
correct normalized net. Then verifies the fixed version.

Run: python scripts/verify_p0_real_unit_balance.py
"""

import torch
import numpy as np

# ---------------------------------------------------------------------------
# 1. Simulate realistic scaler params (order-of-magnitude from VPP data)
# ---------------------------------------------------------------------------
# aux = [load, pv, wind, batt_p, batt_soc]
aux_mean = torch.tensor([2.5, 0.5, 0.3, 0.0, 5.0], dtype=torch.float32)
aux_std = torch.tensor([1.0, 0.3, 0.2, 0.5, 3.0], dtype=torch.float32)

target_mean = torch.tensor([2.5], dtype=torch.float32)  # net = load - pv - wind + batt ≈ 1.7
target_std = torch.tensor([1.2], dtype=torch.float32)

# ---------------------------------------------------------------------------
# 2. Create synthetic component theory values in REAL MW
# ---------------------------------------------------------------------------
B, P = 4, 96
torch.manual_seed(42)

# Realistic component values
load_real = 2.0 + 0.8 * torch.randn(B, P, 1)     # ~2.0 MW
pv_real = 0.3 + 0.2 * torch.randn(B, P, 1)       # ~0.3 MW daytime avg
wind_real = 0.15 + 0.1 * torch.randn(B, P, 1)    # ~0.15 MW
batt_real = 0.1 * torch.randn(B, P, 1)            # ~0 MW mean
soc_real = 3.0 + 1.0 * torch.randn(B, P, 1)      # ~3.0 MWh

component_real = torch.cat([load_real, pv_real, wind_real, batt_real, soc_real], dim=-1)

# Ground truth: net = load - pv - wind + batt
net_real = load_real - pv_real - wind_real + batt_real
net_norm_correct = (net_real - target_mean.view(1, 1, -1)) / (
    target_std.view(1, 1, -1) + 1e-6
)

# ---------------------------------------------------------------------------
# 3. Simulate the CURRENT (buggy) code path
# ---------------------------------------------------------------------------
def norm_aux(x, mean, std):
    return (x - mean.view(1, 1, -1)) / (std.view(1, 1, -1) + 1e-6)

def norm_target(x, mean, std):
    return (x - mean.view(1, 1, -1)) / (std.view(1, 1, -1) + 1e-6)

component_norm = norm_aux(component_real, aux_mean, aux_std)

# Simulate residual head output (small values in same z-score space)
component_residual = 0.05 * torch.randn(B, P, 5)

# OLD (buggy): power balance in mixed z-score space
def build_component_net_old(component_norm, component_residual):
    load_th_res = component_norm[..., 0:1] + component_residual[..., 0:1]
    pv_th_res = component_norm[..., 1:2] + component_residual[..., 1:2]
    wind_th_res = component_norm[..., 2:3] + component_residual[..., 2:3]
    batt_th_res = component_norm[..., 3:4] + component_residual[..., 3:4]
    return load_th_res - pv_th_res - wind_th_res + batt_th_res

pred_net_old = build_component_net_old(component_norm, component_residual)

# ---------------------------------------------------------------------------
# 4. NEW (fixed): power balance in real MW
# ---------------------------------------------------------------------------
def build_component_net_new(component_real, component_residual, aux_std, target_mean, target_std):
    # Denorm residual: zero-mean delta → scale by aux_std only
    component_residual_real = component_residual * aux_std.view(1, 1, -1)

    # Add to theory in real units
    component_pred_real = component_real + component_residual_real

    # Power balance in real MW
    pred_net_real = (
        component_pred_real[..., 0:1]   # load
        - component_pred_real[..., 1:2] # pv
        - component_pred_real[..., 2:3] # wind
        + component_pred_real[..., 3:4] # batt_p
    )

    # Normalize to target space
    return (pred_net_real - target_mean.view(1, 1, -1)) / (
        target_std.view(1, 1, -1) + 1e-6
    )

pred_net_new = build_component_net_new(
    component_real, component_residual, aux_std, target_mean, target_std,
)

# ---------------------------------------------------------------------------
# 5. Verify
# ---------------------------------------------------------------------------
# 5a. Old code produces DIFFERENT output than correct normalized net
old_error = (pred_net_old - net_norm_correct).abs().mean().item()
print(f"OLD (buggy)  vs correct norm:  MAE = {old_error:.6f}")
print(f"  -> OLD is distorted: {old_error > 0.01}")

# 5b. New code: verify power balance is correct
# Reconstruct pred_net_real from pred_net_new
pred_net_real_from_new = pred_net_new * target_std.view(1, 1, -1) + target_mean.view(1, 1, -1)

# Reconstruct from components in real units
component_residual_real = component_residual * aux_std.view(1, 1, -1)
component_pred_real = component_real + component_residual_real
pred_net_real_reconstructed = (
    component_pred_real[..., 0:1]
    - component_pred_real[..., 1:2]
    - component_pred_real[..., 2:3]
    + component_pred_real[..., 3:4]
)

real_match_error = (pred_net_real_from_new - pred_net_real_reconstructed).abs().mean().item()
print(f"NEW (fixed) real-unit consistency:  MAE = {real_match_error:.10f}")
print(f"  -> Real-unit match: {real_match_error < 1e-6}")

# 5c. New code: compare to correct normalized net
new_error = (pred_net_new - net_norm_correct).abs().mean().item()
print(f"NEW (fixed) vs correct norm:       MAE = {new_error:.6f}")

# Without residual, new should exactly match correct
component_residual_zero = torch.zeros_like(component_residual)
pred_net_new_zero_res = build_component_net_new(
    component_real, component_residual_zero, aux_std, target_mean, target_std,
)
zero_res_error = (pred_net_new_zero_res - net_norm_correct).abs().mean().item()
print(f"NEW (zero residual) vs correct:    MAE = {zero_res_error:.10f}")
print(f"  -> Exact match (should be ~0): {zero_res_error < 1e-8}")

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
all_pass = (
    old_error > 0.01  # Bug confirmed
    and real_match_error < 1e-6  # New real-unit consistency
    and zero_res_error < 1e-8  # New exact match without residual
)
print(f"ALL CHECKS PASS: {all_pass}")
if not all_pass:
    print("FAILURES:")
    if old_error <= 0.01:
        print(f"  - Bug NOT confirmed (old_error={old_error:.6f} <= 0.01)")
    if real_match_error >= 1e-6:
        print(f"  - Real-unit consistency FAIL (error={real_match_error:.6e})")
    if zero_res_error >= 1e-8:
        print(f"  - Zero-residual exact match FAIL (error={zero_res_error:.6e})")
