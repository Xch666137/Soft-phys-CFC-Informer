"""TDD verification for P0-2: Load branch uses load history.

Verifies that when x_load_hist is provided, the load branch uses it
instead of x_net_hist for autoreg correction and GRU temporal input.

Run: set PYTHONPATH=. && python scripts/verify_p0_load_branch.py
"""

import torch
import numpy as np

from physformer.models.physformer.physical_layer import ExplicitVPPPhysicalLayer


# ---------------------------------------------------------------------------
# 1. Create a minimal PhysicalLayer
# ---------------------------------------------------------------------------
phys_layer = ExplicitVPPPhysicalLayer(
    d_model=64,
    weather_dim=3,
    battery_state_dim=2,
    time_feat_dim=8,
    load_gru_hidden=32,
    load_gru_use_temp=True,
    load_temp_model="mlp",
)
phys_layer.eval()
print("PhysicalLayer creation: OK")

# ---------------------------------------------------------------------------
# 2. Create different x_net_hist and x_load_hist
# ---------------------------------------------------------------------------
B, S, P = 2, 96, 24

# Make net and load DIFFERENT (net = load - pv - wind + batt)
x_net_hist = torch.ones(B, S, 1) * 2.0        # net ~2 MW
x_load_hist = torch.ones(B, S, 1) * 3.0        # load ~3 MW (different from net)
x_weather_hist = torch.randn(B, S, 3)
x_weather_future = torch.randn(B, P, 3)
x_battery_hist = torch.randn(B, S, 2)
x_mark_enc = torch.randn(B, S, 8)
y_mark = torch.randn(B, P, 8)

# ---------------------------------------------------------------------------
# 3. Forward pass: WITHOUT x_load_hist (backward compat, uses net)
# ---------------------------------------------------------------------------
with torch.no_grad():
    _, states_net = phys_layer(
        x_weather_hist=x_weather_hist,
        x_weather_future=x_weather_future,
        y_mark=y_mark,
        x_net_hist=x_net_hist,
        x_battery_hist=x_battery_hist,
        portfolio_ids=None,
        x_mark_enc=x_mark_enc,
    )

# ---------------------------------------------------------------------------
# 4. Forward pass: WITH x_load_hist (uses load history)
# ---------------------------------------------------------------------------
with torch.no_grad():
    _, states_load = phys_layer(
        x_weather_hist=x_weather_hist,
        x_weather_future=x_weather_future,
        y_mark=y_mark,
        x_net_hist=x_net_hist,
        x_battery_hist=x_battery_hist,
        portfolio_ids=None,
        x_mark_enc=x_mark_enc,
        x_load_hist=x_load_hist,
    )

# ---------------------------------------------------------------------------
# 5. Verify
# ---------------------------------------------------------------------------
load_theory_net = states_net["load_theory_real"]
load_theory_load = states_load["load_theory_real"]

diff = (load_theory_net - load_theory_load).abs().mean().item()
print(f"\nLoad theory MAE (net vs load input): {diff:.6f}")
print(f"  Different: {diff > 1e-6}")

# The load theory should differ when different input histories are provided
# (net=2 vs load=3 → different autoreg and GRU input → different output)
if diff > 1e-6:
    print("  -> Load branch correctly uses x_load_hist when provided")
else:
    print("  -> WARNING: Load theory unchanged — x_load_hist may not be wired")

# Other components should be unchanged (they don't depend on load/net history)
pv_diff = (states_net["pv_theory_real"] - states_load["pv_theory_real"]).abs().mean().item()
wind_diff = (states_net["wind_theory_real"] - states_load["wind_theory_real"]).abs().mean().item()
print(f"PV theory diff:  {pv_diff:.2e} (should be ~0)")
print(f"Wind theory diff: {wind_diff:.2e} (should be ~0)")

# Verify backward compat: call without x_load_hist should still work
with torch.no_grad():
    _, states_bc = phys_layer(
        x_weather_hist=x_weather_hist,
        x_weather_future=x_weather_future,
        y_mark=y_mark,
        x_net_hist=x_net_hist,
        x_battery_hist=x_battery_hist,
        portfolio_ids=None,
        x_mark_enc=x_mark_enc,
        # no x_load_hist — backward compat
    )
bc_diff = (states_bc["load_theory_real"] - load_theory_net).abs().mean().item()
print(f"Backward compat diff: {bc_diff:.2e} (should be ~0)")

print()
print("=" * 60)
all_pass = diff > 1e-6 and pv_diff < 1e-4 and wind_diff < 1e-4 and bc_diff < 1e-8
print(f"ALL CHECKS PASS: {all_pass}")
