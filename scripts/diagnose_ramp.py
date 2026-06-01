"""
P0-2: Diagnose why RampViol is 0.0% across all C23 experiments.
Checks: (1) net_ramp_limit value, (2) whether diff values exceed it,
(3) comparison with seed=2024 experiments where RampViol was non-zero.
"""
import json
import os
import sys
import numpy as np

results_dir = r"C:\Users\Xch\.codex\worktrees\7c57\Physformer\results"

def check_run(name):
    mpath = os.path.join(results_dir, name, "metrics.json")
    lpath = os.path.join(results_dir, name, "train.log")

    result = {"name": name, "ramp_limit": None, "ramp_viol": None, "has_pred": False}

    if os.path.exists(mpath):
        with open(mpath) as f:
            m = json.load(f)
        result["ramp_viol"] = m.get("net_ramp_violation")

    if os.path.exists(lpath):
        with open(lpath, encoding='utf-8', errors='ignore') as f:
            for line in f:
                if "net_ramp_limit" in line:
                    try:
                        result["ramp_limit"] = float(line.split("net_ramp_limit=")[1].strip())
                    except:
                        pass
                    break

    # Check if pred.npy exists for analysis
    ppath = os.path.join(results_dir, name, "pred.npy")
    result["has_pred"] = os.path.exists(ppath)
    if result["has_pred"]:
        pred = np.load(ppath)
        diff = np.abs(pred[:, 1:] - pred[:, :-1])
        result["diff_mean"] = float(diff.mean())
        result["diff_max"] = float(diff.max())
        result["diff_p999"] = float(np.quantile(diff, 0.999))

        if result["ramp_limit"] is not None:
            violations = (diff > result["ramp_limit"]).mean() * 100
            result["computed_viol"] = float(violations)

    return result

# Check C23 experiments vs old P1 experiments
experiments = [
    # C23 (all zero)
    "physformer_c23_baseline", "physformer_c23_detach", "physformer_c23_e3",
    # Old P1 (had non-zero in V6.1)
    "physformer_p1a_baseline_s2025", "physformer_p1a_detach_s2025",
    "physformer_p1a_baseline_s2026", "physformer_p1a_detach_s2026",
    # V6.1 (had non-zero ramp)
    "physformer_v6_1_baseline", "physformer_v6_1_detach", "physformer_v6_1_no_temp",
]

print(f"{'Experiment':<40} {'RampLimit':>10} {'RampViol':>10} {'HasPred':>8}")
print("-" * 75)
for name in experiments:
    r = check_run(name)
    rv = f"{r['ramp_viol']:.4f}" if r['ramp_viol'] is not None else "N/A"
    rl = f"{r['ramp_limit']:.6f}" if r['ramp_limit'] is not None else "N/A"
    print(f"{name:<40} {rl:>10} {rv:>10} {str(r['has_pred']):>8}")

# If pred files exist for any, do detailed diff analysis
print("\n--- Detailed diff analysis (experiments with pred.npy) ---")
for name in experiments:
    r = check_run(name)
    if r.get("computed_viol") is not None:
        print(f"{name}: limit={r['ramp_limit']:.6f} diff_max={r['diff_max']:.6f} computed_viol={r['computed_viol']:.4f}%")

print("\nRoot causes for RampViol=0:")
print("  (A) net_ramp_limit too large → diffs never exceed it → violation=0 legitimately")
print("  (B) net_ramp_limit=0.0 → violations=diff>0=False → function returns 0 from line 27")
print("  (C) Pred array not saved → can't verify the calculation")
print("  (D) Seed=2025 test set has different ramp characteristics than seed=2024")
