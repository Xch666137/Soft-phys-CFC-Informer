"""
D-4: Collapse root cause diagnosis.
Diagnose why physics branches collapse to ~0.0145 MW (Load) / ~0.02135 MW (Battery)
across three trigger conditions (d512, α≤0.5, no_temp_s2026).

Three hypotheses:
  H1 — Constant output: gradient death → frozen params → near-constant branch output
  H2 — Softplus saturation: F.softplus(load_pre) saturates at 0 or linear regime
  H3 — Gradient competition: data gradient overwhelms physics gradient → Adam
       adaptive LR systematically suppresses physics parameter updates

Usage: python scripts/diagnose_collapse.py <results_dir> [--checkpoint]
"""
import argparse
import json
import os
import sys

import numpy as np


def diagnose_from_physics_states(npz_path):
    """H1: Check if physics branch outputs are near-constant."""
    data = np.load(npz_path, allow_pickle=True)
    comp_theory = data.get("component_theory_real")
    if comp_theory is None:
        print("  [SKIP] No component_theory_real in npz")
        return None, None, None

    comp_names = ["Load", "PV", "Wind", "Batt_Power", "Batt_SOC"]
    results = {}
    for i, name in enumerate(comp_names):
        if i >= comp_theory.shape[-1]:
            break
        channel = comp_theory[..., i]
        std = float(np.std(channel))
        mean = float(np.mean(channel))
        cv = std / (abs(mean) + 1e-8)  # coefficient of variation
        results[name] = {"mean": mean, "std": std, "cv": cv}

    return results, comp_theory, data


def diagnose_h1(comp_stats):
    """H1: Constant output check — if CV < 0.05, output is near-constant."""
    print("\n  === H1: Constant Output ===")
    collapsed = []
    for name, stats in comp_stats.items():
        is_const = stats["cv"] < 0.05
        flag = "<<< COLLAPSED (const)" if is_const else ""
        print(f"  {name:15s}: mean={stats['mean']:.6f}  std={stats['std']:.6f}  cv={stats['cv']:.4f}  {flag}")
        if is_const:
            collapsed.append(name)
    if collapsed:
        print(f"  => H1 LIKELY: {collapsed} are near-constant (frozen parameters)")
    else:
        print("  => H1 unlikely: outputs vary (gradient is flowing)")
    return collapsed


def diagnose_h2(comp_theory, data, args):
    """H2: Softplus saturation — check if load_pre is in saturation regime."""
    print("\n  === H2: Softplus Saturation ===")
    # softplus(x): x <= -5 → ~0, x >= 10 → ~x
    # Without saved load_pre, we infer from the theory output range.
    # softplus(x) range: [0, ∞)
    # If theory ≈ 0 constantly → load_pre << 0
    # If theory ≈ some linear value → load_pre >> 0
    for i, name in enumerate(["Load", "PV", "Wind", "Batt_Power"]):
        if i >= comp_theory.shape[-1]:
            break
        channel = comp_theory[..., i]
        min_v = float(channel.min())
        max_v = float(channel.max())
        near_zero = (abs(channel) < 1e-4).mean()

        # Softplus output near 0 means input was heavily negative
        if near_zero > 0.5:
            print(f"  {name}: {near_zero*100:.0f}% near-zero → H2 possible (softplus saturated at 0)")
        # Softplus output ≈ input means input was large positive
        elif min_v > 1.0:
            print(f"  {name}: min={min_v:.3f} max={max_v:.3f} → H2 possible (softplus in linear regime)")
        else:
            print(f"  {name}: min={min_v:.6f} max={max_v:.6f} near_zero_frac={near_zero:.3f} → normal range")

    # Check for BatterySOC — no softplus there
    if comp_theory.shape[-1] >= 5:
        soc = comp_theory[..., 4]
        print(f"  Batt_SOC: min={float(soc.min()):.3f} max={float(soc.max()):.3f} (no softplus)")

    # If we have aux_mean/aux_std, show what collapse values mean in physical space
    print("\n  Note: Collapse MAE values (~0.0145 Load, ~0.02135 Battery) are in real MW.")
    print("  If branch output = constant, MAE = mean absolute deviation of true values from that constant.")


def diagnose_from_train_log(log_path, metrics_path):
    """H3: Check gradient norms from training log."""
    print("\n  === H3: Gradient Competition ===")
    if not os.path.exists(log_path):
        print("  [SKIP] No train.log found")
        return

    with open(log_path, errors='ignore') as f:
        lines = f.readlines()

    # Extract GradNorm(Net) and GradNorm(Theory) from each epoch
    norm_net_vals = []
    norm_theory_vals = []
    ratios = []

    for line in lines:
        if "GradNorm(Net):" in line and "GradNorm(Theory):" in line:
            try:
                net_part = line.split("GradNorm(Net):")[1].split("|")[0].strip()
                theory_part = line.split("GradNorm(Theory):")[1].strip().split()[0]
                n_net = float(net_part)
                n_theory = float(theory_part)
                norm_net_vals.append(n_net)
                norm_theory_vals.append(n_theory)
                ratios.append(n_net / (n_theory + 1e-8))
            except (ValueError, IndexError):
                pass

    if not norm_net_vals:
        print("  [SKIP] No gradient norm data in train.log")
        return

    print(f"  Epochs with gradient data: {len(norm_net_vals)}")
    print(f"  GradNorm(Net)   — mean: {np.mean(norm_net_vals):.2e}  median: {np.median(norm_net_vals):.2e}")
    print(f"  GradNorm(Theory) — mean: {np.mean(norm_theory_vals):.2e}  median: {np.median(norm_theory_vals):.2e}")
    print(f"  Ratio (Net/Theory) — mean: {np.mean(ratios):.1f}  median: {np.median(ratios):.1f}  max: {np.max(ratios):.1f}")

    # Check for sudden ratio spike (gradient competition signature)
    if len(ratios) > 10:
        first_half = np.mean(ratios[:len(ratios)//2])
        second_half = np.mean(ratios[len(ratios)//2:])
        print(f"  Ratio 1st half: {first_half:.1f}  2nd half: {second_half:.1f}")
        if second_half > first_half * 3:
            print("  => H3 LIKELY: gradient ratio spiked → data path dominated physics path")
        elif np.mean(norm_theory_vals) < 1e-5:
            print("  => H3 LIKELY: GradNorm(Theory) dead → Adam suppressed physics updates")
        else:
            print("  => H3 unclear: no clear gradient competition signature")

    # Check for epoch-by-epoch trend
    print(f"  Epoch    GradNorm(Net)      GradNorm(Theory)    Ratio")
    for i in range(min(len(norm_net_vals), 12)):  # first 12 epochs
        r = ratios[i]
        flag = " <-- SPIKE" if i > 0 and r > ratios[0] * 5 else ""
        print(f"  {i+1:5d}    {norm_net_vals[i]:<18.2e} {norm_theory_vals[i]:<18.2e} {r:<8.1f}{flag}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose physics branch collapse root cause")
    parser.add_argument("results_dir", help="Path to experiment results directory")
    parser.add_argument("--checkpoint", action="store_true",
                        help="Also analyze training state checkpoint (requires torch)")
    args = parser.parse_args()

    results_dir = args.results_dir
    npz_path = os.path.join(results_dir, "extras", "physics_states.npz")
    log_path = os.path.join(results_dir, "train.log")
    metrics_path = os.path.join(results_dir, "metrics.json")

    print(f"Diagnosing: {results_dir}")
    print(f"=" * 70)

    # Check if this is a collapsed experiment
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        load_mae = m.get("component_load_mae", 0)
        batt_mae = m.get("component_battery_power_mae", 0)
        print(f"\nComponent MAE | Load: {load_mae:.6f} | Battery: {batt_mae:.6f}")
        if load_mae > 0.01 or batt_mae > 0.01:
            print("=> LIKELY COLLAPSED: component MAE >> normal (~0.002)")
        else:
            print("=> Normal component MAE — not a collapsed experiment")

    # H1 & H2
    if os.path.exists(npz_path):
        comp_stats, comp_theory, npz_data = diagnose_from_physics_states(npz_path)
        if comp_stats:
            diagnose_h1(comp_stats)
            diagnose_h2(comp_theory, npz_data, args)
    else:
        print(f"\n[SKIP] physics_states.npz not found at {npz_path}")

    # H3
    diagnose_from_train_log(log_path, metrics_path)

    print("\n" + "=" * 70)
    print("Summary: Check the flagged hypotheses above. Cross-reference with the")
    print("three known trigger conditions (d512, α≤0.5, no_temp_s2026) for pattern.")
    print("If H1+H2 are ruled out but H3 is flagged → gradient competition mechanism.")
    print("If H1 is flagged in all three triggers → shared underlying collapse pathway.")

if __name__ == "__main__":
    main()
