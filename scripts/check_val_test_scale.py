"""
V-1: Compute target_std from Val/Test MSE relationship.
Theory: test_mse = val_mse * target_std^2  (since val is normalized, test is in MW^2)
"""
import json
import os
import sys

results_dir = r"C:\Users\Xch\.codex\worktrees\7c57\Physformer\results"

# Find all experiments that have both train.log and metrics.json
experiments = []
for root, dirs, files in os.walk(results_dir):
    if "metrics.json" in files:
        experiments.append(root)

print(f"{'Experiment':<55} {'Best Val MSE':>13} {'Test MSE':>13} {'Ratio':>10} {'target_std':>12}")
print("-" * 110)

for exp_dir in sorted(experiments):
    metrics_path = os.path.join(exp_dir, "metrics.json")
    log_path = os.path.join(exp_dir, "train.log")

    if not os.path.exists(log_path):
        continue

    with open(metrics_path) as f:
        metrics = json.load(f)

    test_mse = metrics.get("mse")
    if test_mse is None or test_mse == 0:
        continue

    # Extract best Val MSE from train.log
    best_val_mse = None
    try:
        with open(log_path, errors='ignore') as f:
            for line in f:
                if "Validation Net MSE" in line and "best:" in line:
                    # "EarlyStopping counter: X out of Y (Validation Net MSE best: Z.ZZZZZZ)"
                    parts = line.split("best:")
                    if len(parts) > 1:
                        best_val_mse = float(parts[1].strip().rstrip(")"))
    except:
        pass

    if best_val_mse is None:
        continue

    ratio = best_val_mse / test_mse
    target_std = (test_mse / best_val_mse) ** 0.5 if best_val_mse > 0 else 0

    short_name = exp_dir.replace(results_dir, "")[-50:]
    print(f"{short_name:<55} {best_val_mse:13.6f} {test_mse:13.3e} {ratio:10.1f} {target_std:12.6f}")

print("\nInterpretation: test_mse = val_mse * target_std^2")
print("Consistent target_std across experiments confirms the scale relationship.")
print("If target_std is consistent, Val/Test 'divergence' is just unit conversion.")
