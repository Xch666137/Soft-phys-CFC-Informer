"""
V-3: Validate d512 "Val best but Test worst" interpretation.
Maps batch log files to experiment metrics.json and checks Val/Test scale consistency.
"""
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import json
import os
import re
import sys

results_dir = r"C:\Users\Xch\.codex\worktrees\7c57\Physformer\results"

# Map: experiment_dir -> (log_file_pattern, batch_log_dir)
mapping = [
    # Batch 1 (seed=2025) — logs from p1_batch1/
    ("physformer_p1a_baseline_s2025", "p1_batch1/baseline_s2025.log"),
    ("physformer_p1a_detach_s2025", "p1_batch1/detach_s2025.log"),
    ("physformer_p1a_notemp_s2025", "p1_batch1/notemp_s2025.log"),
    # Batch 2 (seed=2026) — logs from p1_batch2/ or runs/
    ("physformer_p1a_baseline_s2026", "p1_batch2/baseline_s2026.log"),
    ("physformer_p1a_detach_s2026", "physformer_p1a_detach_s2026/train.log"),
    ("physformer_p1a_notemp_s2026", "physformer_p1a_notemp_s2026/train.log"),
    # Batch 3 (gradient scaling, seed=2024) — logs from runs/
    ("physformer_p1a_detach_a03", "physformer_p1a_detach_a03/train.log"),
    ("physformer_p1a_detach_a05", "physformer_p1a_detach_a05/train.log"),
    ("physformer_p1a_detach_a07", "physformer_p1a_detach_a07/train.log"),
    # Batch 4 (encoder bottleneck, seed=2024) — logs from runs/
    ("physformer_p1b_d512", "physformer_p1b_d512/train.log"),
    ("physformer_p1b_e3", "physformer_p1b_e3/train.log"),
    ("physformer_p1b_d512_e3", "physformer_p1b_d512_e3/train.log"),
]

print(f"{'Experiment':<40} {'Best Val MSE':>13} {'Test MSE':>13} {'Ratio':>10} {'target_std':>12} {'Val->Test*':>13} {'Match?':>8}")
print("-" * 115)

results_summary = []

for exp_dir, log_rel in mapping:
    metrics_path = os.path.join(results_dir, exp_dir, "metrics.json")
    log_path = os.path.join(results_dir, log_rel)

    if not os.path.exists(metrics_path):
        print(f"{exp_dir:<40} {'SKIP: no metrics.json':>13}")
        continue

    with open(metrics_path) as f:
        metrics = json.load(f)
    test_mse = metrics.get("mse")
    test_mae = metrics.get("mae")
    test_rmse = metrics.get("rmse")
    theory_mae = metrics.get("theory_mae")

    # Extract best Val MSE from log
    best_val_mse = None
    if os.path.exists(log_path):
        with open(log_path, errors='ignore') as f:
            for line in f:
                if "Validation Net MSE" in line and "best:" in line:
                    parts = line.split("best:")
                    if len(parts) > 1:
                        best_val_mse = float(parts[1].strip().rstrip(")"))
    else:
        print(f"{exp_dir:<40} {'SKIP: no log file':>13}")
        continue

    if test_mse is None or test_mse == 0:
        print(f"{exp_dir:<40} {'SKIP: no test MSE':>13}")
        continue

    ratio = best_val_mse / test_mse
    target_std_est = (test_mse / best_val_mse) ** 0.5 if best_val_mse > 0 else 0
    val_to_test = best_val_mse * (0.0045 ** 2)  # expected test MSE using reference target_std
    match = "YES" if abs(val_to_test - test_mse) / (test_mse + 1e-10) < 0.15 else "NO"

    short = exp_dir.replace("physformer_p1", "")
    print(f"{short:<40} {best_val_mse:13.6f} {test_mse:13.3e} {ratio:10.1f} {target_std_est:12.6f} {val_to_test:13.3e} {match:>8}")

    results_summary.append({
        "exp": short,
        "val_mse": best_val_mse,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "theory_mae": theory_mae,
        "target_std_est": target_std_est,
        "match": match,
    })

print("\n--- d512 Overfitting Check ---")
print("If d512's Val->Test prediction matches (match=YES), the 'overfitting' narrative is wrong.")
print("If d512 consistently has match=NO (Val predicts BETTER Test than observed), overfitting is plausible.\n")

# Sort by test_mae to rank
ranked = sorted(results_summary, key=lambda x: x["test_mae"])
for i, r in enumerate(ranked):
    flag = ""
    if "d512" in r["exp"]:
        flag = " <-- d512 config"
    if "e3" in r["exp"] and "d512" not in r["exp"]:
        flag = " <-- e3 config"
    print(f"  {i+1:2d}. {r['exp']:<42s} Test MAE={r['test_mae']:.6f}  MSE={r['test_mse']:.3e}  Val MSE={r['val_mse']:.4f}  Match={r['match']}{flag}")

print("\nV-3 Conclusion:")
d512_entries = [r for r in results_summary if "d512" in r["exp"]]
d512_matches = [r for r in d512_entries if r["match"] == "YES"]
if len(d512_matches) == len(d512_entries):
    print(f"  All {len(d512_entries)} d512 experiments: Val→Test scale is CONSISTENT.")
    print("  'd512 best Val but worst Test' narrative is an ARTIFACT of comparing different units.")
    print("  FIX: d512 actually had best Val MSE in normalized space; this predicts")
    print("  the corresponding Test MSE via target_std². No overfitting occurred.")
else:
    non_matches = [r for r in d512_entries if r["match"] == "NO"]
    print(f"  {len(non_matches)}/{len(d512_entries)} d512 experiments show Val/Test DIVERGENCE.")
    print("  Overfitting interpretation is plausibly CORRECT for these runs.")
