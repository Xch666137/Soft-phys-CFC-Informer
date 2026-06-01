"""
C08 variance decomposition extended to T1 multi-seed: 3 arms x 3 seeds = 9 runs.

Verifies whether O37 "bias-clearance regime escape via detach" replicates across
seeds, not just on the May 23 single-seed (s2025).

Per arm we aggregate (mean +/- std across seeds) of: per-component bias, diagonal
variance, cross-covariance contribution, observed var(error_net), cancellation
ratio, and the detach-vs-baseline regime-diagnosis deltas.

Usage:
    set PYTHONPATH=C:/Users/Xch/.codex/worktrees/7c57/Physformer
    & "E:\\Miniconda3\\envs\\PhysFormer\\python.exe" scripts/c08_t1_3seed.py
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

# Reuse the single-seed analyzer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from c08_vgpu_3way import analyze_run, extract_y_aux, COMP_NAMES, NET_SIGN  # noqa

ROOT = Path(__file__).resolve().parent.parent
ARMS = ["baseline", "e3", "detach"]
SEEDS = [2025, 2026, 2027]
RUNS = [(arm, seed, f"physformer_c23_{arm}_vgpu_s{seed}") for arm in ARMS for seed in SEEDS]


def msd(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def fmt_msd(values, fmt="{:+.3e}"):
    m, s = msd(values)
    return (fmt + " +/- " + fmt).format(m, s)


def main():
    print("=" * 90)
    print("C08 variance decomposition: T1 3-seed x 3-arm (9 runs)")
    print("=" * 90)

    print("\n[1/3] Extracting y_aux from test set (single shared loader)...")
    y_aux = extract_y_aux()

    raw = {}  # (arm, seed) -> result dict from analyze_run
    for arm, seed, dirname in RUNS:
        rd = ROOT / "runs" / dirname
        print(f"  analyzing {arm:<10} s{seed} ...")
        raw[(arm, seed)] = analyze_run(arm, rd, y_aux)

    # Aggregate per-arm across seeds
    print("\n[2/3] Per-arm aggregate (mean +/- std across 3 seeds)\n")

    def per_arm(field):
        return {arm: [raw[(arm, s)][field] for s in SEEDS] for arm in ARMS}

    def per_arm_comp(field, comp):
        return {arm: [raw[(arm, s)][field][comp] for s in SEEDS] for arm in ARMS}

    scalar_fields = [
        ("agg_mae",            "{:.4e}"),
        ("agg_mse",            "{:.4e}"),
        ("theory_mae",         "{:.4e}"),
        ("cancellation_mean",  "{:.4f}"),
        ("var_sum_diag",       "{:.3e}"),
        ("cov_cross_term",     "{:+.3e}"),
        ("var_net_observed",   "{:.3e}"),
        ("var_net_predicted",  "{:.3e}"),
    ]
    head = f"  {'metric':<24} | {'baseline':<26} | {'e3':<26} | {'detach':<26}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for field, fmt in scalar_fields:
        d = per_arm(field)
        row = f"  {field:<24} |"
        for arm in ARMS:
            row += f" {fmt_msd(d[arm], fmt):<26} |"
        print(row)

    print("\n[2b] Per-component bias (mean across seeds, real MW):")
    head2 = f"  {'component':<16} | {'baseline':<28} | {'e3':<28} | {'detach':<28}"
    print(head2)
    print("  " + "-" * (len(head2) - 2))
    for comp in COMP_NAMES:
        d = per_arm_comp("bias_per_comp", comp)
        row = f"  {comp:<16} |"
        for arm in ARMS:
            m, s = msd(d[arm])
            row += f" {m:+.4e} +/- {s:.2e}      |"
        print(row)

    print("\n[2c] Per-component diag variance (mean across seeds, real MW^2):")
    print(head2)
    print("  " + "-" * (len(head2) - 2))
    for comp in COMP_NAMES:
        d = per_arm_comp("var_per_comp", comp)
        row = f"  {comp:<16} |"
        for arm in ARMS:
            m, s = msd(d[arm])
            row += f" {m:.4e} +/- {s:.2e}       |"
        print(row)

    print("\n[3/3] Detach-vs-Baseline regime diagnosis (per-seed delta then aggregate)\n")
    # Compute the deltas individually per seed, then aggregate
    diag_drops, cross_drops, agg_mse_drops, cancel_changes, bias_clearances = [], [], [], [], []
    bias_clear_per_comp = {comp: [] for comp in COMP_NAMES}
    for seed in SEEDS:
        bl = raw[("baseline", seed)]
        de = raw[("detach", seed)]
        diag_drops.append((de["var_sum_diag"] - bl["var_sum_diag"]) / bl["var_sum_diag"])
        cross_drops.append((de["cov_cross_term"] - bl["cov_cross_term"]) / abs(bl["cov_cross_term"]))
        agg_mse_drops.append((de["agg_mse"] - bl["agg_mse"]) / bl["agg_mse"])
        cancel_changes.append(de["cancellation_mean"] - bl["cancellation_mean"])
        # average per-component |bias| reduction
        bl_bias = np.array([abs(bl["bias_per_comp"][c]) for c in COMP_NAMES])
        de_bias = np.array([abs(de["bias_per_comp"][c]) for c in COMP_NAMES])
        bias_clearances.append(float(((de_bias - bl_bias) / (bl_bias + 1e-12)).mean()))
        for comp in COMP_NAMES:
            bl_b = abs(bl["bias_per_comp"][comp])
            de_b = abs(de["bias_per_comp"][comp])
            bias_clear_per_comp[comp].append((de_b - bl_b) / (bl_b + 1e-12))

    def pct_fmt(vals):
        m, s = msd(vals)
        return f"{m*100:+.2f}% +/- {s*100:.2f}%"

    print(f"  diag-variance shift   (detach - baseline)/baseline : {pct_fmt(diag_drops)}     per-seed: {[f'{v*100:+.1f}%' for v in diag_drops]}")
    print(f"  cov-cross shift       (detach - baseline)/|baseline|: {pct_fmt(cross_drops)}     per-seed: {[f'{v*100:+.1f}%' for v in cross_drops]}")
    print(f"  agg MSE shift         (detach - baseline)/baseline : {pct_fmt(agg_mse_drops)}     per-seed: {[f'{v*100:+.1f}%' for v in agg_mse_drops]}")
    print(f"  cancellation mean delta (detach - baseline)        : {fmt_msd(cancel_changes, '{:+.4f}')}     per-seed: {[f'{v:+.4f}' for v in cancel_changes]}")
    print(f"  per-comp |bias| mean reduction (detach vs baseline): {pct_fmt(bias_clearances)}     per-seed: {[f'{v*100:+.1f}%' for v in bias_clearances]}")
    print()
    print("  Per-component |bias| reduction (detach vs baseline), aggregate:")
    for comp in COMP_NAMES:
        print(f"      {comp:>16}: {pct_fmt(bias_clear_per_comp[comp])}     per-seed: {[f'{v*100:+.1f}%' for v in bias_clear_per_comp[comp]]}")

    # Replication test of O37: regime-escape signature
    # O37 claims: bias clearance + diag var reduction + weaker cancellation cov-cross
    print("\n  --- Regime diagnosis (replicated check) ---")
    diag_drop_m, _ = msd(diag_drops)
    cross_drop_m, _ = msd(cross_drops)
    bias_clear_m, _ = msd(bias_clearances)
    agg_mse_drop_m, _ = msd(agg_mse_drops)

    signals = {
        "bias_clearance":   bias_clear_m < -0.20,   # detach clears bias by >20%
        "diag_var_reduce":  diag_drop_m < -0.10,
        "weaker_cancel":    cross_drop_m > 0,        # less negative cov-cross = less cancellation
        "agg_mse_better":   agg_mse_drop_m < 0,
    }
    signal_values = {
        "bias_clearance":   f"{bias_clear_m:.2%}",
        "diag_var_reduce":  f"{diag_drop_m:.2%}",
        "weaker_cancel":    f"{cross_drop_m:.2%}",
        "agg_mse_better":   f"{agg_mse_drop_m:.2%}",
    }
    for sig, ok in signals.items():
        verdict = "YES" if ok else "NO"
        print(f"    {sig:<20}: {verdict:<3}  (mean = {signal_values[sig]})")

    n_yes = sum(signals.values())
    print(f"\n  O37 replication signals: {n_yes}/4")
    if n_yes >= 3:
        print("  => O37 corroborated across seeds: detach exhibits regime-escape signature.")
    elif n_yes == 2:
        print("  => partial replication; regime-escape claim weakened, mechanism mixed.")
    else:
        print("  => O37 NOT replicated; single-seed s2025 was an outlier.")

    # Save aggregated results
    out = ROOT / "scripts" / "c08_t1_3seed_results.json"
    serializable = {
        "by_run": {f"{arm}_s{seed}": raw[(arm, seed)] for arm in ARMS for seed in SEEDS},
        "regime_diagnosis": {
            "diag_drops_per_seed": diag_drops,
            "cross_drops_per_seed": cross_drops,
            "agg_mse_drops_per_seed": agg_mse_drops,
            "cancel_changes_per_seed": cancel_changes,
            "bias_clearances_per_seed": bias_clearances,
            "diag_drop_mean":  diag_drop_m,
            "cross_drop_mean": cross_drop_m,
            "agg_mse_drop_mean": agg_mse_drop_m,
            "bias_clearance_mean": bias_clear_m,
            "signals": signals,
            "n_yes": n_yes,
        },
    }
    # Strip cov_pairs to keep JSON small
    for k in serializable["by_run"]:
        serializable["by_run"][k] = {kk: vv for kk, vv in serializable["by_run"][k].items() if kk != "cov_pairs"}
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2, default=float)
    print(f"\n  Summary JSON: {out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
