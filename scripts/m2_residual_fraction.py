"""
D2 — M2 mechanism candidate: residual fraction 重构.

Tests whether detach changes the *role* of the residual head (its absorption
function) vs baseline/e3, even when residual_std stays nearly equal.

Computed per run (from already-saved npy files, no checkpoint reload):
  - residual fraction  : |r| / (|theory| + |r|)       per sample, then quantiles
  - Pearson r(theory_net, residual_net)               per run
  - K-S test on residual-fraction distributions       detach vs baseline per seed
  - residual / true ratio quantiles                   sanity check

Falsification (see docs/plans/2026-05-25-detach-mechanism-search.md §4 D2):
  PASS if:  K-S p < 0.05 on residual fraction (detach vs baseline) in ≥ 2/3 seed
            AND mean |ΔPearson(theory, residual)| across seeds > 0.1
            AND ≥ 1 of {Q25, Q50, Q75} of residual fraction shows > 10%
                relative difference detach vs baseline
  FAIL if:  any single condition fails.

Headline metric: K-S D statistic (effect size, not p — p is meaningless at N≈3M).

Usage:
    & "E:\\Miniconda3\\envs\\PhysFormer\\python.exe" scripts/m2_residual_fraction.py
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.stats import ks_2samp
except ImportError:
    print("[fatal] scipy required: conda install -n PhysFormer scipy")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
ARMS = ["baseline", "e3", "detach", "detach_e3"]
SEEDS = [2025, 2026, 2027]


def run_dir(arm: str, seed: int) -> Path:
    return ROOT / "runs" / f"physformer_c23_{arm}_vgpu_s{seed}"


def msd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def per_run_stats(rd: Path) -> dict:
    theory = np.load(rd / "extras" / "theory_net.npy").astype(np.float64).ravel()
    residual_raw = np.load(rd / "extras" / "residual_net.npy").astype(np.float64)
    # (N, T, 5) → aggregate sign-weighted sum: load - pv - wind + batt
    residual = (residual_raw[..., 0] - residual_raw[..., 1]
                - residual_raw[..., 2] + residual_raw[..., 3]).ravel()

    abs_t = np.abs(theory)
    abs_r = np.abs(residual)
    denom = abs_t + abs_r
    safe = denom > 1e-30
    fraction = np.zeros_like(theory)
    fraction[safe] = abs_r[safe] / denom[safe]

    cov_tr = float(np.cov(theory, residual, ddof=1)[0, 1])
    sd_t = float(theory.std(ddof=1))
    sd_r = float(residual.std(ddof=1))
    pearson = cov_tr / (sd_t * sd_r + 1e-30)

    return dict(
        n_samples=int(theory.size),
        residual_fraction=fraction,
        pearson_theory_residual=float(pearson),
        residual_mean=float(residual.mean()),
        residual_std=sd_r,
        residual_fraction_q25=float(np.quantile(fraction, 0.25)),
        residual_fraction_q50=float(np.quantile(fraction, 0.50)),
        residual_fraction_q75=float(np.quantile(fraction, 0.75)),
        residual_fraction_q95=float(np.quantile(fraction, 0.95)),
        residual_fraction_mean=float(fraction.mean()),
    )


def main():
    print("=" * 96)
    print("D2 / M2 diagnostic: residual fraction & corr(theory_net, residual_net)")
    print("=" * 96)

    raw: dict[tuple[str, int], dict] = {}
    available_arms: list[str] = []
    for arm in ARMS:
        any_seed = False
        for seed in SEEDS:
            rd = run_dir(arm, seed)
            if not (rd / "extras" / "theory_net.npy").exists():
                continue
            print(f"  loading {arm:<10} s{seed} ...")
            raw[(arm, seed)] = per_run_stats(rd)
            any_seed = True
        if any_seed:
            available_arms.append(arm)
    print(f"\n  Available arms: {available_arms}")

    # ------------------------------------------------------------------
    # 1) Per-arm aggregate of scalar fields (mean ± std)
    # ------------------------------------------------------------------
    scalar_fields = [
        ("pearson_theory_residual",   "{:+.4f}"),
        ("residual_fraction_mean",    "{:.4f}"),
        ("residual_fraction_q25",     "{:.4f}"),
        ("residual_fraction_q50",     "{:.4f}"),
        ("residual_fraction_q75",     "{:.4f}"),
        ("residual_fraction_q95",     "{:.4f}"),
        ("residual_std",              "{:.4e}"),
    ]
    print(f"\n  Per-arm aggregates (mean ± std across seeds):\n")
    header = f"  {'metric':<28}" + "".join(f" | {arm:<22}" for arm in available_arms)
    print(header)
    print("  " + "-" * (len(header) - 2))
    summary: dict[str, dict] = {}
    for field, fmt in scalar_fields:
        row = f"  {field:<28}"
        for arm in available_arms:
            vals = [raw[(arm, s)][field] for s in SEEDS if (arm, s) in raw]
            m, sd = msd(vals)
            summary.setdefault(arm, {})[field] = {"mean": m, "std": sd, "values": vals}
            row += f" | {fmt.format(m):>10} ± {fmt.format(sd):>8}"
        print(row)

    # ------------------------------------------------------------------
    # 2) K-S on residual fraction, detach vs baseline per seed
    # ------------------------------------------------------------------
    print("\n  K-S test on residual fraction: detach vs baseline, per seed:\n")
    ks_results: dict[int, dict] = {}
    if "baseline" in available_arms and "detach" in available_arms:
        for seed in SEEDS:
            if (("baseline", seed) in raw and ("detach", seed) in raw):
                a = raw[("baseline", seed)]["residual_fraction"]
                b = raw[("detach", seed)]["residual_fraction"]
                stat = ks_2samp(a, b)
                ks_results[seed] = {
                    "D": float(stat.statistic),
                    "p": float(stat.pvalue),
                    "n_baseline": int(a.size),
                    "n_detach": int(b.size),
                }
                print(f"    seed={seed}: D = {stat.statistic:.4f}   "
                      f"p = {stat.pvalue:.3e}   "
                      f"N = {a.size} vs {b.size}")
    n_low_p = sum(1 for v in ks_results.values() if v["p"] < 0.05)
    n_total_seeds = len(ks_results)

    # ------------------------------------------------------------------
    # 3) Δ Pearson detach vs baseline
    # ------------------------------------------------------------------
    if "baseline" in available_arms and "detach" in available_arms:
        deltas = []
        for seed in SEEDS:
            if (("baseline", seed) in raw and ("detach", seed) in raw):
                deltas.append(
                    raw[("detach", seed)]["pearson_theory_residual"]
                    - raw[("baseline", seed)]["pearson_theory_residual"]
                )
        m, sd = msd(deltas)
        delta_pearson = {"per_seed": deltas, "mean": m, "std": sd, "mean_abs": abs(m)}
    else:
        delta_pearson = {"mean_abs": float("nan")}

    print(f"\n  ΔPearson(theory, residual) detach vs baseline:")
    print(f"    per-seed: {delta_pearson.get('per_seed', [])}")
    print(f"    mean = {delta_pearson.get('mean', float('nan')):+.4f}   "
          f"|mean| = {delta_pearson.get('mean_abs', float('nan')):.4f}")

    # ------------------------------------------------------------------
    # 4) Quantile diff detach vs baseline
    # ------------------------------------------------------------------
    quantile_diffs: dict[str, float] = {}
    if "baseline" in available_arms and "detach" in available_arms:
        for q in ["residual_fraction_q25", "residual_fraction_q50", "residual_fraction_q75"]:
            bl = summary["baseline"][q]["mean"]
            de = summary["detach"][q]["mean"]
            rel = (de - bl) / max(abs(bl), 1e-30)
            quantile_diffs[q] = rel
        print(f"\n  Quantile diffs (detach - baseline) / |baseline|:")
        for q, rel in quantile_diffs.items():
            print(f"    {q:>28}: {rel:+.2%}")

    # ------------------------------------------------------------------
    # 5) Falsification verdict
    # ------------------------------------------------------------------
    verdict: dict[str, object] = {}
    ks_pass = n_low_p >= 2 and n_total_seeds >= 2
    pearson_pass = delta_pearson.get("mean_abs", 0.0) > 0.1
    quantile_pass = any(abs(v) > 0.10 for v in quantile_diffs.values())

    verdict["ks_low_p_count"] = n_low_p
    verdict["ks_pass"] = ks_pass
    verdict["delta_pearson_mean_abs"] = delta_pearson.get("mean_abs", float("nan"))
    verdict["pearson_pass"] = pearson_pass
    verdict["quantile_diffs"] = quantile_diffs
    verdict["quantile_pass"] = quantile_pass

    m2_overall = ks_pass and pearson_pass and quantile_pass
    verdict["M2_overall"] = m2_overall
    print(f"\n  Verdict:")
    print(f"    K-S p < 0.05 in {n_low_p}/{n_total_seeds} seeds → {'PASS' if ks_pass else 'FAIL'}")
    print(f"    |ΔPearson| > 0.1                              → {'PASS' if pearson_pass else 'FAIL'}")
    print(f"    ≥1 quantile >10% relative diff                 → {'PASS' if quantile_pass else 'FAIL'}")
    print(f"\n  M2 (residual fraction 重构) verdict: "
          f"{'SUPPORTED' if m2_overall else 'NOT SUPPORTED (so far)'}")

    # ------------------------------------------------------------------
    # 6) Persist (drop the per-sample fraction arrays; they're huge)
    # ------------------------------------------------------------------
    out = ROOT / "scripts" / "m2_residual_fraction_results.json"
    serializable = {
        "arms_available": available_arms,
        "seeds": SEEDS,
        "per_run": {
            f"{arm}_s{seed}": {k: v for k, v in raw[(arm, seed)].items()
                               if k != "residual_fraction"}
            for (arm, seed) in raw
        },
        "summary": summary,
        "ks_detach_vs_baseline": ks_results,
        "delta_pearson_detach_vs_baseline": delta_pearson,
        "quantile_diffs": quantile_diffs,
        "verdict": verdict,
    }
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2, default=float)
    print(f"\n  Results JSON: {out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
