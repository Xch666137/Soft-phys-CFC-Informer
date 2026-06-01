"""
D1 — M1 mechanism candidate: capacity-cancellation 解耦.

Tests whether detach (and detach×e3 if available) breaks the theory↔residual
cancellation coupling vs baseline/e3.

Two signal sets per run:
  (a) aggregate-level: cov(theory_net, residual_net), residual stats, theory stats
      — computed from runs/.../extras/{theory_net.npy, residual_net.npy}
  (b) per-component (from physics_states.npz): bias_per_comp, var_sum_diag,
      cov_cross_term — reuses c08_vgpu_3way.analyze_run for parity with N83.

Cross-seed aggregation (mean ± std across {2025, 2026, 2027}), per arm in
ARMS = ["baseline", "e3", "detach", "detach_e3"]. detach_e3 is skipped if
its run dirs are absent (E1 not yet complete).

Falsification (see docs/plans/2026-05-25-detach-mechanism-search.md §4 D1):
  PASS if:  detach mean |cov(theory, residual)| < 30% × baseline mean |cov|
            AND per-comp |bias| reduction count ≥ 9/15
  FAIL if:  either condition is violated.

Usage:
    set PYTHONPATH=C:/Users/Xch/.codex/worktrees/7c57/Physformer
    & "E:\\Miniconda3\\envs\\PhysFormer\\python.exe" scripts/m1_cov_cross_t1.py
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from c08_vgpu_3way import analyze_run, extract_y_aux, COMP_NAMES  # noqa: E402

ARMS = ["baseline", "e3", "detach", "detach_e3"]
SEEDS = [2025, 2026, 2027]


def run_dir(arm: str, seed: int) -> Path:
    return ROOT / "runs" / f"physformer_c23_{arm}_vgpu_s{seed}"


def analyze_aggregate(rd: Path) -> dict:
    """Aggregate-level theory vs residual statistics (1D over N×T samples).

    residual_net.npy is per-component (N, T, 5) with order [load, pv, wind, batt_p, batt_soc].
    Aggregate residual = load_res - pv_res - wind_res + batt_res per net equation.
    theory_net.npy is already aggregate (N, T, 1).
    """
    theory = np.load(rd / "extras" / "theory_net.npy").astype(np.float64).ravel()
    residual_raw = np.load(rd / "extras" / "residual_net.npy").astype(np.float64)
    # (N, T, 5) → aggregate sign-weighted sum
    # net = load - pv - wind + batt; soc not in net
    residual = (residual_raw[..., 0] - residual_raw[..., 1]
                - residual_raw[..., 2] + residual_raw[..., 3]).ravel()
    assert theory.shape == residual.shape, (theory.shape, residual.shape)

    cov = float(np.cov(theory, residual, ddof=1)[0, 1])
    var_theory = float(theory.var(ddof=1))
    var_resid = float(residual.var(ddof=1))
    pearson = cov / (np.sqrt(var_theory) * np.sqrt(var_resid) + 1e-30)

    return dict(
        n_samples=int(theory.size),
        cov_theory_residual=cov,
        pearson_theory_residual=float(pearson),
        theory_mean=float(theory.mean()),
        theory_std=float(np.sqrt(var_theory)),
        residual_mean=float(residual.mean()),
        residual_std=float(np.sqrt(var_resid)),
    )


def msd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def main():
    print("=" * 96)
    print("D1 / M1 diagnostic: cov(theory, residual) + per-component bias across arms × seeds")
    print("=" * 96)

    print("\n[1/4] Extracting y_aux for per-component decomposition ...")
    y_aux = extract_y_aux()

    print("\n[2/4] Analyzing each run (aggregate + per-component) ...\n")
    raw: dict[tuple[str, int], dict] = {}
    available_arms: list[str] = []
    for arm in ARMS:
        any_seed_present = False
        for seed in SEEDS:
            rd = run_dir(arm, seed)
            if not (rd / "extras" / "theory_net.npy").exists():
                print(f"  [skip] {arm:<10} s{seed}: theory_net.npy missing -> {rd.name}")
                continue
            any_seed_present = True
            print(f"  analyzing {arm:<10} s{seed} ...")
            agg = analyze_aggregate(rd)
            try:
                comp = analyze_run(arm, rd, y_aux)
            except Exception as exc:
                print(f"    [warn] per-comp analyze_run failed: {exc}")
                comp = {}
            raw[(arm, seed)] = {"agg": agg, "comp": comp}
        if any_seed_present:
            available_arms.append(arm)

    print(f"\n  Available arms with ≥1 seed: {available_arms}")

    # ----------------------------------------------------------------------
    # 3) Aggregate-level cov(theory, residual) per arm
    # ----------------------------------------------------------------------
    print("\n[3/4] Aggregate-level cov(theory, residual), per arm (mean ± std across seeds)\n")
    agg_field = "cov_theory_residual"

    header = f"  {'metric':<28}" + "".join(f" | {arm:<26}" for arm in available_arms)
    print(header)
    print("  " + "-" * (len(header) - 2))

    agg_summary: dict[str, dict] = {}
    for field, fmt in [
        ("cov_theory_residual",     "{:+.3e}"),
        ("pearson_theory_residual", "{:+.4f}"),
        ("theory_mean",             "{:+.4e}"),
        ("theory_std",              "{:.4e}"),
        ("residual_mean",           "{:+.4e}"),
        ("residual_std",            "{:.4e}"),
    ]:
        row = f"  {field:<28}"
        for arm in available_arms:
            vals = [raw[(arm, s)]["agg"][field] for s in SEEDS if (arm, s) in raw]
            m, sd = msd(vals)
            agg_summary.setdefault(arm, {})[field] = {"mean": m, "std": sd, "values": vals}
            row += f" | {fmt.format(m):>10} ± {fmt.format(sd):>10}"
        print(row)

    # ----------------------------------------------------------------------
    # 4) Per-component bias and diag/cov from physics_states
    # ----------------------------------------------------------------------
    print("\n[3b] Per-component |bias| (mean across seeds, real MW)\n")
    header2 = f"  {'component':<16}" + "".join(f" | {arm:<24}" for arm in available_arms)
    print(header2)
    print("  " + "-" * (len(header2) - 2))

    bias_summary: dict[str, dict] = {}
    for comp in COMP_NAMES:
        row = f"  {comp:<16}"
        for arm in available_arms:
            vals = [
                abs(raw[(arm, s)]["comp"]["bias_per_comp"][comp])
                for s in SEEDS
                if (arm, s) in raw and "bias_per_comp" in raw[(arm, s)]["comp"]
            ]
            m, sd = msd(vals)
            bias_summary.setdefault(arm, {})[comp] = {"mean": m, "std": sd, "values": vals}
            row += f" | {m:.4e} ± {sd:.2e}"
        print(row)

    # ----------------------------------------------------------------------
    # 5) Falsification thresholds
    # ----------------------------------------------------------------------
    print("\n[4/4] Falsification verdict\n")

    verdict: dict[str, object] = {}

    if "baseline" in available_arms and "detach" in available_arms:
        bl_cov = agg_summary["baseline"]["cov_theory_residual"]["mean"]
        de_cov = agg_summary["detach"]["cov_theory_residual"]["mean"]
        ratio_cov = abs(de_cov) / max(abs(bl_cov), 1e-30)
        cov_pass = ratio_cov < 0.30
        verdict["cov_ratio_detach_over_baseline"] = ratio_cov
        verdict["cov_pass"] = cov_pass
        print(f"  |cov(theory, residual)| detach / baseline = {ratio_cov:.4f} "
              f"(threshold < 0.30) → {'PASS' if cov_pass else 'FAIL'}")

        # per-comp bias reduction count: 5 comp × 3 seeds = 15, count where
        # |bias|^detach[s] < |bias|^baseline[s]
        wins = 0
        total = 0
        for comp in COMP_NAMES:
            for seed in SEEDS:
                if (("baseline", seed) in raw
                        and ("detach", seed) in raw
                        and "bias_per_comp" in raw[("baseline", seed)]["comp"]
                        and "bias_per_comp" in raw[("detach", seed)]["comp"]):
                    bl_b = abs(raw[("baseline", seed)]["comp"]["bias_per_comp"][comp])
                    de_b = abs(raw[("detach", seed)]["comp"]["bias_per_comp"][comp])
                    total += 1
                    if de_b < bl_b:
                        wins += 1
        bias_pass = wins >= 9 and total >= 15
        verdict["bias_reduction_count"] = wins
        verdict["bias_total"] = total
        verdict["bias_pass"] = bias_pass
        print(f"  per-component |bias| reductions (detach vs baseline) = {wins}/{total} "
              f"(threshold ≥ 9/15) → {'PASS' if bias_pass else 'FAIL'}")

        m1_overall = cov_pass and bias_pass
        verdict["M1_overall"] = m1_overall
        print(f"\n  M1 (capacity-cancellation 解耦) verdict: "
              f"{'SUPPORTED' if m1_overall else 'NOT SUPPORTED (so far)'}")
    else:
        print("  [skip] need both baseline and detach data to compute verdict.")

    # detach_e3 adjudication (M1 vs M4)
    if "detach_e3" in available_arms and "detach" in available_arms:
        # Use metrics.json MAE for cleaner comparison (consistent with C09 evidence)
        de_mae = []
        de3_mae = []
        for seed in SEEDS:
            for arm, store in [("detach", de_mae), ("detach_e3", de3_mae)]:
                m_path = run_dir(arm, seed) / "metrics.json"
                if m_path.exists():
                    with open(m_path) as f:
                        store.append(json.load(f)["mae"])
        de_m, de_s = msd(de_mae)
        de3_m, de3_s = msd(de3_mae)
        gap = de3_m - de_m
        margin = max(de_s, 3.7e-5)  # C09 detach std fallback
        if gap < -margin:
            adjudication = "M4 (deeper encoder + detach > detach alone)"
        elif gap > margin:
            adjudication = "M1 (deeper encoder cancellation budget cut by detach)"
        else:
            adjudication = "INCONCLUSIVE (within ± 1 std of detach baseline)"
        verdict["detach_e3_mean"] = de3_m
        verdict["detach_e3_std"] = de3_s
        verdict["detach_mean"] = de_m
        verdict["adjudication_M1_vs_M4"] = adjudication
        print(f"\n  E1 adjudication M1 vs M4:")
        print(f"    detach    MAE = {de_m:.4e} ± {de_s:.2e}")
        print(f"    detach_e3 MAE = {de3_m:.4e} ± {de3_s:.2e}")
        print(f"    gap = {gap:+.2e}  (margin = ±{margin:.2e})")
        print(f"    => {adjudication}")
    elif "detach_e3" not in available_arms:
        print(f"\n  [pending] E1 (detach_e3) not yet present — adjudication deferred.")

    # ----------------------------------------------------------------------
    # 6) Persist
    # ----------------------------------------------------------------------
    out = ROOT / "scripts" / "m1_cov_cross_t1_results.json"
    serializable = {
        "arms_available": available_arms,
        "seeds": SEEDS,
        "aggregate_summary": agg_summary,
        "bias_summary": bias_summary,
        "verdict": verdict,
        "by_run_aggregate": {
            f"{arm}_s{seed}": raw[(arm, seed)]["agg"]
            for (arm, seed) in raw
        },
        "by_run_perComp": {
            f"{arm}_s{seed}": {
                k: v for k, v in raw[(arm, seed)]["comp"].items() if k != "cov_pairs"
            }
            for (arm, seed) in raw if raw[(arm, seed)]["comp"]
        },
    }
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2, default=float)
    print(f"\n  Results JSON: {out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
