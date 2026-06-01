"""
D3 — M3 mechanism candidate: flat-minimum (sharpness) via parameter perturbation.

Tests whether detach's converged minimum is flatter than baseline/e3's, which
would explain the N82 observation that detach has the smallest cross-seed std
on ALL aggregate metrics.

For each run:
  1) Load PhysFormer model from checkpoint.pth (CPU).
  2) Build test loader, take a 5% subset (deterministic).
  3) Evaluate baseline net_mse_real on the subset.
  4) For N_DIRS random unit-norm gaussian directions ξ (defined over the
     encoder parameter subspace only — phys_layer / film / residual_head are
     NOT perturbed), and for each ε in EPSILONS:
       a) θ ← θ + ε·ξ      (encoder params only)
       b) measure loss
       c) θ ← θ              (restore)
       d) record sharpness = (loss - baseline_loss) / ε²
  5) Aggregate: sharpness_mean ± std per (run, ε), then per (arm, ε) across seeds.

Falsification (see docs/plans/2026-05-25-detach-mechanism-search.md §4 D3):
  PASS if:  detach mean sharpness < 0.7 × baseline mean sharpness    AND
            cross-seed std(sharpness)_detach < std(sharpness)_baseline AND
            sharpness monotonically increases with ε (sanity).

CPU FP32 only. Dry-run mode prints single-checkpoint single-direction timing
before committing to the full sweep.

Usage:
    set PYTHONPATH=C:/Users/Xch/.codex/worktrees/7c57/Physformer
    # Dry-run on baseline_s2025 only to estimate timing
    & "E:\\Miniconda3\\envs\\PhysFormer\\python.exe" scripts/m3_sharpness_perturbation.py --dry-run
    # Full sweep (9 or 12 ckpts; expect several hours)
    & "E:\\Miniconda3\\envs\\PhysFormer\\python.exe" scripts/m3_sharpness_perturbation.py
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from physformer.config import load_config, config_to_args, finalize_args  # noqa: E402
from physformer.train.physformer_exp import PhysFormerExperiment  # noqa: E402

ARMS = ["baseline", "e3", "detach", "detach_e3"]
SEEDS = [2025, 2026, 2027]
EPSILONS = [1e-3, 3e-3, 1e-2]
N_DIRECTIONS = 30
TEST_FRACTION = 0.05
PERTURB_SEED_BASE = 20260525


class FakeCli:
    """Mimic the argparse Namespace finalize_args expects."""

    def __init__(self, run_dir: Path, run_name: str):
        self.epochs = None
        self.lr = None
        self.batch_size = None
        self.gpu = None
        self.num_workers = 0
        self.patience = None
        self.seed = None
        self.run_name = run_name
        self.run_dir = str(run_dir)
        self.init_from_run = None
        self.resume = True
        self.debug_nan = False
        self.save_gate_details = False


def load_exp(run_name: str) -> PhysFormerExperiment:
    """Build PhysFormerExperiment + load checkpoint, CPU FP32."""
    run_dir = ROOT / "runs" / run_name
    cfg_path = run_dir / "config_merged.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing config_merged.yaml in {run_dir}")
    cfg = load_config(cfg_path)
    args = config_to_args(cfg)
    args, cfg = finalize_args(args, cfg, FakeCli(run_dir, run_name))
    args.use_gpu = False
    args.use_amp = False
    args.num_workers = 0
    args.pin_memory = False
    args.persistent_workers = False
    args.prefetch_factor = 2
    args.root_path = str(ROOT)  # local data path (config_merged.yaml has remote /root/autodl-tmp)
    args.data_path = "data_processed/multi_portfolio/portfolio_dataset_for_training.csv"
    exp = PhysFormerExperiment(args)
    ckpt = run_dir / "checkpoint.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint.pth in {run_dir}")
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    exp.model.eval()
    return exp


def take_subset_batches(exp: PhysFormerExperiment, fraction: float, max_batches: int | None = None):
    _, loader = exp._get_data("test")
    total = len(loader)
    target = max(1, int(total * fraction))
    if max_batches is not None:
        target = min(target, max_batches)
    batches = []
    for i, b in enumerate(loader):
        if i >= target:
            break
        batches.append(b)
    return batches, target, total


def evaluate_loss(exp: PhysFormerExperiment, batches) -> float:
    """Return mean net_mse_real (MW²) over the cached batches."""
    exp.model.eval()
    losses_real = []
    losses_norm = []
    with torch.no_grad():
        for b in batches:
            result = exp._process_one_batch(b, collect_debug=False, compute_loss=True)
            terms = result.get("terms", {})
            v_real = terms.get("net_mse_real")
            v_norm = terms.get("net_mse")
            if isinstance(v_real, torch.Tensor):
                losses_real.append(float(v_real.detach().cpu()))
            if isinstance(v_norm, torch.Tensor):
                losses_norm.append(float(v_norm.detach().cpu()))
    if losses_real:
        return float(np.mean(losses_real))
    if losses_norm:
        return float(np.mean(losses_norm))
    raise RuntimeError("no net_mse term in batch result terms")


def encoder_parameters(model):
    """Only model.encoder.* params, with requires_grad coerced off (we use copy_)."""
    return [p for p in model.encoder.parameters()]


def random_unit_direction(params, rng):
    """Gaussian direction over the concatenated parameter vector, L2 normalized."""
    direction = [torch.from_numpy(rng.standard_normal(tuple(p.shape)).astype(np.float32))
                 for p in params]
    flat = torch.cat([d.reshape(-1) for d in direction])
    norm = float(flat.norm())
    if norm < 1e-30:
        norm = 1e-30
    return [d / norm for d in direction]


def measure_sharpness(exp, batches, epsilons, n_directions, rng):
    """For each (direction, ε) compute Δloss/ε² with proper restore."""
    encoder_params = encoder_parameters(exp.model)
    originals = [p.data.clone() for p in encoder_params]

    baseline_loss = evaluate_loss(exp, batches)

    records = []
    for d_idx in range(n_directions):
        direction = random_unit_direction(encoder_params, rng)
        for eps in epsilons:
            try:
                for p, o, d in zip(encoder_params, originals, direction):
                    p.data.copy_(o + eps * d)
                loss = evaluate_loss(exp, batches)
            finally:
                for p, o in zip(encoder_params, originals):
                    p.data.copy_(o)
            delta = loss - baseline_loss
            sharpness = delta / (eps * eps)
            records.append({
                "direction": d_idx, "epsilon": eps,
                "loss": loss, "delta": delta, "sharpness": sharpness,
            })

    return baseline_loss, records


def aggregate_records(records: list[dict]) -> dict:
    """Per-ε mean ± std of sharpness across directions."""
    by_eps: dict[float, list[float]] = {}
    for r in records:
        by_eps.setdefault(r["epsilon"], []).append(r["sharpness"])
    out = {}
    for eps, vals in by_eps.items():
        out[f"eps_{eps}"] = {
            "n_directions": len(vals),
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
        }
    return out


def msd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def run_one(arm: str, seed: int, n_directions: int, epsilons, fraction: float,
            max_batches: int | None) -> dict:
    run_name = f"physformer_c23_{arm}_vgpu_s{seed}"
    print(f"\n--- run: {run_name} ---")
    t0 = time.time()
    exp = load_exp(run_name)
    print(f"    model + scaler ready in {time.time() - t0:.1f}s")

    batches, n_batches, n_total = take_subset_batches(exp, fraction, max_batches)
    print(f"    test subset: {n_batches}/{n_total} batches (fraction={fraction})")

    t1 = time.time()
    base = evaluate_loss(exp, batches)
    t_per_eval = time.time() - t1
    print(f"    baseline loss = {base:.4e} | single forward eval = {t_per_eval:.1f}s")

    n_evals = 1 + n_directions * len(epsilons)
    est = n_evals * t_per_eval
    print(f"    plan: {n_evals} evals × {t_per_eval:.1f}s ≈ {est:.0f}s = {est/60:.1f}min")

    rng = np.random.default_rng(PERTURB_SEED_BASE + seed)
    baseline_loss, records = measure_sharpness(exp, batches, epsilons, n_directions, rng)
    summary = aggregate_records(records)
    print(f"    sharpness summary by ε:")
    for k, v in summary.items():
        print(f"      {k}: mean={v['mean']:+.3e}  std={v['std']:.3e}  "
              f"min={v['min']:+.3e}  max={v['max']:+.3e}  N={v['n_directions']}")

    return {
        "run_name": run_name,
        "baseline_loss": baseline_loss,
        "n_directions": n_directions,
        "epsilons": list(epsilons),
        "fraction": fraction,
        "n_batches": n_batches,
        "n_total_batches": n_total,
        "summary": summary,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run baseline_s2025 with 5 directions × 1 ε only")
    parser.add_argument("--n-dirs", type=int, default=N_DIRECTIONS)
    parser.add_argument("--fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--arms", nargs="+", default=ARMS,
                        help="Subset of arms to evaluate (default: all available)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args_cli = parser.parse_args()

    print("=" * 96)
    print("D3 / M3 diagnostic: parameter-perturbation sharpness on encoder weights")
    print("=" * 96)

    if args_cli.dry_run:
        print("\n[DRY-RUN] baseline_s2025, 5 directions × 1 ε to estimate timing\n")
        results = [run_one("baseline", 2025, n_directions=5, epsilons=[1e-3],
                           fraction=args_cli.fraction, max_batches=None)]
        print("\nDone (dry-run).")
        return

    # Full sweep
    n_dirs = args_cli.n_dirs
    epsilons = EPSILONS
    fraction = args_cli.fraction

    all_results: list[dict] = []
    for arm in args_cli.arms:
        for seed in args_cli.seeds:
            rd = ROOT / "runs" / f"physformer_c23_{arm}_vgpu_s{seed}"
            if not (rd / "checkpoint.pth").exists():
                print(f"\n  [skip] missing ckpt: {rd.name}")
                continue
            try:
                res = run_one(arm, seed, n_dirs, epsilons, fraction, max_batches=None)
                res["arm"] = arm
                res["seed"] = seed
                all_results.append(res)
            except Exception as exc:
                print(f"  [error] {arm}_s{seed}: {exc}")

    # ----------------------------------------------------------------
    # Cross-seed aggregation
    # ----------------------------------------------------------------
    print("\n\n=== Cross-seed aggregation (mean ± std across seeds, per arm and per ε) ===\n")
    by_arm: dict[str, dict] = {}
    arms_seen = {r["arm"] for r in all_results}
    for arm in [a for a in args_cli.arms if a in arms_seen]:
        by_arm[arm] = {}
        for eps in epsilons:
            key = f"eps_{eps}"
            vals = [r["summary"][key]["mean"] for r in all_results
                    if r["arm"] == arm and key in r["summary"]]
            stds = [r["summary"][key]["std"] for r in all_results
                    if r["arm"] == arm and key in r["summary"]]
            m, s = msd(vals)
            ms, ss = msd(stds)
            by_arm[arm][key] = {
                "sharpness_mean_of_means": m,
                "sharpness_std_of_means":  s,
                "intra_seed_std_mean":    ms,
                "intra_seed_std_std":     ss,
                "per_seed_means": vals,
                "per_seed_intra_seed_stds": stds,
            }
        # Print
        print(f"  arm: {arm}")
        for eps in epsilons:
            v = by_arm[arm][f"eps_{eps}"]
            print(f"    ε={eps:.0e}: mean(sharpness)={v['sharpness_mean_of_means']:+.3e} "
                  f"± {v['sharpness_std_of_means']:.3e}    "
                  f"intra-seed std={v['intra_seed_std_mean']:.3e}")

    # ----------------------------------------------------------------
    # Falsification verdict (detach vs baseline)
    # ----------------------------------------------------------------
    verdict: dict[str, object] = {}
    if "baseline" in by_arm and "detach" in by_arm:
        eps_passes = {}
        for eps in epsilons:
            key = f"eps_{eps}"
            bl_m = by_arm["baseline"][key]["sharpness_mean_of_means"]
            de_m = by_arm["detach"][key]["sharpness_mean_of_means"]
            bl_s = by_arm["baseline"][key]["sharpness_std_of_means"]
            de_s = by_arm["detach"][key]["sharpness_std_of_means"]
            ratio = de_m / max(abs(bl_m), 1e-30) if bl_m > 0 else float("nan")
            ratio_pass = (bl_m > 0) and (de_m < 0.7 * bl_m)
            std_pass = de_s < bl_s
            eps_passes[key] = {
                "ratio": ratio, "ratio_pass": ratio_pass,
                "std_pass": std_pass,
                "bl_m": bl_m, "de_m": de_m, "bl_s": bl_s, "de_s": de_s,
            }
            print(f"\n  ε={eps:.0e} verdict:")
            print(f"    baseline mean = {bl_m:+.3e} ± {bl_s:.3e}")
            print(f"    detach   mean = {de_m:+.3e} ± {de_s:.3e}")
            print(f"    ratio detach/baseline = {ratio:.4f}    (threshold < 0.70)  "
                  f"→ {'PASS' if ratio_pass else 'FAIL'}")
            print(f"    cross-seed std comparison: detach<{bl_s:.2e}?  "
                  f"→ {'PASS' if std_pass else 'FAIL'}")
        # Monotonicity
        means_in_eps_order = [by_arm["detach"][f"eps_{e}"]["sharpness_mean_of_means"]
                              for e in sorted(epsilons)]
        monotonic = all(b >= a - 1e-12 for a, b in zip(means_in_eps_order, means_in_eps_order[1:]))
        verdict["epsilons"] = eps_passes
        verdict["detach_monotonic_in_eps"] = monotonic
        m3_overall = all(v["ratio_pass"] and v["std_pass"] for v in eps_passes.values()) and monotonic
        verdict["M3_overall"] = m3_overall
        print(f"\n  monotonic sharpness vs ε for detach? → {'YES' if monotonic else 'NO'}")
        print(f"\n  M3 (flat-minimum / sharpness) verdict: "
              f"{'SUPPORTED' if m3_overall else 'NOT SUPPORTED (so far)'}")

    # ----------------------------------------------------------------
    # Persist
    # ----------------------------------------------------------------
    out = ROOT / "scripts" / "m3_sharpness_results.json"
    # Drop per-direction records from JSON to keep size manageable; keep summary only
    minimal_results = []
    for r in all_results:
        rr = {k: v for k, v in r.items() if k != "records"}
        minimal_results.append(rr)
    serializable = {
        "config": {
            "n_directions": n_dirs, "epsilons": list(epsilons),
            "test_fraction": fraction, "perturb_seed_base": PERTURB_SEED_BASE,
        },
        "per_run": minimal_results,
        "per_arm_per_eps": by_arm,
        "verdict": verdict,
    }
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2, default=float)
    print(f"\n  Results JSON: {out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
