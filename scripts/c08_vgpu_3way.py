"""
C08 variance decomposition adapted for c23 vgpu_s2025 three-way comparison.
Tests: is `detach` inside or outside the cancellation regime under true 3-stage?

Usage:
    set PYTHONPATH=C:/Users/Xch/.codex/worktrees/7c57/Physformer
    & "E:\\Miniconda3\\envs\\PhysFormer\\python.exe" scripts/c08_vgpu_3way.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = str(ROOT)
DATA_PATH = "data_processed/multi_portfolio/portfolio_dataset_for_training.csv"

CONFIG = dict(
    root_path=DATA_ROOT,
    data_path=DATA_PATH,
    features="M",
    target=None,
    time_col="date",
    id_col="portfolio_id",
    region_col="region_id",
    split_col="split",
    split_strategy="portfolio_manifest",
    task_mode="net_injection",
    target_cols=["p_vpp_mw"],
    known_future_covariate_cols=["temperature", "irradiance", "wind_speed"],
    history_state_cols=["p_battery_mw", "e_battery_soc_mwh"],
    aux_target_cols=[
        "p_load_mw", "p_pv_mw", "p_wind_mw",
        "p_battery_mw", "e_battery_soc_mwh",
    ],
    seq_len=672,
    pred_len=96,
    label_len=0,
    batch_size=64,
    num_workers=0,
    pin_memory=False,
    model="PhysFormer",
)

RUNS = {
    "baseline": "physformer_c23_baseline_vgpu_s2025",
    "e3":       "physformer_c23_e3_vgpu_s2025",
    "detach":   "physformer_c23_detach_vgpu_s2025",
}
COMP_NAMES = ["load", "pv", "wind", "battery_power", "battery_soc"]
# net = load - pv - wind + battery_power; battery_soc not in net equation
NET_SIGN = np.array([+1, -1, -1, +1, 0], dtype=float)


class SimpleArgs:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def extract_y_aux():
    sys.path.insert(0, str(ROOT))
    from physformer.data import data_provider
    args = SimpleArgs(**CONFIG)
    dataset, loader = data_provider(args, "test")
    all_y = []
    for batch in loader:
        all_y.append(batch[5].cpu().numpy())
    y_raw = np.concatenate(all_y, axis=0)
    mean = dataset.aux_scaler.mean_.reshape(1, 1, -1)
    std  = dataset.aux_scaler.scale_.reshape(1, 1, -1)
    y_real = y_raw * std + mean
    print(f"y_aux: {y_real.shape}, range=[{y_real.min():.5f}, {y_real.max():.5f}] MW")
    return y_real


def analyze_run(name, run_dir, y_aux_real):
    pstates = dict(np.load(run_dir / "extras" / "physics_states.npz", allow_pickle=True))
    theory = pstates["component_theory_real"]
    with open(run_dir / "metrics.json") as f:
        m = json.load(f)
    err = (theory - y_aux_real).reshape(-1, 5)   # (N*T, 5)

    var = np.array([err[:, i].var() for i in range(5)])
    cov = np.cov(err.T)
    mae = np.mean(np.abs(err), axis=0)
    bias = np.mean(err, axis=0)

    # var(error_net) = sum_i s_i^2 var_i + 2 sum_{i<j} s_i s_j cov_ij
    # since |s_i|=1 (or 0 for soc), s_i^2 = 1 when in net, else 0
    s = NET_SIGN
    var_sum_diag = float(sum(var[i] for i in range(5) if s[i] != 0))
    cov_cross_term = 0.0
    cov_pairs = {}
    for i in range(5):
        for j in range(i + 1, 5):
            if s[i] == 0 or s[j] == 0:
                continue
            term = 2.0 * s[i] * s[j] * cov[i, j]
            cov_cross_term += term
            cov_pairs[f"{COMP_NAMES[i]}_{COMP_NAMES[j]}"] = {
                "sign_product": int(s[i] * s[j]),
                "cov": float(cov[i, j]),
                "contribution_to_var_net": float(term),
            }
    var_net_predicted = var_sum_diag + cov_cross_term

    # observed var of true aggregate theory error
    err_net = err @ s
    var_net_observed = float(err_net.var())

    # Cancellation ratio: 1 - |signed_sum| / sum_of_abs (per-sample mean)
    abs_sum = np.sum(np.abs(err[:, s != 0]), axis=1)
    cancel = 1.0 - np.abs(err_net) / (abs_sum + 1e-12)
    cancel_mean = float(cancel.mean())

    return dict(
        run=name,
        mae_per_comp=dict(zip(COMP_NAMES, mae.tolist())),
        bias_per_comp=dict(zip(COMP_NAMES, bias.tolist())),
        var_per_comp=dict(zip(COMP_NAMES, var.tolist())),
        var_sum_diag=var_sum_diag,
        cov_cross_term=cov_cross_term,
        var_net_predicted=var_net_predicted,
        var_net_observed=var_net_observed,
        cancellation_mean=cancel_mean,
        cov_pairs=cov_pairs,
        agg_mae=m["mae"],
        agg_mse=m["mse"],
        theory_mae=m["theory_mae"],
    )


def main():
    print("=" * 78)
    print("C08 Variance Decomposition: c23 vgpu_s2025 three-way comparison")
    print("=" * 78)

    print("\n[1/3] Extracting y_aux from test set...")
    y_aux = extract_y_aux()

    results = {}
    for tag, dirname in RUNS.items():
        rd = ROOT / "runs" / dirname
        print(f"\n[2/3] Analyzing {tag} ({rd.name})...")
        r = analyze_run(tag, rd, y_aux)
        results[tag] = r
        print(f"  Aggregate MAE (real MW)   : {r['agg_mae']:.6f}")
        print(f"  Aggregate MSE (real MW^2) : {r['agg_mse']:.3e}")
        print(f"  Theory MAE                : {r['theory_mae']:.6f}")
        print(f"  Cancellation ratio (mean) : {r['cancellation_mean']:.4f}")
        print(f"  Var(theory_err) per comp [MW^2]:")
        for n, v in r["var_per_comp"].items():
            print(f"      {n:>16}: {v:.3e}   bias={r['bias_per_comp'][n]:+.5f}   mae={r['mae_per_comp'][n]:.5f}")
        print(f"  Var decomposition of error_net:")
        print(f"      sum diag var               : {r['var_sum_diag']:.3e}")
        print(f"      cross cov contribution     : {r['cov_cross_term']:+.3e}")
        print(f"      predicted var(error_net)   : {r['var_net_predicted']:.3e}")
        print(f"      observed  var(error_net)   : {r['var_net_observed']:.3e}")
        print(f"  Top |cov| pairs (sign-weighted contribution):")
        sorted_pairs = sorted(r["cov_pairs"].items(),
                              key=lambda kv: abs(kv[1]["contribution_to_var_net"]),
                              reverse=True)
        for k, v in sorted_pairs[:5]:
            print(f"      {k:>28}: cov={v['cov']:+.3e}  sign={v['sign_product']:+d}  contrib={v['contribution_to_var_net']:+.3e}")

    print("\n[3/3] === THREE-WAY COMPARISON ===\n")
    print(f"  {'metric':<28}" + "".join(f"{k:>14}" for k in results.keys()))
    rows = [
        ("Aggregate MAE",            lambda r: r["agg_mae"]),
        ("Aggregate MSE",            lambda r: r["agg_mse"]),
        ("Theory MAE",               lambda r: r["theory_mae"]),
        ("Cancellation ratio",       lambda r: r["cancellation_mean"]),
        ("Sum diag var",             lambda r: r["var_sum_diag"]),
        ("Cross cov contrib",        lambda r: r["cov_cross_term"]),
        ("Var(error_net) observed",  lambda r: r["var_net_observed"]),
        ("Var(error_net) predicted", lambda r: r["var_net_predicted"]),
    ]
    for label, fn in rows:
        vals = [fn(results[k]) for k in results]
        print(f"  {label:<28}" + "".join(f"{v:>14.4e}" if abs(v) < 1 else f"{v:>14.6f}" for v in vals))

    print("\n  Cancellation budget by run (negative = cancellation; positive = amplification):")
    print(f"  {'pair':<28}" + "".join(f"{k:>14}" for k in results.keys()))
    pair_keys = list(results["baseline"]["cov_pairs"].keys())
    for pk in pair_keys:
        vals = [results[k]["cov_pairs"][pk]["contribution_to_var_net"] for k in results]
        print(f"  {pk:<28}" + "".join(f"{v:>+14.3e}" for v in vals))

    # Final diagnosis
    print("\n  --- Regime diagnosis ---")
    bl = results["baseline"]
    de = results["detach"]
    diag_drop = (de["var_sum_diag"] - bl["var_sum_diag"]) / bl["var_sum_diag"]
    cross_drop = (de["cov_cross_term"] - bl["cov_cross_term"]) / abs(bl["cov_cross_term"])
    agg_drop = (de["agg_mse"] - bl["agg_mse"]) / bl["agg_mse"]
    print(f"  detach vs baseline | var(diag) Δ% = {diag_drop:+.2%}")
    print(f"  detach vs baseline | cov-cross  Δ% = {cross_drop:+.2%}  (cov-cross is typically NEGATIVE; less-negative = LESS cancellation)")
    print(f"  detach vs baseline | agg MSE   Δ% = {agg_drop:+.2%}")

    if diag_drop < -0.10 and (cross_drop > 0 or abs(cross_drop) < 0.10):
        print("  => detach reduced component variance WITHOUT losing cancellation:")
        print("     evidence of REGIME ESCAPE (better components + similar cov structure).")
    elif diag_drop < 0 and cross_drop < 0:
        print("  => detach reduced both diag var AND cov-cross — better components AND stronger cancellation.")
    elif diag_drop > 0:
        print("  => detach increased diag var — improvement must come from cov restructure.")
    else:
        print("  => mixed signal; inspect per-pair contributions above.")

    out = ROOT / "scripts" / "c08_vgpu_3way_results.json"
    with open(out, "w") as f:
        json.dump(
            {k: {kk: vv for kk, vv in v.items() if kk != "cov_pairs"} for k, v in results.items()},
            f, indent=2, default=float,
        )
    print(f"\n  Summary JSON: {out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
