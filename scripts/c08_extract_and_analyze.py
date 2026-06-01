"""
C08: Component error correlation analysis.
Extracts y_aux (per-component true values) from test dataset, then computes
per-component error correlation matrices for V6 S2 gru64 and gru96.

Hypothesis: If component errors are anti-correlated in a model, they cancel
in the aggregate sum (net = load - pv - wind + batt). When component coupling
tightens (e.g., gru64), errors become less anti-correlated → less cancellation
→ aggregate MSE degrades despite better per-component MAE.

Usage:
    python scripts/c08_extract_and_analyze.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Config — mirrors V6 S2
# ---------------------------------------------------------------------------
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
        "p_load_mw",
        "p_pv_mw",
        "p_wind_mw",
        "p_battery_mw",
        "e_battery_soc_mwh",
    ],
    seq_len=672,
    pred_len=96,
    label_len=0,
    batch_size=64,
    num_workers=0,
    pin_memory=False,
    model="PhysFormer",
)


class SimpleArgs:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def extract_y_aux() -> NDArray:
    """Load test dataset via data_provider, extract all y_aux in real MW."""
    sys.path.insert(0, str(ROOT))
    from physformer.data import data_provider

    args = SimpleArgs(**CONFIG)
    dataset, loader = data_provider(args, "test")

    # Collect per-batch y_aux
    all_y_aux = []
    for batch in loader:
        # batch layout: (x_net_hist, x_weather_hist, x_battery_hist,
        #                x_weather_future, y_target, y_aux, x_mark_enc, y_mark, portfolio_idx)
        y_aux = batch[5]  # (B, 96, 5)
        all_y_aux.append(y_aux.cpu().numpy())

    y_aux_raw = np.concatenate(all_y_aux, axis=0)  # (N, 96, 5)

    # Denormalize
    aux_mean = dataset.aux_scaler.mean_.reshape(1, 1, -1)
    aux_std = dataset.aux_scaler.scale_.reshape(1, 1, -1)
    y_aux_real = y_aux_raw * aux_std + aux_mean

    print(f"Extracted y_aux: {y_aux_real.shape}, range=[{y_aux_real.min():.4f}, {y_aux_real.max():.4f}]")
    return y_aux_real


def load_model_outputs(run_name: str) -> dict:
    """Load saved test outputs for a given run."""
    run_dir = ROOT / "results" / "v6_s2_ablation" / run_name

    # Component theory (N, 96, 5) — real MW
    pstates = dict(np.load(run_dir / "extras" / "physics_states.npz", allow_pickle=True))
    component_theory = pstates["component_theory_real"]  # (N, 96, 5)

    # Residuals (N, 96, 5) — these are in NORMALIZED units from the model output
    residuals_norm = np.load(run_dir / "extras" / "residual_net.npy")

    # Aggregate pred/true (N, 96, 1) — real MW
    pred_agg = np.load(run_dir / "pred.npy")
    true_agg = np.load(run_dir / "true.npy")

    # Load metrics for scaling info
    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)

    # Load test_loss_terms for component MAE (normalized)
    with open(run_dir / "extras" / "test_loss_terms.json") as f:
        loss_terms = json.load(f)

    return {
        "component_theory": component_theory,
        "residuals_norm": residuals_norm,
        "pred_agg": pred_agg.squeeze(-1),
        "true_agg": true_agg.squeeze(-1),
        "metrics": metrics,
        "loss_terms": loss_terms,
    }


def compute_error_correlations(
    component_theory: NDArray,  # (N, 96, 5) real MW
    residuals_norm: NDArray,    # (N, 96, 5) normalized
    y_aux_real: NDArray,        # (N, 96, 5) real MW
    comp_names: list[str],
) -> dict:
    """
    Compute per-component error correlations.

    Two types of component "prediction":
    1. Theory-only:  pred_theory = component_theory
    2. Theory+residual: pred_full = component_theory + residual_denormed
       (We approximate residual in real MW via residual_std scaling)

    Returns correlation matrices and summary stats.
    """
    # Theory-only errors
    error_theory = component_theory - y_aux_real  # (N, 96, 5)

    # For full prediction, we need residuals in real MW.
    # residual_net.npy is in normalized units; we approximate real MW via
    # the ratio residual_std_real / residual_std_norm per component.
    # Simpler: compute full component prediction via component_theory + residual
    # where residual in real MW ≈ residual_norm * component_scale
    # But we don't have component-wise residual scales.
    #
    # Alternative: use the fact that theory_mae (from metrics) uses component_theory
    # directly. The "component MAE" in metrics IS theory-only MAE (see test code line 449-456).
    # For the C08 hypothesis, theory-only errors are sufficient — the key mechanism
    # is about whether component theory errors are anti-correlated.
    #
    # We'll compute both approaches where possible.

    N, T, C = error_theory.shape
    # Flatten: (N*T, C)
    err_flat = error_theory.reshape(-1, C)

    # --- Correlation matrix ---
    corr = np.corrcoef(err_flat.T)  # (5, 5)

    # --- Per-component error stats ---
    mae_per_comp = np.mean(np.abs(err_flat), axis=0)
    rmse_per_comp = np.sqrt(np.mean(err_flat ** 2, axis=0))
    bias_per_comp = np.mean(err_flat, axis=0)

    # --- Aggregate error decomposition ---
    # net = load - pv - wind + batt
    # error_net = error_load - error_pv - error_wind + error_batt
    err_load, err_pv, err_wind, err_batt, err_soc = err_flat.T
    error_agg_reconstructed = err_load - err_pv - err_wind + err_batt

    # --- Cancellation analysis ---
    # If all errors were independent and same-sign:
    #   total = |err_load| + |err_pv| + |err_wind| + |err_batt|
    # Actual aggregate error: |err_load - err_pv - err_wind + err_batt|
    # Cancellation ratio = 1 - (actual / total) — higher = more cancellation
    abs_sum = np.abs(err_load) + np.abs(err_pv) + np.abs(err_wind) + np.abs(err_batt)
    cancellation_ratio = 1.0 - np.abs(error_agg_reconstructed) / (abs_sum + 1e-10)
    mean_cancellation = float(np.mean(cancellation_ratio))

    # --- Pairwise error products (to detect anti-correlation in sums) ---
    # error_net = eL - ePV - eW + eB
    # var(error_net) = var(eL) + var(ePV) + var(eW) + var(eB)
    #   - 2*cov(eL, ePV) - 2*cov(eL, eW) + 2*cov(eL, eB)
    #   + 2*cov(ePV, eW) - 2*cov(ePV, eB) - 2*cov(eW, eB)
    cov = np.cov(err_flat.T)

    # Decompose aggregate error variance
    var_terms = {
        "var_load": cov[0, 0],
        "var_pv": cov[1, 1],
        "var_wind": cov[2, 2],
        "var_batt": cov[3, 3],
        "cov_load_pv": cov[0, 1],   # subtracts (same sign in net equation)
        "cov_load_wind": cov[0, 2],  # subtracts
        "cov_load_batt": cov[0, 3],  # adds (opposite sign)
        "cov_pv_wind": cov[1, 2],   # adds (both subtracted)
        "cov_pv_batt": cov[1, 3],   # subtracts (opposite sign)
        "cov_wind_batt": cov[2, 3],  # subtracts (opposite sign)
    }

    return {
        "correlation_matrix": corr,
        "correlation_labels": comp_names,
        "mae_per_component": dict(zip(comp_names, mae_per_comp.tolist())),
        "rmse_per_component": dict(zip(comp_names, rmse_per_comp.tolist())),
        "bias_per_component": dict(zip(comp_names, bias_per_comp.tolist())),
        "mean_cancellation_ratio": mean_cancellation,
        "variance_decomposition": var_terms,
        "covariance_matrix": cov,
        "reconstructed_agg_rmse": float(np.sqrt(np.mean(error_agg_reconstructed ** 2))),
    }


def compute_pairwise_cancellation_matrix(
    component_theory: NDArray,
    y_aux_real: NDArray,
) -> NDArray:
    """
    For each pair of components, compute how much their errors cancel
    in the aggregate sum, accounting for sign conventions.

    Returns a 5x5 matrix M where M[i,j] = mean(|e_i + sign_ij * e_j| / (|e_i| + |e_j|))
    Values < 1 indicate cancellation.
    """
    sign_matrix = np.array([
        # L   PV  W   B   SOC
        [ 1, -1, -1,  1,  0],  # Load: net += L
        [-1,  1,  1, -1,  0],  # PV:   net -= PV
        [-1,  1,  1, -1,  0],  # Wind: net -= W
        [ 1, -1, -1,  1,  0],  # Batt: net += B
        [ 0,  0,  0,  0,  0],  # SOC: not in net
    ])

    error = component_theory - y_aux_real  # (N, 96, 5)
    N, T, C = error.shape
    err = error.reshape(-1, C)

    cancel = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            s = sign_matrix[i, j]
            if s == 0:
                cancel[i, j] = np.nan
                continue
            combined = err[:, i] + s * err[:, j]
            denom = np.abs(err[:, i]) + np.abs(err[:, j]) + 1e-10
            cancel[i, j] = float(np.mean(np.abs(combined) / denom))

    return cancel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("C08: Component Error Correlation Analysis")
    print("=" * 70)

    # Step 1: Extract y_aux
    print("\n[1/4] Extracting y_aux from test dataset...")
    y_aux_real = extract_y_aux()

    # Step 2: Load model outputs for gru64 and gru96
    comp_names = ["load", "pv", "wind", "battery_power", "battery_soc"]

    results = {}
    for run in ["physformer_v6_s2_gru64", "physformer_v6_s2_gru96"]:
        print(f"\n[2/4] Loading {run}...")
        data = load_model_outputs(run)
        print(f"  pred_agg: {data['pred_agg'].shape}, true_agg: {data['true_agg'].shape}")
        print(f"  component_theory: {data['component_theory'].shape}")
        print(f"  residuals_norm: {data['residuals_norm'].shape}")

        # Verify shapes match
        assert data["component_theory"].shape[:2] == y_aux_real.shape[:2], \
            f"Theory shape {data['component_theory'].shape} vs aux {y_aux_real.shape}"

        print(f"\n[3/4] Computing error correlations for {run}...")
        analysis = compute_error_correlations(
            data["component_theory"],
            data["residuals_norm"],
            y_aux_real,
            comp_names,
        )

        print(f"\n  --- Correlation Matrix ({run}) ---")
        print(f"  {'':>8} " + " ".join(f"{n:>8}" for n in comp_names))
        for i, name in enumerate(comp_names):
            row = " ".join(f"{analysis['correlation_matrix'][i, j]:8.4f}" for j in range(5))
            print(f"  {name:>8} {row}")

        print(f"\n  --- Per-Component Theory MAE (real MW) ---")
        for name, mae in analysis["mae_per_component"].items():
            print(f"  {name:>16}: {mae:.6f}")

        print(f"\n  --- Variance Decomposition ---")
        for k, v in analysis["variance_decomposition"].items():
            print(f"  {k:>20}: {v:.8f}")

        print(f"\n  Mean cancellation ratio: {analysis['mean_cancellation_ratio']:.4f}")
        print(f"  Reconstructed agg RMSE: {analysis['reconstructed_agg_rmse']:.6f}")

        # Pairwise cancellation
        cancel_mat = compute_pairwise_cancellation_matrix(
            data["component_theory"], y_aux_real,
        )
        print(f"\n  --- Pairwise Cancellation Matrix (<1 = cancellation) ---")
        print(f"  {'':>8} " + " ".join(f"{n:>8}" for n in comp_names))
        for i, name in enumerate(comp_names):
            row = " ".join(
                f"{cancel_mat[i, j]:8.4f}" if not np.isnan(cancel_mat[i, j]) else f"{'--':>8}"
                for j in range(5)
            )
            print(f"  {name:>8} {row}")

        results[run] = analysis

    # Step 4: Compare gru64 vs gru96
    print(f"\n[4/4] === gru64 vs gru96: Correlation Difference ===")
    r64 = results["physformer_v6_s2_gru64"]
    r96 = results["physformer_v6_s2_gru96"]

    diff_corr = r64["correlation_matrix"] - r96["correlation_matrix"]
    print(f"  {'':>8} " + " ".join(f"{n:>8}" for n in comp_names))
    for i, name in enumerate(comp_names):
        row = " ".join(f"{diff_corr[i, j]:+8.4f}" for j in range(5))
        print(f"  {name:>8} {row}")

    # Key diagnostics for C08:
    # Are component errors more correlated (positive or negative) in gru64 vs gru96?
    print("\n  --- Key C08 Diagnostics ---")
    for i in range(5):
        for j in range(i + 1, 5):
            r64_val = r64["correlation_matrix"][i, j]
            r96_val = r96["correlation_matrix"][i, j]
            sign_i = {0: "+", 1: "-", 2: "-", 3: "+", 4: "0"}[i]
            sign_j = {0: "+", 1: "-", 2: "-", 3: "+", 4: "0"}[j]
            net_effect = "cancel" if sign_i != sign_j else "add"
            delta = r64_val - r96_val
            print(
                f"  r({comp_names[i]}, {comp_names[j]}): "
                f"gru64={r64_val:+.4f}, gru96={r96_val:+.4f}, "
                f"Δ={delta:+.4f} | net effect: {net_effect} "
                f"({'more' if abs(r64_val) > abs(r96_val) else 'less'} correlated in gru64)"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
