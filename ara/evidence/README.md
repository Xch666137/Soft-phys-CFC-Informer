# Evidence Index

Each evidence file maps to the claims it supports. All numerical values are exact from experiment logs.

| Evidence File | Source | Claims Supported | Description |
|---------------|--------|------------------|-------------|
| [tables/version_comparison.md](tables/version_comparison.md) | research_directions.md | C01, C02, C04 | V3–V5 aggregate metrics (Test MAE, Theory MAE, Val MSE) |
| [tables/v5_component_metrics.md](tables/v5_component_metrics.md) | research_directions.md | C03, C06 | V5 per-component MAE (Load, PV, Wind, Battery Power, Battery SOC) |
| [tables/v4_v5_component_comparison.md](tables/v4_v5_component_comparison.md) | research_directions.md | C03 | V4 vs V5 component-level comparison |

## Phase A/B Evidence Map

Later iGT evidence is indexed through the exploration tree and run directories rather
than copied into static evidence tables. This table is the current evidence bridge for
C07-C13.

| Experiment | Claims Supported | Primary Evidence | Key Result / Status |
|------------|------------------|------------------|---------------------|
| E07 | C07 | `trace/exploration_tree.yaml` N33-N36, N55 | Gradient isolation has a non-monotonic optimum; detach is stable, full isolation degrades. |
| E08 | C08 | `docs/analysis/c08_variance_decomposition.md`, N63, N74, N83 | Component covariance cancellation can preserve aggregate accuracy despite worse component errors. |
| E09 | C09 | `runs/physformer_c23_*_vgpu_s{2025,2026,2027}/metrics.json`, N82 | Selective detach wins aggregate metrics and cross-seed variance under c23. |
| E10 | C10 | `trace/exploration_tree.yaml` N88-N91 | Detach x e3 behaves like baseline, supporting the encoder-depth cancellation-channel mechanism. |
| E11 | C11 | `results/physformer_igt_a1_s20{25,26,27}/metrics.json`, N100 | A1 8-token iGT reaches MAE 0.001811 +/- 0.000006, beating c23 with 20x lower variance. |
| E12 | C12 | `trace/exploration_tree.yaml` N102-N105 | A2-A5 fixed-prior additions monotonically worsen Test MAE despite better Val MSE. |
| E13 | C13 | `tmp/remote_metrics/*`, `runs/physformer_igt_b1_r1_reg_finetune_s{2025,2026,2027}/metrics.json`, N132 | Repaired R1-reg gives decomposable 4-component forecasts at about +1% aggregate MAE vs A1. |
| E14 | C13 | pending | Dispatch proxy validation not executed; operational dispatch benefit remains unproven. |
