# Version Comparison — Aggregate Metrics (V3 → V5)

- **Source**: `docs/research_directions.md` version comparison table
- **Claims**: C01, C02, C04
- **Notes**: All MAE values in kW. Theory MAE = mean absolute deviation of theory branch output from actual component values. V4 is the current best aggregate performer.

| Version | Key Changes | Test MAE (kW) | Theory MAE (kW) | Val MSE | vs V3 ΔMAE | vs V3 ΔTheory |
|---------|-------------|---------------|-----------------|---------|------------|---------------|
| V3 | Baseline (CosineAnnealing) | 1.932 | 4.874 | 0.387 | — | — |
| V4 | +calendar +component_loss(0.05) +load_proxy | 1.976 | 3.811 | 0.381 | +2.3% | **-21.8%** |
| V4.1 | V4 +sigmoid_gate +time_proj +norm_loss | 2.002 | 4.966 | 0.379 | +3.6% | +1.9% |
| V4.2 | V4 -gate -time_proj +norm_loss +schedule_fix | 2.048 | 2.981 | 0.378 | +6.0% | **-38.8%** |
| V5 | Component-consistent residual +TemporalDecoder +Curriculum | 2.120 | 3.074 | 0.404 | +9.7% | -36.9% |
