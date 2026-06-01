# V5 Training Results Analysis Target

## Context
PhysFormer V5 introduces "Component-consistent residual with curriculum training". We need to analyze whether V5 improves over previous versions (V3, V4, V4_1, V4_2).

## Metrics Comparison (test set, normalized scale)

| Metric | V3 | V4 | V4_1 | V4_2 | V5 |
|--------|------|------|------|------|------|
| MAE | 0.001932 | 0.001976 | 0.002002 | 0.002048 | 0.002120 |
| MSE (×10⁻⁶) | 7.459 | 7.350 | 7.684 | 8.043 | 8.224 |
| RMSE | 0.002731 | 0.002711 | 0.002772 | 0.002836 | 0.002868 |
| Theory MAE | 0.004874 | 0.003811 | 0.004966 | 0.002981 | 0.003074 |
| Residual std (MW) | 0.006545 | 0.004312 | 0.005076 | 0.002695 | 0.002686 |
| Residual mean (MW) | N/A | 0.002259 | -0.003336 | -0.001151 | -0.000848 |
| Net ramp violation | 0.003039 | 0.002887 | 0.003100 | 0.007386 | 0.002948 |
| SOC bound violation | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## V5 Component-Level Metrics (unique to V5 and V4_1)
| Component | V4 | V4_1 | V4_2 | V5 |
|-----------|------|------|------|------|
| Load MAE | 0.014707 | 0.002044 | 0.002093 | 0.002069 |
| PV MAE | 0.003998 | 0.003795 | 0.002449 | 0.001892 |
| Wind MAE | 0.000825 | 0.000459 | 0.000355 | 0.000313 |
| Battery Power MAE | 0.021345 | 0.001532 | 0.001686 | 0.001340 |
| Battery SOC MAE | 0.020111 | 0.004579 | 0.006190 | 0.004422 |

## V5 Training Dynamics
- 42 epochs, early stopped (patience=20, counter reached 20)
- Best Val MSE: 0.403846 (epoch ~10)
- Curriculum training phase: epochs 41-42 showed train loss drop (0.11 → 0.03) but Val MSE did not improve
- Training time: ~20 hours total
- GPU memory: 1.23 GB

## Key Questions for Analysis
1. Why does V5 have worse aggregate metrics (MAE/MSE/RMSE) than all previous versions?
2. Is V5's improvement in component-level metrics a fair trade-off?
3. What is the effect of the curriculum training phase?
4. Should V5 be adopted or should we revert to an earlier version?
