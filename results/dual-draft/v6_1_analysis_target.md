# V6.1 Ablation Analysis Target

## Background

PhysFormer is a physics-guided Transformer for VPP aggregated net power forecasting.
It decomposes the forecast into theory-driven (physics equations) and residual (data-driven)
components across four asset types: Load, PV, Wind, Battery.

The V6 architecture replaced the HDD/CDD-based Load physics branch with:
- **Temp MLP**: 3-layer MLP (1→32→32→1) for temperature→load response
- **GRU temporal module**: Processes historical net_load + temperature + calendar features
  to produce temporal corrections

V6.1 tested two independent mechanisms to reinforce the physics-data boundary:

- **N29**: Remove temperature from GRU input → GRU only sees [net_load, calendar],
  making Temp MLP the sole temperature→load pathway
- **N30**: Activate selective gradient detach in Phase 2 → detach load/PV/wind theory
  outputs from the residual head's gradient flow, preventing the residual head from
  influencing physics branch learning (battery gradient kept intact)

## Experiment Design

2×2 factorial ablation. All share: restart_t0=20, load_gru_hidden=64, phase_1_epochs=8,
phase_2_epochs=50, patience=20, early_stop_start_epoch=20, seed=2024.

| Config | N29 (remove temp) | N30 (selective detach) |
|--------|:---:|:---:|
| baseline (V6 S2 gru64) | ✗ | ✗ |
| no_temp | ✓ | ✗ |
| detach | ✗ | ✓ |
| full | ✓ | ✓ |

## Raw Results (metrics.json)

All values in MW unless noted.

| Metric | baseline | no_temp | detach | full |
|--------|:---:|:---:|:---:|:---:|
| Test MAE | 0.002017 | 0.001999 | 0.002031 | 0.002032 |
| Test MSE (×10⁻⁶) | 7.581 | 7.938 | 8.161 | 8.214 |
| RMSE | 0.002753 | 0.002817 | 0.002857 | 0.002866 |
| Theory MAE | 0.002602 | 0.002431 | 0.002342 | 0.002509 |
| Ramp Violation | 0.0037% | 0.0043% | 0.0090% | 0.0046% |
| Residual Mean | −0.000949 | −0.000849 | −0.000065 | −0.000849 |
| Residual Std | 0.002276 | 0.002165 | 0.002199 | 0.002190 |
| Component Load MAE | 0.001936 | 0.001886 | 0.001898 | 0.001925 |
| Component PV MAE | 0.002551 | 0.001992 | 0.001738 | 0.001998 |
| Component Wind MAE | 0.000319 | 0.000314 | 0.000313 | 0.000314 |
| Component Batt P MAE | 0.001703 | 0.001496 | 0.001391 | 0.001526 |
| Component Batt SOC MAE | 0.008910 | 0.005957 | 0.006299 | 0.006152 |

## Training Dynamics (Val MSE trajectory)

| Epoch | baseline* | no_temp | detach | full |
|-------|-----------|---------|--------|------|
| 1 | ~10.7 | 10.74 | 9.70 | 11.16 |
| 5 | — | 0.439 | 0.439 | 0.425 |
| 8 | — | — | 0.451 | 0.418 |
| 10 | — | 0.418 | 0.424 | 0.413 |
| 12 | — | 0.410 | 0.432 | 0.423 |
| 15 | — | 0.417 | 0.421 | 0.427 |
| 20 | — | 0.442 | 0.419 | 0.436 |
| 25 | — | 0.440 | 0.422 | 0.436 |
| 30 | — | 0.440 | 0.419 | 0.435 |
| 35 | — | — | 0.419 | 0.436 |
| Best (ep≥20) | — | 0.432 (ep28) | 0.412 (ep26) | 0.432 (ep29) |

*Baseline training trajectory not monitored in real-time; values from its metrics.json.

## Key Architecture Details

1. **Load branch**: Temp MLP (1→32→32→1) for temperature→load. GRU (d_gru=64, 2 layers)
   for temporal correction. In baseline/detach: GRU input = [net_load, temp, calendar].
   In no_temp/full: GRU input = [net_load, calendar] (temp removed).

2. **Selective detach (N30)**: In model.py forward(), when detach_mode="selective":
   physics_for_head[..., :3] = physics_features[..., :3].detach() — cuts gradient from
   residual head to load/pv/wind theory branches. Battery gradient preserved.

3. **Phase structure**: Phase 1 (epochs 1-8): component_loss_weight=0.03, detach_mode="none".
   Phase 2 (epochs 9-50): component_loss_weight decays, detach_mode set per config.

## Research Questions

1. Does removing temperature from GRU (N29) improve the physics-data boundary,
   as hypothesized from the capacity-ceiling principle (N28)?
2. Does selective gradient detach (N30) reinforce component-consistent residual
   learning, or does it over-constrain gradient flow?
3. Are N29 and N30 complementary or antagonistic when combined?
4. What is the overall recommendation for the thesis: which variant should be the
   reported PhysFormer configuration, and why?
