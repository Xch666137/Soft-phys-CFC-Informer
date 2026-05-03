# V4 Phase 1 Results — vs V3 Baseline Comparison

## V4 Changes (Phase 1)
1. Calendar embeddings: 8 sin/cos dims → 10 dims (+is_weekend, +is_holiday)
2. Historical load proxy: AR correction via recent net injection in `_load_branch`
3. Component loss supervision: `component_loss_weight=0.05` on load/pv/wind/battery MAE
4. Per-component evaluation in test(): new metrics for load, pv, wind, battery_power, battery_soc

## V3 Baseline
- Test MSE: 7.46e-6 MW², MAE: 1.932 kW, RMSE: 2.731 kW
- Theory MAE: 4.874 kW, Residual std: 6.545 kW
- Val MSE best: 0.387 (epoch 37), epoch 1 Val MSE: 2.95
- 100 epochs planned, early stopped at epoch 51 (counter=14/25)

## V4 Results
- Test MSE: 7.35e-6 MW², MAE: 1.976 kW, RMSE: 2.711 kW
- Theory MAE: 3.811 kW (-21.8%), Residual std: 4.312 kW (-34.1%)
- Residual mean: 2.259 kW (NEW metric)
- Val MSE best: 0.381 (epoch 11, 3.4x faster convergence vs V3 epoch 37)
- 100 epochs planned, early stopped at epoch 36 (counter=25/25)

### Per-Component Theory MAE (NEW in V4)
- component_load_mae: 14.71 kW
- component_pv_mae: 4.00 kW
- component_wind_mae: 0.83 kW
- component_battery_power_mae: 21.35 kW
- component_battery_soc_mae: 0.020 MWh
- SOC bound violation: 0.0

## V4 Training Dynamics
- Epochs 3-11: Rapid descent (Val MSE 1.03→0.381)
- Epoch 11: Best Val MSE 0.381 (Val Loss=0.383, Val SOC=0.000)
- Epochs 12-19: Plateau, first cosine cycle ending
- Epoch 20: WarmRestart #1 (LR reset to 1e-4), Val MSE 0.397
- Epochs 21-24: Recovery attempt, Val MSE dropped to 0.385 (not beating 0.381)
- Epochs 25-34: Second plateau, LR approaching zero
- Epoch 35: WarmRestart #2, Val MSE 0.398
- Epoch 36: Val MSE 0.398, EarlyStop=25/25 → stopped
- Test completed normally, duration ~32 min

## Key Questions
1. Why does early stopping trigger at epoch 36 without beating epoch 11? Is the LR schedule suboptimal?
2. Component breakdown: load (14.71 kW) is 18x worse than wind (0.83 kW). Is the load branch still underpowered?
3. Theory MAE improved 21.8% but MAE regressed 2.3%. Tradeoff analysis needed.
4. Val SOC = 0.000 all epochs. Is component_loss_weight too low or is SOC structural prior sufficient?
5. Training stopped at 36/100 epochs. Was there untapped potential from restart #3?
