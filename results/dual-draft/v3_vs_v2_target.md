# V3 Training Results — V2 vs V3 Comparison

We compare the PhysFormer V3 training results with the V2 baseline and identify improvements, regressions, and remaining optimization opportunities.

## V2 → V3 Changes (7 items)
1. `film_scale`: 0.2 → 0.5 (stronger physical feature modulation)
2. `theory_net`: 1-dim → 32-dim projection (more capacity for physics residual)
3. Wind gates: boolean hard → soft sigmoid (gradient flow)
4. PV temperature coefficient: asymmetric → symmetric
5. Split strategy: old → `portfolio_manifest` (cross-split portfolio consistency)
6. Loss: removed `soc_transition_loss` + `anti_overlap_loss` (simplification)
7. Scheduler: CosineAnnealing → CosineAnnealingWarmRestarts ($T_0=15$, $T_{\mathrm{mult}}=1$, 3 LR restarts → 4 cycles)

## V2 Results (Baseline)
- Test mean squared error (MSE): $9.0 \times 10^{-6}~\mathrm{MW}^2$
- Mean absolute error (MAE): $2.21~\mathrm{kW}$
- Root mean squared error (RMSE): $2.96~\mathrm{kW}$
- Theory MAE: $17.01~\mathrm{kW}$
- Best validation (Val) MSE: $0.420$
- Epoch 1 Val MSE: $331{,}843$

## V3 Results
- Test MSE: $7.46 \times 10^{-6}~\mathrm{MW}^2$
- MAE: $1.932~\mathrm{kW}$
- RMSE: $2.731~\mathrm{kW}$
- Theory MAE: $4.874~\mathrm{kW}$
- Residual standard deviation: $0.006545~\mathrm{MW}$ ($6.545~\mathrm{kW}$)
- State of charge (SOC) violation: $0.0$
- Best Val MSE: $0.3867$ (epoch 37)

## V3 Training Dynamics (from train.log)

### Cycle #1 (epochs 1–20, first cosine annealing period)
- Epoch 1: Train $= 720{,}948$, Val $= 2.9517$, learning rate (LR) $= 3.6 \times 10^{-5}$
- Epoch 5: Train $= 0.427$, Val $= 0.5387$, LR $= 1 \times 10^{-4}$ (linear warmup ends)
- Epoch 10: Train $= 0.123$, Val $= 0.4369$, LR $= 7.5 \times 10^{-5}$
- Epoch 13: Val $= 0.4052$ (**best of Cycle #1**), LR $= 4.5 \times 10^{-5}$
- Epoch 20: Val $= 0.4075$, LR resets to $1 \times 10^{-4}$ (first LR restart)

### Cycle #2 (epochs 21–35, after first LR restart)
- Epoch 21: Val $= 0.3969$ (immediate improvement; +3.3 % vs Cycle #1 best)
- Epoch 25: Val $= 0.3937$
- Epoch 30: Val $= 0.3918$ (**best of Cycle #2**)
- Epoch 35: LR resets to $1 \times 10^{-4}$ (second LR restart)

### Cycle #3 (epochs 36–50, after second LR restart)
- Epoch 37: Val $= 0.3867$ (**GLOBAL BEST**; +1.3 % vs Cycle #2 best), LR $= 9.6 \times 10^{-5}$
- Epochs 38–50: Early stopping counter $1 \to 13$ (no new best), Val oscillates in $0.390$–$0.396$
- Epoch 50: LR resets to $1 \times 10^{-4}$ (third LR restart)

### Cycle #4 (epoch 51+, after third LR restart)
- Epoch 51: Val $= 0.3967$, early stopping counter $= 14/25$
- **Training stopped at epoch 51** (manual truncation; counter 14 < patience 25, so early stopping did not trigger)

### Key observations
- Validation loss $=$ validation MSE for all epochs (`soc_weight` $= 0.1$ contributes $< 10^{-6}$)
- Epoch 1 Train/Val gap: $244{,}000\times$ ($720{,}948$ vs $2.95$)
- Each LR restart produced a new best: Cycle #1 $+3.3~\%$, Cycle #2 $+1.3~\%$, Cycle #3 produced global best but Cycle #4 had no chance to mature
- Training truncated at 51/100 epochs

## V2 → V3 Improvements
| Metric | V2 | V3 | $\Delta$ |
|--------|----|----|-----|
| Test MSE | $9.0 \times 10^{-6}$ | $7.46 \times 10^{-6}$ | $-17.1~\%$ |
| MAE | $2.21~\mathrm{kW}$ | $1.93~\mathrm{kW}$ | $-12.7~\%$ |
| RMSE | $2.96~\mathrm{kW}$ | $2.73~\mathrm{kW}$ | $-7.8~\%$ |
| Theory MAE | $17.01~\mathrm{kW}$ | $4.87~\mathrm{kW}$ | $-71.3~\%$ |
| Val MSE | $0.420$ | $0.387$ | $-7.9~\%$ |
