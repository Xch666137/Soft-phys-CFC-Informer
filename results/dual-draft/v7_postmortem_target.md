# V7 Parallel Ablation Post-Mortem Analysis Target

## Context

PhysFormer VPP net power forecasting. Architecture: shared Transformer encoder + FiLM-conditioned physics branches (PV, Wind, Battery) + behavioral Load branch (Temp MLP + GRU temporal module) + per-component residual heads. Training: CosineAnnealingWarmRestarts (T_0=20, warmup=5), AdamW (lr=1e-4, wd=1e-5), 2-phase curriculum (Phase1 epochs 1-8: component_loss_weight=0.03, Phase2 epochs 9+: cw=0.02).

## Experiment Design

3 parallel ablations on AutoDL vGPU-32GB (RTX 4080 SUPER), all using V6 S2 gru64 base architecture (load_gru_hidden=64, temp in GRU, restart_t0=20):

| ID | Config | Temperature Model | Loss Schedule |
|----|--------|------------------|---------------|
| V6 S2 gru64 | baseline | Temp MLP (1->32->32->1, 1153 params) | curriculum 0.03->0.02 |
| V7 BP | physformer_v7_balance_point | LearnedBalancePoint (5 params) | curriculum 0.03->0.02 |
| C05 w002 | physformer_c05_fixed_w002 | Temp MLP (1153 params) | fixed λ=0.02 |
| C05 w005 | physformer_c05_fixed_w005 | Temp MLP (1153 params) | fixed λ=0.05 |

## LearnedBalancePoint Model

load_temp = α_h * softplus(T_bal_h − T) + α_c * softplus(T − T_bal_c)
Parameters: T_bal_h (init 18°C), T_bal_c (init 24°C), α_heat_raw, α_cool_raw (softplus positive), base_offset.

## Final Test Metrics

| Metric | V6 S2 gru64 | V7 BP | C05 w002 | C05 w005 |
|--------|------------|-------|----------|----------|
| MAE | 0.002017 | 0.002016 | 0.002148 | 0.002058 |
| MSE | 7.58e-06 | 7.76e-06 | 8.14e-06 | 7.78e-06 |
| RMSE | 0.002753 | 0.002786 | 0.002853 | 0.002789 |
| Theory MAE | 0.002602 | 0.002920 | 0.002661 | 0.002946 |
| Residual Mean | -0.000949 | -0.000120 | -0.000213 | -0.000975 |
| Residual Std | 0.002276 | 0.003304 | 0.002597 | 0.002969 |
| Ramp Viol % | 0.0037 | 0.0056 | 0.0032 | 0.0035 |
| Load MAE | 0.001936 | 0.002495 | 0.002124 | 0.002168 |
| PV MAE | 0.002551 | 0.004233 | 0.003850 | 0.003738 |
| Wind MAE | 0.000319 | 0.000852 | 0.000653 | 0.000686 |
| Batt P MAE | 0.001703 | 0.003590 | 0.003103 | 0.002649 |
| Batt SOC MAE | 0.008910 | 0.016826 | 0.016055 | 0.014618 |

## Convergence Pattern (Val MSE)

ALL 4 experiments converge to best Val MSE between epochs 4-9. NO experiment improves after epoch 14.

| Config | Best Epoch | Best Val MSE | Best post-epoch-14 |
|--------|-----------|-------------|-------------------|
| V6 S2 gru64 | 9 | 0.386476 | 0.422546 (ep15) |
| V7 BP | 4 | 0.427316 | 0.430146 (ep27) |
| C05 w002 | 7 | 0.377086 | 0.430822 (ep27) |
| C05 w005 | 6 | 0.407788 | 0.416501 (ep31) |

The CosineAnnealingWarmRestarts at epoch 25 gave a transient Val MSE dip (C05 w005: 0.4238 at ep26, V7 BP: 0.4301 at ep27) but could not beat the early best. All experiments early-stopped at epoch 39 (patience=20, start_epoch=20).

## Previous V6.1 Results (for reference)

V6.1 2x2 ablation (N29: remove temp from GRU, N30: selective gradient detach) on same base architecture:

| Variant | MAE | MSE | Theory MAE | Residual Mean |
|---------|-----|-----|-----------|---------------|
| baseline | 0.001952 | 7.691e-06 | 0.002483 | -0.000949 |
| no_temp (N29) | 0.001999 | 7.938e-06 | 0.002431 | -0.000849 |
| detach (N30) | 0.002031 | 8.161e-06 | 0.002342 | -0.000065 |
| full (N29+N30) | 0.002032 | 8.214e-06 | 0.002509 | -0.000849 |

## Five Questions for Analysis

1. **BalancePoint paradox**: V7 BP has 5 params vs Temp MLP's 1153. Its Theory MAE is +12.2% worse (0.002920 vs 0.002602), yet its Residual Mean is near-zero (-0.000120 vs -0.000949) and aggregate MAE is identical (0.002016 vs 0.002017). Why does a worse theory model achieve equal aggregate accuracy with a smaller residual correction? Is this a sign of a deeper architectural property?

2. **Convergence speed paradox**: ALL 4 experiments converge to their best Val MSE between epochs 4-9, with zero improvement after epoch 14 despite 25 more epochs of training + cosine restart at epoch 25. What causes this ultra-rapid convergence followed by a 25-epoch plateau? Is the optimizer, LR schedule, or model architecture the root cause?

3. **Gradient isolation validity**: V6.1 detach (N30) achieved Theory MAE -10% vs baseline (0.002342 vs 0.002602) by selectively detaching residual-head gradients from load/PV/wind theory branches. But V6.1 full (N29+N30) was antagonistic — too much isolation starved the encoder. Is the "sweet spot" of gradient isolation a genuine mechanism or an artifact of single-seed noise? Could a milder form (e.g., gradient scaling instead of binary detach) achieve better results?

4. **V7 path forward**: BalancePoint failed to beat Temp MLP. Should we: (a) abandon temperature-model improvements and focus on gradient isolation, (b) try Lagged Temperature States (N40), (c) try a hybrid (BalancePoint backbone + tiny MLP correction ~10 params), or (d) accept Temp MLP as "good enough" and move on?

5. **Optimizer and LR schedule**: AdamW + CosineAnnealingWarmRestarts (T_0=20) with warmup=5 shows ultra-fast convergence followed by 25-epoch plateau. Would a different optimizer (Adam, SGD with momentum, LAMB) or schedule (OneCycleLR, flat+decay, cyclic with shorter T_0) better suit this model's convergence dynamics? Given that best checkpoints are at epochs 4-9, could we train for only 15 epochs and get the same results?
