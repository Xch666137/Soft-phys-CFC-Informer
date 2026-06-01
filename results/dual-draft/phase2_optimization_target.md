# PhysFormer Phase 2 Optimization Failure — Root Cause Analysis & Solution Design

## Problem Statement

PhysFormer uses a two-phase curriculum training strategy:
- **Phase 1** (epochs 1-8): Strong physics supervision (cw=0.03, rr=0.05, tw=0.2)
- **Phase 2** (epochs 9+): Weak physics supervision (cw=0.01, rr=0.01, tw=0.1)

Starting from V5, Phase 2 **consistently fails to improve Val MSE beyond Phase 1's best**. This has been observed across V5, V5.1, and V5.2.

## Training Data Evidence

### V5 (no detach)
```
Phase 1 best: Epoch 8, Val MSE = 0.4127
Phase 2:      Epoch 9-19, Val MSE = 0.41-0.43 (oscillating, never below 0.4127)
Restart:      Epoch 22 briefly hits 0.4038, immediately bounces back to 0.41+
```

### V5.2 (full detach of physics_features)
```
Phase 1 best: Epoch 7, Val MSE = 0.4089
Phase 2:      Epoch 9-19, Val MSE = 0.43-0.45 (worse than Phase 1!)
Restart:      Epoch 20-39, Val MSE = 0.42-0.44 (stuck, early stop at epoch 39)
```

### V5.2 Test Results (vs V5.1)
| Metric | V5.1 | V5.2 | Change |
|--------|------|------|--------|
| MAE | 0.002019 | 0.002133 | +5.6% |
| Theory MAE | 0.004400 | 0.002936 | -33.3% |
| Batt Power MAE | 0.001960 | 0.021346 | +989% |
| Batt SOC MAE | 0.010823 | 0.020113 | +85.8% |

## Current Architecture

```
Input → Encoder → TemporalDecoder → WeatherFusion → PhysicsFiLM → UnifiedResidualHead
                                                                           ↓
PhysicalLayer → theory_net + physics_features ──────────────────────→ component_residual
                                                                           ↓
                                                    pred_net = theory_net + residual
```

Loss function:
```
total_loss = net_mse
           + theory_loss_weight * theory_mse      (0.1 in Phase 2)
           + soc_weight * soc_bounds_loss          (0.1)
           + component_loss_weight * component_mae (0.01 in Phase 2)
           + res_reg_weight * res_reg              (0.01 in Phase 2)
```

## Key Files
- `physformer/models/physformer.py` — forward pass, detach_mode
- `physformer/models/conditioning.py` — PhysicsFiLM, UnifiedResidualHead
- `physformer/utils/losses.py` — PhysLoss with phase-aware weights
- `physformer/exp/exp_physformer.py` — training loop, phase switching
- `configs/physformer_v5_3.yaml` — latest config

## Root Cause Hypothesis

### H1: Loss landscape突变 + LR太小
Phase 1's strong physics supervision creates a specific basin in loss landscape. When Phase 2 suddenly reduces physics loss weights, the gradient direction changes. But by epoch 9, LR has decayed to ~8e-5, too small to escape Phase 1's basin.

### H2: Full detach creates optimization dead zone (V5.2 specific)
With `physics_features.detach()`, phys_layer loses gradient from residual loss. Only theory_mse (weight 0.1) provides gradient to phys_layer — insufficient for improvement.

### H3: Phase 2 loss weights too weak to provide meaningful signal
cw=0.01 and rr=0.01 are so small that physics supervision becomes negligible. The dominant signal (net_mse) doesn't directly improve physics representations.

### H4: Shared encoder creates gradient conflict
Both theory and residual pathways share the same encoder. Phase 1 optimizes encoder for physics; Phase 2's net_mse pulls encoder in a different direction, degrading physics representations.

## Constraints
- No new loss terms (user preference: avoid multi-loss complexity)
- Keep curriculum structure (Phase 1 strong physics → Phase 2 refinement)
- Must improve both Theory MAE and Aggregate MAE simultaneously
- Battery component accuracy must not collapse

## Questions for Review
1. Is the Phase 2 failure fundamental to the curriculum design, or fixable with better hyperparameters?
2. Should we abandon two-phase training entirely, or fix the transition?
3. What is the minimal change that would make Phase 2 actually improve over Phase 1?
4. Is selective detach (V5.3 approach) sufficient, or does the LR strategy also need to change?
5. Are there alternative training strategies (e.g., single-phase with adaptive loss weights) that would be more robust?
