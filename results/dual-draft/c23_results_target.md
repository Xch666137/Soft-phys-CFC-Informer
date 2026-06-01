# C23 Experiment Results — Analysis Target

## Context

PhysFormer VPP net power forecasting. Testing three code modifications:

- **C-2**: Phase 1→2 Adam optimizer state soft transition (preserve exp_avg/exp_avg_sq instead of hard zeroing)
- **C-3**: Three-stage curriculum cw decay: 0.03 (epoch 0-7) → 0.02 (epoch 8-29) → 0.01 (epoch 30-49)
- **e3**: Encoder depth e_layers=3 (vs default 2)

All experiments: seed=2025, V6 S2 gru64 architecture, OneCycleLR pct_start=0.16, early_stop_start_epoch=5/patience=8.

## Results

| Config | MAE | MSE | Theory MAE | RampViol% |
|--------|-----|-----|------------|-----------|
| p1a_baseline_s2025 (old: hard reset, 2-stage cw=0.03→0.02, e2, no detach) | 0.002009 | 7.666e-6 | 0.002904 | 0.0 |
| p1a_detach_s2025 (old: hard reset, 2-stage, e2, detach) | 0.001988 | 7.952e-6 | 0.002257 | 0.0 |
| c23_baseline (C-2+C-3, e2, no detach) | 0.002058 | 8.661e-6 | 0.002240 | 0.0 |
| c23_detach (C-2+C-3, e2, detach) | 0.001993 | 7.928e-6 | 0.002485 | 0.0 |
| c23_e3 (C-2+C-3, e3, no detach) | 0.002160 | 8.607e-6 | 0.002604 | 0.0 |

### Component MAE

| Config | Load | PV | Wind | BattP | BattSOC |
|--------|------|----|------|-------|---------|
| c23_baseline | 0.00187 | 0.00194 | 0.00032 | 0.00160 | 0.00768 |
| c23_detach | 0.00201 | 0.00305 | 0.00040 | 0.00243 | 0.01421 |
| c23_e3 | 0.00215 | 0.00380 | 0.00072 | 0.00316 | 0.01619 |

### Effect Isolation

1. **C-2+C-3 vs old (baseline, no detach)**: Theory -22.9%, MAE +2.4%
2. **C-2+C-3 vs old (detach)**: Theory +10.1%, MAE +0.3%
3. **detach effect under C-2+C-3**: MAE -3.1%, Theory +10.9%
4. **e3 effect under C-2+C-3**: MAE +5.0%, Theory +16.3%
5. **c23_baseline has best Theory MAE overall (0.002240)**, beating even old detach (0.002257)

## Key Questions for Review

1. Why does C-2+C-3 improve Theory by 22.9% in baseline mode but worsen it by 10.1% in detach mode?
2. Why does e3 worsen both aggregate and theory under C-2+C-3, reversing N57's finding that e3 improved physics?
3. c23_detach components show severe degradation (PV +57%, BattP +52%, BattSOC +85%) vs c23_baseline — is this the expected detach trade-off or an interaction bug?
4. Is the 0.0% RampViol across all experiments suspicious? (all values exactly zero)
5. c23_baseline achieves best Theory MAE (0.002240) but second-worst MAE (0.002058) — does this confirm H03's Pareto trade-off or suggest the 3-stage cw undershot the knee?

## Code Changes Under Review

File: `physformer/train/physformer_exp.py`
- L274-303: Phase logic with soft/hard transition + 3-stage phases (2a/2b)
- L217-229: `_log_phase_transition` helper method
- L225-230: OneCycleLR pct_start=0.16

File: `physformer/loss.py`
- L165-218: PhysLoss with phase_2a_cw parameter, set_phase("2a") support

File: `configs/base/v5_base.yaml`
- L64-65: phase_2a_epochs, phase_2a_cw
- L71: phase_reset_mode: soft
