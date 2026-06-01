# Peer Review Consensus Report

- File: `?`
- Reviewers: **A** vs **codex**
- Matching rule: loose (v2: category equivalents + range overlap), tolerance=2

## Summary
| Bucket | Count |
|---|---:|
| Consensus (both flagged) | 6 |
| Only A | 1 |
| Only codex | 2 |
| **Total unique** | **9** |

`A` reported 7 raw issues; `codex` reported 8 raw issues.

## 1. Consensus Issues (6)
_Both reviewers independently flagged the same location + (compatible) category._

### L1 — `training_dynamics` (severity: high/medium)
- **A**: 
  - Fix: `-`
- **codex**: Preserving Adam moments across a phase/loss-regime transition can carry stale gradient statistics into a different objective.
  - Fix: `Use a damped transition such as scaling Adam moments by a small factor, resetting only affected parameter groups, or warming the new loss weight over several epochs while monitoring gradient norms.`

### L1 — `training_dynamics` (severity: high/medium)
- **A**: 
  - Fix: `-`
- **codex**: The curriculum stage boundary at epoch 8 coincides with OneCycleLR's likely learning-rate peak from pct_start=0.16 over 50 epochs.
  - Fix: `Decouple LR and curriculum transitions by moving the curriculum boundary away from the LR peak, or introduce a short transition ramp with gradient-norm diagnostics.`

### L1 — `physics_fidelity↔training_dynamics` (severity: high/high)
- **A**: 
  - Fix: `-`
- **codex**: Early stopping can terminate training before the third curriculum stage is reached or meaningfully evaluated.
  - Fix: `Make early stopping phase-aware, for example disable it until after epoch 30 plus a minimum dwell time, or require validation improvement checks separately within each curriculum stage.`

### L1 — `architecture` (severity: medium/medium)
- **A**: 
  - Fix: `-`
- **codex**: The proposal bundles optimizer-state transition and curriculum-weight changes, creating a coupled experiment that cannot attribute effects to either mechanism.
  - Fix: `Run a 2x2 ablation with old/new optimizer transition crossed against old/new curriculum, then evaluate detach and e3 only after the base interaction is understood.`

### L1 — `missing_mechanism` (severity: medium/medium)
- **A**: 
  - Fix: `-`
- **codex**: All conclusions are based on a single random seed despite small reported differences between configurations.
  - Fix: `Repeat each configuration across multiple seeds and report mean, standard deviation, and paired deltas for MAE, Theory MAE, and component metrics.`

### L1 — `implementation` (severity: low/medium)
- **A**: 
  - Fix: `-`
- **codex**: Phase behavior is split across trainer logic, mutable loss state, and config fields, increasing integration risk.
  - Fix: `Centralize phase scheduling in a single scheduler object that emits explicit loss weights and reset actions per epoch, and pass those values into the loss call instead of mutating loss phase state.`

## 2. Only from A (1)

- **results/dual-draft/c23_results_target.md §Results table** `[medium/correctness]` 
  - Fix: `-`
  - Why: -

## 3. Only from codex (2)

- **c23_results_target.md:17** `[high/physics_fidelity]` RampViol is exactly 0.0 for every configuration, which makes the ramp constraint non-discriminative as a physics metric.
  - Original: `| p1a_baseline_s2025 ... | 0.0 |
| p1a_detach_s2025 ... | 0.0 |
| c23_baseline ... | 0.0 |
| c23_detach ... | 0.0 |
| c23_e3 ... | 0.0 |`
  - Fix: `Validate the RampViol implementation with synthetic violating predictions, report raw ramp margins or violation magnitudes, and check whether clipping, normalization, or threshold scaling is masking violations.`
  - Why: A constraint metric that is always zero cannot guide design decisions and may indicate a disabled, incorrectly scaled, or trivially satisfied physics check.

- **c23_results_target.md:27** `[medium/physics_fidelity]` Aggregate MAE hides large component-level regressions in PV, battery power, and battery SOC.
  - Original: `| c23_baseline | 0.00187 | 0.00194 | 0.00032 | 0.00160 | 0.00768 |
| c23_detach | 0.00201 | 0.00305 | 0.00040 | 0.00243 | 0.01421 |
| c23_e3 | 0.00215 | 0.00380 | 0.00072 | 0.00316 | 0.01619 |`
  - Fix: `Add component-level acceptance gates and physics constraints, especially SOC transition consistency and battery power feasibility, rather than selecting primarily on net MAE or Theory MAE.`
  - Why: VPP net-power accuracy can improve through cancellation between physically wrong components, which weakens deployability and interpretability.
