# Peer Review Consensus Report

- File: `results/dual-draft/v6_1_analysis_target.md`
- Reviewers: **claude** vs **codex**
- Matching rule: loose (v2: category equivalents + range overlap), tolerance=2

## Summary
| Bucket | Count |
|---|---:|
| Consensus (both flagged) | 5 |
| Only claude | 4 |
| Only codex | 4 |
| **Total unique** | **13** |

`claude` reported 9 raw issues; `codex` reported 9 raw issues.

## 1. Consensus Issues (5)
_Both reviewers independently flagged the same location + (compatible) category._

### L13 — `missing_mechanism↔architecture` (severity: low/high)
- **claude**: The V6.1 ablation tests two mechanisms to reinforce the physics-data boundary, but doesn't test the complementary direction: should the residual head have MORE access to physics features (e.g., concatenating theory outputs without detach)? The current design only tests 'less connection', not 'different connection'.
  - Fix: `Consider an additional experiment: residual head receives theory outputs via a learnable gate (FiLM-style) rather than direct concatenation. This would test whether the problem is the connection TYPE rather than the connection STRENGTH.`
- **codex**: Feeding historical net_load into the load physics branch couples aggregate system behavior back into a component branch.
  - Fix: `Separate component-specific load history from aggregate net power, or feed the GRU deweathered/load-only residual features with explicit masking of PV, wind, and battery effects.`

### L19 — `training_dynamics` (severity: high/high)
- **claude**: Single seed (2024) — all conclusions are statistically unverified. no_temp vs baseline MAE difference is only 0.000018 MW (0.9%), which could fall entirely within run-to-run variance.
  - Fix: `Run 3+ seeds per config and report mean ± std. If resources are limited, at minimum run baseline and the best variant (no_temp) with 2 additional seeds each to estimate variance.`
- **codex**: Selective detach cuts useful final-loss gradients from most physics branches during Phase 2.
  - Fix: `Replace hard detach with a scheduled gradient gate, auxiliary consistency losses, or branch-specific gradient clipping so physics branches still receive controlled signal from forecast error.`

### L19 — `architecture↔missing_mechanism` (severity: high/medium)
- **claude**: N29 and N30 are antagonistic when combined: full (N29+N30) is worse than either no_temp (N29 alone) or detach (N30 alone) on nearly all metrics. Both mechanisms target gradient isolation — removing temp reduces input overlap, selective detach blocks gradient flow — and stacking them creates a double bottleneck.
  - Fix: `Frame this as a key finding: the physics-data boundary has a 'sweet spot' of gradient isolation. Too little (baseline) allows cross-contamination; too much (full) starves the shared encoder of learning signal. Either N29 or N30 alone hits the sweet spot; combining them overshoots.`
- **codex**: Removing current temperature from the GRU does not make the Temp MLP the sole temperature pathway.
  - Fix: `Validate the boundary with temperature-perturbation tests and remove or orthogonalize temperature-correlated history/calendar proxies if sole-path behavior is required.`

### L28 — `architecture↔missing_mechanism` (severity: high/medium)
- **claude**: No single configuration dominates all metrics — baseline wins MSE/RMSE/Ramp, no_temp wins MAE, detach wins Theory MAE and components. The thesis needs a principled selection criterion, not just 'pick the best per metric'.
  - Fix: `Define a composite score (e.g., 0.5×MAE + 0.3×MSE + 0.2×Theory_MAE) or declare a primary metric a priori. For VPP dispatch applications, ramp violations are safety-critical and should be weighted heavily — baseline wins this by a large margin.`
- **codex**: Ramp violations are measured but the design does not include a mechanism to enforce ramp feasibility.
  - Fix: `Add a differentiable ramp-rate penalty, constrained output projection, or battery-aware feasibility layer on the final net-power trajectory.`

### L73 — `missing_mechanism↔architecture` (severity: medium/high)
- **claude**: The V6.1 ablation tests N29 vs N30, but the curriculum training itself (C05 claim) remains untested. All experiments used the same curriculum schedule — there's no fixed-weight baseline to confirm that curriculum training is superior to constant component_loss_weight.
  - Fix: `Either run a controlled curriculum-vs-fixed-weight experiment (recommended for thesis completeness), or explicitly scope C05 as 'not tested in this work, left for future study' to avoid overclaiming.`
- **codex**: Detaching physics features only blocks gradients; it does not stop the residual head from learning to undo the physics outputs.
  - Fix: `Constrain the residual head with residual magnitude budgets, zero-mean or orthogonality penalties, and conditioning on exogenous error features rather than raw theory outputs alone.`

## 2. Only from claude (4)

- **v6_1_analysis_target.md:33** `[high/physics_fidelity]` detach ramp violation (0.0090%) is 2.4× baseline (0.0037%) — a safety-critical degradation. The near-zero residual mean (−0.000065) comes at the cost of worse extreme-event handling.
  - Original: `Ramp Violation: baseline 0.0037%, detach 0.0090%`
  - Fix: `Investigate the time-series shape of detach predictions vs baseline during high-ramp periods. Selective detach may cause the residual head to under-correct during rapid net-load changes because gradient isolation prevents the encoder from learning ramp-aware representations.`
  - Why: For VPP dispatch, ramp violations can cause real economic losses (imbalance charges). A 2.4× increase may be unacceptable regardless of theory improvement. This trade-off must be explicitly discussed in the thesis.

- **v6_1_analysis_target.md:35** `[medium/physics_fidelity]` detach residual mean (−0.000065 MW) is essentially zero — the residual head is perfectly unbiased. This is the strongest evidence that selective detach works as designed: the residual becomes a true zero-mean correction rather than a systematic compensator.
  - Original: `Residual Mean: baseline −0.000949, detach −0.000065`
  - Fix: `Emphasize this in the thesis as the primary evidence for N30's mechanism. The near-zero residual mean means the theory branches are producing physically correct baseline predictions, and the residual head only corrects unbiased noise — exactly the intended 'physics-guided + data-driven correction' paradigm.`
  - Why: The residual mean is a direct measure of whether the physics layer is producing unbiased estimates. detach achieves this perfectly. This is the cleanest ablation signal in the entire V6.1 experiment and should be the headline result for N30.

- **v6_1_analysis_target.md:36** `[medium/missing_mechanism]` All V6.1 variants improve every component MAE over baseline, yet all degrade aggregate MSE. This is the classic physics-accuracy Pareto tradeoff: component-consistent predictions don't necessarily sum to better aggregate forecasts because component errors can cancel in the aggregate.
  - Original: `All component MAEs better in V6.1 variants, but MSE worse`
  - Fix: `Compute the correlation matrix of per-component residuals to check whether component errors are anti-correlated (canceling in aggregate) in baseline but not in V6.1 variants. If confirmed, this explains the paradox and provides a mechanistic narrative for the tradeoff.`
  - Why: A reviewer will ask: 'If every component is more accurate, why is the aggregate worse?' The thesis must have an answer. Error cancellation among components is the most likely mechanism and should be verified.

- **v6_1_analysis_target.md:44** `[high/training_dynamics]` no_temp and full show early-peak-then-degrade pattern (best Val MSE at ep10-12, then deteriorate), while detach peaks later (ep26) and degrades less. This suggests N29 causes the model to find a sharp local minimum that doesn't generalize, while N30 provides regularization.
  - Original: `no_temp: ep12 0.410 → ep20 0.442; detach: ep26 0.412 → stable`
  - Fix: `Report the generalization gap (best Val MSE vs final Test MAE) as a metric. Consider whether the restart_t0=20 warm restart at epoch 20 and 40 interacts with this — no_temp's degradation accelerates after epoch 20 restart, suggesting periodic LR resets destabilize the temp-free GRU.`
  - Why: This is counter-intuitive: removing temperature (simplifying the model) causes worse generalization. The mechanism may be that temperature in the GRU acts as an implicit regularizer by forcing the GRU to learn a joint representation of load and weather, rather than overfitting to load autocorrelation alone.

## 3. Only from codex (4)

- **v6_1_analysis_target.md:10** `[medium/physics_fidelity]` The scalar unconstrained Temp MLP is too weakly physical to replace a degree-day style load model.
  - Original: `- **Temp MLP**: 3-layer MLP (1→32→32→1) for temperature→load response`
  - Fix: `Use a constrained temperature-response module with heating/cooling balance points, monotonic piecewise basis functions, and optional lagged temperature states.`
  - Why: Load response to temperature is asymmetric, thresholded, and often hysteretic; an unconstrained MLP can learn nonphysical shapes while still fitting aggregate error.

- **v6_1_analysis_target.md:20** `[medium/physics_fidelity]` Keeping only battery gradients intact makes the battery branch a likely slack variable for errors from detached branches.
  - Original: `  influencing physics branch learning (battery gradient kept intact)`
  - Fix: `Either apply the same detach policy to all physical branches or constrain the battery branch with explicit SOC dynamics, charge/discharge limits, and energy-balance penalties.`
  - Why: The residual head can push battery representations to absorb load, PV, or wind mismatch, which can improve aggregate loss while degrading physical interpretability.

- **v6_1_analysis_target.md:24** `[medium/implementation]` The ablation uses a single seed, so the reported differences are not reliable enough to choose a thesis configuration.
  - Original: `2×2 factorial ablation. All share: restart_t0=20, load_gru_hidden=64, phase_1_epochs=8,
phase_2_epochs=50, patience=20, early_stop_start_epoch=20, seed=2024.`
  - Fix: `Run multiple seeds with confidence intervals and paired statistical comparisons for Test MAE, Theory MAE, ramp violations, and component metrics.`
  - Why: The metric deltas are small enough that optimizer stochasticity and early stopping variance could change the ranking.

- **v6_1_analysis_target.md:81** `[medium/training_dynamics]` Decaying component loss in the same phase that applies detach can leave physics branches under-supervised.
  - Original: `3. **Phase structure**: Phase 1 (epochs 1-8): component_loss_weight=0.03, detach_mode="none".
   Phase 2 (epochs 9-50): component_loss_weight decays, detach_mode set per config.`
  - Fix: `Maintain a nonzero per-branch component loss floor, monitor gradient norms per branch, and delay detach until component validation metrics have stabilized.`
  - Why: Once residual-head gradients are removed and component loss decays, the load/PV/wind branches may receive too little signal to stay aligned with the final forecast objective.
