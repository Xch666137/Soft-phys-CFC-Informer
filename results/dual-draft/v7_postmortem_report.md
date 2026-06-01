# Peer Review Consensus Report

- File: `results/dual-draft/v7_postmortem_target.md`
- Reviewers: **claude** vs **codex**
- Matching rule: loose (v2: category equivalents + range overlap), tolerance=2

## Summary
| Bucket | Count |
|---|---:|
| Consensus (both flagged) | 2 |
| Only claude | 4 |
| Only codex | 12 |
| **Total unique** | **18** |

`claude` reported 6 raw issues; `codex` reported 14 raw issues.

## 1. Consensus Issues (2)
_Both reviewers independently flagged the same location + (compatible) category._

### L3 — `physics_fidelity↔training_dynamics` (severity: medium/medium)
- **claude**: Gradient isolation (selective detach) mechanism is physically sound and effect size (-10% Theory MAE) exceeds likely seed variance, but battery-as-slack variable pathology (RampViol +144%) needs mitigation before thesis claims
  - Fix: `Three-tier validation: (1) Multi-seed (≥3) replication of detach to confirm effect size and quantify seed variance. (2) Test gradient SCALING (residual→encoder gradient multiplied by α=0.3, 0.5, 0.7) instead of binary detach — this creates a continuous isolation parameter and may find a sweet spot with better RampViol. (3) Add battery ramp penalty to the loss to counteract the slack-variable effect — a soft constraint on battery power rate-of-change during detach training.`
- **codex**: The curriculum reduces component supervision after epoch 8 even though the best checkpoints occur around epochs 4-9.
  - Fix: `Use normalized per-component losses with adaptive weighting, or pretrain theory branches before enabling residual heads instead of lowering component pressure near convergence.`

### L4 — `missing_mechanism↔architecture` (severity: medium/high)
- **claude**: V7 path forward should prioritize gradient isolation over temperature model improvements — the residual head's robustness to temp model quality makes further temp architecture work low-ROI for thesis completion
  - Fix: `Priority order for thesis: (1) Multi-seed validate detach gradient isolation (highest impact: Theory MAE -10%, novel mechanism). (2) Accept Temp MLP as the Load model — it works, the residual head compensates for its imperfections, and BP's failure actually strengthens the narrative ('physics constraints help but data-driven flexibility is needed for behavioral components'). (3) Lagged Temperature States (N40) is worth a QUICK test only if (1) succeeds quickly — 1 experiment, 15 epochs with OneCycleLR, check if Load MAE improves. Otherwise skip.`
- **codex**: The architecture lets unconstrained residual heads compensate for weak physics branches, so aggregate accuracy can improve while component fidelity degrades.
  - Fix: `Constrain residuals as bounded corrections after physically valid component predictions, add residual magnitude or orthogonality penalties, and select checkpoints with component-level constraints in addition to aggregate loss.`

## 2. Only from claude (4)

- **v7_postmortem_target.md:Q1** `[high/architecture]` BalancePoint paradox: worse theory (+12.2% MAE) but near-zero residual mean (-0.000120 vs -0.000949) and identical aggregate MAE — root cause is bias-variance tradeoff between constrained and unconstrained temp models
  - Original: `V7 BP Theory MAE 0.002920 vs V6 S2 0.002602 (+12.2%), but Residual Mean -0.000120 vs -0.000949`
  - Fix: `Reframe the finding: BalancePoint is a better PRIOR (unbiased, residual near zero) but worse FIT (noisy, residual std 0.003304 vs 0.002276). Temp MLP is a worse prior (biased, residual -0.000949) but better fit (std 0.002276). The residual head can compensate for both noise and bias, making aggregate MAE insensitive to temp model architecture. This is actually a POSITIVE finding for PhysFormer: the residual mechanism provides robustness to theory model quality.`
  - Why: The 5-param BP model is smooth and monotonic — it captures the broad temperature-load shape correctly (unbiased) but can't fit fine nonlinearities (high noise). The 1153-param MLP overfits local patterns, introducing systematic bias but reducing per-sample noise. The residual head compensates differently in each case: for BP it applies large per-sample corrections, for MLP it applies a systematic offset. The aggregate accuracy parity is not a BP failure — it's evidence that the residual mechanism is effective regardless of theory model quality. This should be framed as an architectural strength, not a BP failure.

- **v7_postmortem_target.md:Q2** `[high/training_dynamics]` Universal ultra-fast convergence (best at epochs 4-9) followed by 25-epoch plateau across all 4 experiments points to encoder representational capacity as the bottleneck, not optimizer or loss schedule
  - Original: `ALL 4 experiments converge to best Val MSE between epochs 4-9. NO experiment improves after epoch 14.`
  - Fix: `Three concrete actions: (1) Reduce total epochs to 15-20 with early_stop_start_epoch=5 — saves 60-70% compute with identical results. (2) Test if increasing encoder capacity (d_model=512, e_layers=3-4) shifts the plateau epoch later — if yes, confirms encoder bottleneck hypothesis. (3) Consider OneCycleLR with peak at epoch 5, total 15 epochs — matches the observed convergence dynamics.`
  - Why: The common element across all 4 experiments is the shared Transformer encoder (d_model=256, e_layers=2, n_heads=8). All converge at the same time despite different temp models and loss schedules. This means the encoder saturates its representational capacity by epoch 4-9, after which neither theory-branch tuning nor residual-head refinement can extract more information. The cosine restart at epoch 25 gives a transient perturbation but can't overcome the encoder ceiling. This is a fundamental architectural finding, not a training issue.

- **v7_postmortem_target.md:Q5** `[high/training_dynamics]` Current AdamW + CosineAnnealingWarmRestarts (T_0=20, warmup=5) is severely mismatched to the observed convergence dynamics — 25 of 39 epochs (64%) are wasted in plateau with zero improvement
  - Original: `ALL experiments converge at epochs 4-9 with zero improvement after epoch 14. Cosine restart at epoch 25 doesn't beat early bests.`
  - Fix: `Three concrete improvements: (1) Shorten T_0 from 20 to 8-10 — matches the actual convergence window; the current T_0=20 means the LR decreases monotonically for 20 epochs while the model is already converged. (2) Switch to OneCycleLR with peak_lr=1e-4, total_epochs=20, pct_start=0.25 — LR rises for 5 epochs (matching warmup), then decays for 15, matching the fast-converge-then-refine pattern. (3) Add SWA (Stochastic Weight Averaging) starting at epoch 5 — averages checkpoints across the plateau, which is known to improve generalization for models with fast convergence + long plateau dynamics. For future experiments, train for only 20 epochs with early_stop_start_epoch=5 — saves 50% compute.`
  - Why: The T_0=20 cosine schedule was designed for models that need 20+ epochs to converge. PhysFormer converges in 4-9 epochs. The cosine schedule then wastes 25 epochs slowly decreasing LR while the model oscillates in a saturated basin. The cosine restart at epoch 25 provides a temporary LR boost but (a) the encoder is already at capacity, so the model can't escape its basin, and (b) the restart period (T_0=20) is 2-5x longer than the actual convergence window. OneCycleLR is designed precisely for this pattern: fast rise to peak LR, then gradual annealing. SWA would extract value from the plateau by averaging, rather than letting it go to waste.

- **v7_postmortem_target.md:Q5** `[low/implementation]` AdamW weight_decay=1e-5 may be too low for a model with small parameter count in theory branches — the tiny weight decay provides negligible regularization
  - Original: `AdamW (lr=1e-4, wd=1e-5)`
  - Fix: `Increase weight_decay to 1e-3 for theory branch parameters specifically (differential weight decay). The theory branches have very few parameters (BP: 5, MLP: 1153) and need stronger regularization to prevent overfitting to noise in the temperature-load relationship. Keep encoder wd=1e-5.`
  - Why: AdamW with wd=1e-5 on a model with ~1M total params means the effective regularization is ~0.01 per parameter per epoch — essentially zero. The theory branches (5-1153 params) fit a 1D function (temperature→load) from limited data (3 years, 15-min resolution) and would benefit from stronger regularization. The ultra-fast convergence + subsequent plateau is partially consistent with weak regularization: the model quickly fits the training signal, then oscillates because there's no penalty for doing so.

## 3. Only from codex (12)

- **results/dual-draft/v7_postmortem_target.md:20** `[medium/physics_fidelity]` The balance-point load model is instantaneous and omits thermal inertia, lagged temperature exposure, humidity, calendar effects, and occupancy-like structure.
  - Original: `load_temp = α_h * softplus(T_bal_h − T) + α_c * softplus(T − T_bal_c)`
  - Fix: `Use lagged temperature states or degree-hour features, optionally with a small constrained correction network over temperature history and calendar context.`
  - Why: Building load typically responds to accumulated thermal state rather than only current temperature, so a static curve is likely misspecified.

- **results/dual-draft/v7_postmortem_target.md:20** `[medium/physics_fidelity]` Plain softplus produces nonzero heating or cooling contribution at the balance point, which biases the physical interpretation of the setpoints.
  - Original: `load_temp = α_h * softplus(T_bal_h − T) + α_c * softplus(T − T_bal_c)`
  - Fix: `Use shifted smooth hinges such as softplus(kx)/k - softplus(0)/k with learnable or fixed sharpness, or a constrained piecewise-linear deadband model.`
  - Why: At x=0, softplus is positive, so the model does not represent zero heating or cooling demand at the learned threshold without relying on a compensating offset.

- **results/dual-draft/v7_postmortem_target.md:21** `[medium/physics_fidelity]` The heating and cooling balance points are independent, so the learned model can collapse or invert the deadband.
  - Original: `Parameters: T_bal_h (init 18°C), T_bal_c (init 24°C), α_heat_raw, α_cool_raw (softplus positive), base_offset.`
  - Fix: `Parameterize T_bal_c = T_bal_h + min_gap + softplus(delta_gap), and regularize setpoints to plausible HVAC ranges.`
  - Why: Without an ordering constraint, optimization can create overlapping heating and cooling regimes that are physically hard to justify.

- **results/dual-draft/v7_postmortem_target.md:21** `[low/implementation]` The design declares a base_offset parameter that is absent from the displayed load_temp equation.
  - Original: `Parameters: T_bal_h (init 18°C), T_bal_c (init 24°C), α_heat_raw, α_cool_raw (softplus positive), base_offset.`
  - Fix: `Either include + base_offset in the formal model and test it, or remove the parameter from the implementation and documentation.`
  - Why: A mismatch between the mathematical spec and parameter list creates risk of a dead parameter or an undocumented bias term.

- **results/dual-draft/v7_postmortem_target.md:27** `[high/physics_fidelity]` The aggregate MAE hides severe component degradation, especially for V7 BP across load, PV, wind, battery power, and SOC.
  - Original: `| MAE | 0.002017 | 0.002016 | 0.002148 | 0.002058 |
| Load MAE | 0.001936 | 0.002495 | 0.002124 | 0.002168 |
| PV MAE | 0.002551 | 0.004233 | 0.003850 | 0.003738 |
| Wind MAE | 0.000319 | 0.000852 | 0.000653 | 0.000686 |
| Batt P MAE | 0.001703 | 0.003590 | 0.003103 | 0.002649 |
| Batt SOC MAE | 0.008910 | 0.016826 | 0.016055 | 0.014618 |`
  - Fix: `Promote component metrics to gating criteria, report Pareto tradeoffs, and require physically valid component decomposition before accepting equal net-power accuracy.`
  - Why: For VPP forecasting, a model that gets net power right through cancellation can still be physically wrong and operationally unsafe.

- **results/dual-draft/v7_postmortem_target.md:31** `[medium/training_dynamics]` Residual Mean is being interpreted as residual correction size even though V7 BP has much higher residual variance.
  - Original: `| Residual Mean | -0.000949 | -0.000120 | -0.000213 | -0.000975 |
| Residual Std | 0.002276 | 0.003304 | 0.002597 | 0.002969 |`
  - Fix: `Report residual MAE, RMSE, signed bias, variance, autocorrelation, and per-component residual energy before concluding that correction is smaller.`
  - Why: A near-zero mean can simply indicate unbiased corrections, while the residual head may still be doing more work through larger positive and negative swings.

- **results/dual-draft/v7_postmortem_target.md:37** `[high/missing_mechanism]` Battery power and SOC errors suggest the design lacks a hard battery state-transition mechanism.
  - Original: `| Batt P MAE | 0.001703 | 0.003590 | 0.003103 | 0.002649 |
| Batt SOC MAE | 0.008910 | 0.016826 | 0.016055 | 0.014618 |`
  - Fix: `Model SOC with a differentiable state update using charge/discharge efficiency, capacity, bounds, and ramp constraints, and penalize or project invalid trajectories.`
  - Why: SOC is a cumulative state, so predicting it as another loose branch output can violate energy conservation and amplify small battery-power errors over time.

- **results/dual-draft/v7_postmortem_target.md:42** `[medium/training_dynamics]` The learning-rate schedule is misaligned with observed convergence because the first restart happens long after all best checkpoints.
  - Original: `ALL 4 experiments converge to best Val MSE between epochs 4-9. NO experiment improves after epoch 14.

The CosineAnnealingWarmRestarts at epoch 25 gave a transient Val MSE dip ... but could not beat the early best.`
  - Fix: `Run an LR-range test and compare shorter schedules such as one-cycle over 8-12 epochs, flat-then-decay, or cosine with T_0 shorter than the observed convergence window.`
  - Why: A restart at epoch 25 cannot address optimization dynamics that have already plateaued by epoch 14.

- **results/dual-draft/v7_postmortem_target.md:44** `[medium/implementation]` Validation MSE values are on a very different scale from final test MSE, but the design does not define the metric transform or units.
  - Original: `| Config | Best Epoch | Best Val MSE | Best post-epoch-14 |
| V6 S2 gru64 | 9 | 0.386476 | 0.422546 (ep15) |
| V7 BP | 4 | 0.427316 | 0.430146 (ep27) |`
  - Fix: `Report whether validation MSE is normalized, scaled, or composite, and include raw-unit validation metrics comparable to the test table.`
  - Why: Without metric-scale clarity, conclusions about convergence, checkpoint quality, and schedule effectiveness are hard to audit.

- **results/dual-draft/v7_postmortem_target.md:51** `[medium/scalability]` The early-stopping configuration wastes most of the training budget after the useful convergence window.
  - Original: `All experiments early-stopped at epoch 39 (patience=20, start_epoch=20).`
  - Fix: `Start early stopping shortly after warmup, use patience around 5-8 with min_delta, and cap this ablation family near 15 epochs unless later improvements are demonstrated.`
  - Why: If no run improves after epoch 14, continuing to epoch 39 consumes roughly two thirds of compute without improving model selection.

- **results/dual-draft/v7_postmortem_target.md:55** `[medium/implementation]` The previous V6.1 baseline is treated as comparable even though its metrics differ materially from the current V6 S2 baseline.
  - Original: `V6.1 2x2 ablation (N29: remove temp from GRU, N30: selective gradient detach) on same base architecture:

| baseline | 0.001952 | 7.691e-06 | 0.002483 | -0.000949 |`
  - Fix: `Rerun the gradient-isolation ablation under the exact current code, data split, seed protocol, and metric pipeline, then report confidence intervals.`
  - Why: Cross-run baseline drift can make the apparent benefit of detach or temperature changes look larger or smaller than it really is.

- **results/dual-draft/v7_postmortem_target.md:70** `[high/training_dynamics]` Binary gradient detach is a blunt isolation mechanism that can improve theory metrics while starving the shared encoder or hurting aggregate accuracy.
  - Original: `V6.1 detach (N30) achieved Theory MAE -10% vs baseline ... by selectively detaching residual-head gradients from load/PV/wind theory branches. But V6.1 full (N29+N30) was antagonistic — too much isolation starved the encoder.`
  - Fix: `Replace binary detach with gradient scaling, PCGrad/GradNorm-style conflict handling, or staged residual unfreezing, and evaluate with multi-seed paired runs.`
  - Why: The reported antagonism is exactly the failure mode expected when shared representations receive incomplete or conflicting gradients.
