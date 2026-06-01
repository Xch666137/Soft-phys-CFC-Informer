# Experiments

## E01: Baseline comparison 鈥?PhysFormer vs. black-box Transformers
- **Verifies**: C01
- **Setup**:
  - Models: PhysFormer (V3/V4), Informer, iTransformer, LSTM
  - Hardware: AutoDL 5090 GPU
  - Dataset: VPP portfolio data with 4 components (Load, PV, Wind, Battery), 15-min resolution
  - System: AdamW optimizer, MSE loss, CosineAnnealing LR schedule, 96鈫?6 sequence length
- **Procedure**:
  1. Train each baseline from scratch with matched hyperparameters (where applicable)
  2. Train PhysFormer V3 (no component supervision) and V4 (component supervision)
  3. Evaluate Test MAE, Theory MAE, Val MSE on held-out test set
  4. Compare component-level errors where available (PhysFormer only)
- **Metrics**: Test MAE (kW), Theory MAE (kW), Val MSE
- **Expected outcome**: PhysFormer V4 achieves lower Theory MAE than all baselines while matching or exceeding best baseline Test MAE. PhysFormer V3 without component supervision should match baseline Test MAE but have higher Theory MAE than V4.
- **Baselines**: Informer, iTransformer, LSTM (all trained on same data split)
- **Dependencies**: none

## E02: Component loss weight sweep 鈥?finding the Pareto-optimal weight
- **Verifies**: C02
- **Setup**:
  - Model: PhysFormer V4 architecture (scalar residual)
  - Component loss weights: [0.01, 0.03, 0.05, 0.10, 0.20]
  - Hardware: AutoDL 5090 GPU
  - Dataset: Same as E01
  - System: Same training recipe as E01, varying only component_loss_weight
- **Procedure**:
  1. Train V4 at each weight for fixed epochs
  2. Record Test MAE and Theory MAE for each weight
  3. Plot Pareto frontier (Test MAE vs Theory MAE)
  4. Identify weight that minimizes distance to origin (optimal trade-off)
- **Metrics**: Test MAE, Theory MAE, per-component MAE
- **Expected outcome**: Weight 0.05 is near the Pareto-optimal knee. Weight 鈮?.10 pushes past the knee (Theory improves, Test degrades). Weight 鈮?.01 gives insufficient physics signal.
- **Baselines**: V3 (no component supervision, 位 = 0)
- **Dependencies**: E01

## E03: Component-consistent residual ablation 鈥?scalar vs. per-component
- **Verifies**: C03
- **Setup**:
  - Models: V4 (scalar residual, 1-dim) vs V5 (per-component residual, 5-dim)
  - Match: total parameters approximately equal, same training recipe
  - Hardware: AutoDL 5090 GPU
  - Dataset: Same as E01
  - System: Same training recipe, curriculum for V5, fixed-weight for V4
- **Procedure**:
  1. Train V4 and V5 from scratch
  2. Compare per-component MAE (Load, PV, Wind, Battery Power, Battery SOC)
  3. Compare aggregate Test MAE and Theory MAE
  4. Analyze residual statistics (mean, std) per component
  5. Check for cross-contamination: V4 residual correlation with Load error vs. PV error
- **Metrics**: Per-component MAE, aggregate Test MAE, Theory MAE, residual statistics
- **Expected outcome**: V5 achieves lower per-component MAE for PV, Wind, Battery. V5 Battery Power improvement is dramatic (>50%) due to elimination of cross-contamination. V5 aggregate MAE is comparable or slightly worse than V4 (due to higher effective component loss weight).
- **Baselines**: V4 scalar residual
- **Dependencies**: E02

## E04: Sigmoid gate ablation 鈥?identity vs. gated residual
- **Verifies**: C04
- **Setup**:
  - Models: V4 (identity residual) vs V4.1 (sigmoid-gated residual)
  - Match: all other hyperparameters identical
  - Hardware: AutoDL 5090 GPU
  - Dataset: Same as E01
  - System: Same training recipe
- **Procedure**:
  1. Train V4 and V4.1 from scratch
  2. Compare Theory MAE and Test MAE
  3. Inspect gate activation values 鈥?does the gate learn to close?
  4. Analyze gradient norms at theory branches for both variants
- **Metrics**: Theory MAE, Test MAE, gate activation distribution, gradient norms
- **Expected outcome**: V4.1 Theory MAE is worse than V4 (gate dampens gradient to theory branches). Gate does not learn a useful adaptive pattern 鈥?it either stays near 0.5 (uninformative) or drifts toward closing.
- **Baselines**: V4 identity residual
- **Dependencies**: E02

## E05: Curriculum vs. fixed-weight training (controlled)
- **Verifies**: C05
- **Setup**:
  - Models: V5 architecture (per-component residual)
  - Training strategies:
    - Curriculum: moderate component-loss schedule (validated as 0.03->0.02 in V7)
    - Fixed: lambda=0.02 and lambda=0.05 controls
  - Hardware: AutoDL 5090 GPU
  - Dataset: Same as E01
- **Procedure**:
  1. Train both variants from scratch for equal total epochs
  2. Compare early checkpoints and final checkpoints; explicitly check whether best validation occurs before or after the Phase1->Phase2 transition
  3. Compare final Test MAE and per-component MAE
  4. Plot Theory MAE vs. epoch for both and test whether curriculum provides robustness to weight choice rather than a superior late-stage optimization trajectory
- **Metrics**: Theory MAE trajectory, final Test MAE, final per-component MAE
- **Expected outcome**: Curriculum outperforms tested fixed-weight controls, but the supported mechanism is robustness to moderate component-loss weights. Do not claim Phase1 warmup -> Phase2 refinement unless best checkpoints occur after the transition under a clean rerun.
- **Baselines**: Fixed-weight lambda=0.02 and lambda=0.05
- **Dependencies**: E03

## E06: Component error asymmetry analysis
- **Verifies**: C06
- **Setup**:
  - Model: V4 (best aggregate) or V5 (best per-component)
  - Dataset: Same as E01
- **Procedure**:
  1. Compute per-component MAE normalized by component capacity (kW/kW_rated)
  2. Compute error CV (std/mean) per component to assess predictability
  3. Analyze Load error by hour-of-day and weekday/weekend
  4. Compare Load error distribution vs. PV/Wind error distributions
  5. Assess whether Load error is systematic (bias) or stochastic (variance)
- **Metrics**: Normalized per-component MAE, error CV, hourly Load error heatmap
- **Expected outcome**: Load normalized MAE is 5-20x higher than Wind/PV. Load error is highly structured by time-of-day (not random), suggesting a systematic modeling gap rather than irreducible noise. Load error decomposition shows both bias (under-predict morning ramp) and variance (evening peak timing).
- **Baselines**: PV/Wind error as within-model controls
- **Dependencies**: E02
