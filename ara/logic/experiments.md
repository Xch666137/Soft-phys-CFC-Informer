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

## E07: Gradient isolation non-monotonic optimum — 2x2 factorial ablation
- **Verifies**: C07
- **Setup**:
  - Models: V6 S2 gru64 baseline vs N29 (remove temp from GRU) vs N30 (selective gradient detach) vs N29+N30 (full)
  - Hardware: AutoDL vGPU-32GB (RTX 4080 SUPER, 32GB)
  - Seeds: s2024 (single-seed), s2025+s2026 (P1-A multi-seed replication)
  - Dataset: NextGen DVPP, 15-min resolution
- **Procedure**: Train 4 arms in 2x2 factorial design. Compare aggregate MAE, Theory MAE, component MAE. Validate cross-seed stability.
- **Metrics**: Test MAE (MW), Theory MAE (MW), per-component MAE, cross-seed ranking stability
- **Expected outcome**: N29 and N30 each individually improve over baseline; N29+N30 degrades (antagonistic). detach > baseline > no_temp ranking stable across seeds.
- **Result**: detach 3-seed mean MAE=0.001998 (best). no_temp unstable - collapse in s2026 (Load 6.6x). Full (N29+N30) worst. Ranking confirmed across 3 seeds.
- **Baselines**: V6 S2 gru64 (temp in GRU, detach_mode=none)
- **Dependencies**: E06, N33, N34, N35, N55

## E08: Component error covariance variance decomposition
- **Verifies**: C08
- **Setup**:
  - Runs: V6 S2 gru64 vs gru96 (capacity regime); c23 baseline/e3/detach (3-stage regime)
  - Method: Extract y_aux from test DataLoader, decompose Var(error_net) = sum(Var) - 2*sum(Cov) per net = load - pv - wind + batt
  - Scripts: scripts/c08_extract_and_analyze.py, scripts/c08_vgpu_3way.py, scripts/c08_t1_3seed.py
- **Procedure**: Decompose aggregate net error variance into per-component diagonal variance + cross-component covariance. Compute cancellation ratio.
- **Metrics**: Per-component bias (MW), diag variance sum (MW2), cross-cov contribution (MW2), cancellation ratio
- **Expected outcome**: Larger shared capacity -> stronger error correlations -> more cancellation -> aggregate preserved (capacity regime). Gradient isolation -> weaker correlations -> less cancellation (isolation regime).
- **Result**: gru96 cancellation 79.8% vs gru64 66.7%. e3 largest diag var + strongest cancellation = worst aggregate. AMD 2-stage: best aggregate + all 5 components degraded (C08 validated).
- **Baselines**: Within-experiment component-to-aggregate comparison
- **Dependencies**: E03, N38, N74, N83

## E09: Selective detach aggregate dominance - multi-seed replication
- **Verifies**: C09
- **Setup**:
  - Models: c23 architecture - baseline (joint gradient), e3 (deeper encoder), detach (selective gradient detach)
  - Seeds: 2025/2026/2027 (T1 multi-seed), plus May 23 s2025 (single-seed)
  - Hardware: AutoDL vGPU-32GB (RTX 4080 SUPER, 32GB)
- **Procedure**: Train 3 arms x 3 seeds = 9 runs. Compare aggregate Test MAE/MSE/RMSE/net_ramp_violation mean +/- std.
- **Metrics**: Test MAE (MW), Test MSE (MW2), Test RMSE (MW), net_ramp_violation, cross-seed std
- **Expected outcome**: detach achieves lowest mean AND smallest cross-seed std on all aggregate metrics.
- **Result**: detach MAE=1.973e-3+/-3.7e-5 (best), baseline=2.069e-3+/-1.34e-4, e3=2.101e-3+/-5.3e-5. detach lowest on all metrics + lowest std.
- **Baselines**: c23 baseline (joint gradient), e3 (deeper encoder)
- **Dependencies**: E07, N82

## E10: Detach mechanism search - detachxe3 cross-validation
- **Verifies**: C10
- **Setup**:
  - E1 (remote): c23 detach + e_layers=3, seeds 2025/2026/2027
  - D1 (local): |cov(theory, residual)| analysis across 12 runs
  - D2 (local): residual fraction distribution analysis
  - D3 (local): sharpness perturbation (finite-difference, 30 dirs x 3 eps x 9 ckpts)
- **Procedure**: Test 4 candidate mechanisms against E1 adjudication. If detachxe3 = baseline (not detach), depth useless under detach -> M1 capacity-cancellation supported.
- **Metrics**: detachxe3 Test MAE vs detach and baseline (E1); |cov| ratio (D1); residual fraction quantiles (D2); sharpness values (D3)
- **Expected outcome**: detachxe3 MAE = baseline, not detach. D1/D2 thresholds not met.
- **Result**: detachxe3 MAE=2.065e-3 = baseline (2.069e-3) >> detach (1.973e-3). D1 |cov| ratio=0.87 (FAIL). D2 |DeltaPearson|=0.018 (FAIL). M1 capacity-cancellation SUPPORTED.
- **Baselines**: T1 detach, T1 baseline, T1 e3
- **Dependencies**: E09, N88, N89, N90, N91

## E11: Component-token separation - A1 iTransformer vs shared-encoder baseline
- **Verifies**: C11
- **Setup**:
  - Models: A1 (8-token inverted Transformer, 5 component GRU + 3 weather MLP, zero physics, zero graph bias, 1.3M params) vs c23 baseline (shared encoder + FiLM + per-component residual, ~3M params)
  - Seeds: 2025/2026/2027 (3 seeds per arm)
  - Hardware: AutoDL vGPU-32GB (RTX 4080 SUPER, 32GB)
  - Dataset: NextGen DVPP, multi-portfolio, 15-min resolution
- **Procedure**: Train A1 from scratch on net MSE only. Compare aggregate Test MAE/MSE/RMSE mean +/- std vs c23 baseline.
- **Metrics**: Test MAE (MW), Test MSE (MW2), Test RMSE (MW), cross-seed std(MAE)
- **Expected outcome**: A1 Test MAE < c23 baseline on >=2/3 seeds with smaller cross-seed variance.
- **Result**: A1 MAE=0.001811+/-6e-6 vs c23 0.002069+/-1.34e-4 (-12.5%). A1 std 20x smaller. Gate PASSED (3/3 seeds).
- **Baselines**: c23 baseline (shared-encoder PhysFormer)
- **Dependencies**: E10, N100

## E12: Fixed physics prior monotonic degradation - A2-A5 ablation chain
- **Verifies**: C12
- **Setup**:
  - A1 (8 tokens, simple FFN decoder) - baseline
  - A2 (+1 physics token)
  - A3 (+5 twin + 3 constraint tokens, 16 total) - compound, individual contributions confounded
  - A4 (+graph bias annealing)
  - A5 (+per-step CrossAttention horizon decoder + weather)
  - Seeds: 2025/2026/2027 (3 seeds per variant)
  - Hardware: AutoDL vGPU-32GB (RTX 4080 SUPER, 32GB)
- **Procedure**: Train each variant from scratch on net MSE only. Compare Test MAE and Val MSE vs A1.
- **Metrics**: Test MAE (MW), E10 Val MSE ratio vs A1, Val->Test divergence
- **Expected outcome**: Strict monotonic degradation: A1 < A2 < A3 < A4 < A5 on Test MAE, Val MSE improving oppositely.
- **Result**: A1 0.001811 < A2 0.001819 < A3 0.001843 < A4 0.001863 < A5 0.001947. Val improves monotonically. Val-better->Test-worse universal.
- **Baselines**: A1 (pure 8-token iTransformer)
- **Dependencies**: E11, N102, N103, N104, N105

## E13: Masked Component Pretraining - repaired B1/R1-reg decomposable forecasting
- **Verifies**: C13 (decomposable forecasting part)
- **Setup**:
  - Architecture: A1 8-token PhysFormer-iGT contract (5 component GRU tokens + 3 weather MLP tokens)
  - Pretraining: MCP with randomly masked component history channels, component MAE on masked channels, and lambda_net=1.0 net MSE anchor
  - Repairs: N131 protocol repair (canonical checkpoint, fatal missing pretrained path, 8-token pretrain/finetune/test contract, iGT SOC placeholder semantics, use_compile=false)
  - Downstream arms: R0 direct test, R1 low-LR finetune, R1-reg tiny component-anchor finetune, R2 few-shot target adaptation
  - Seeds: R1/R1-reg use 2025/2026/2027; R2 uses f05/f10/f20 target-prefix fractions
  - Hardware: AutoDL vGPU-32GB (RTX 4080 SUPER, 32GB)
- **Procedure**:
  1. Pretrain B1 lam10 under repaired N131 protocol.
  2. Export canonical best_val_net and final pretrained checkpoints.
  3. Run R0, R1, R1-reg, and R2 using the same checkpoint and 8-token model contract.
  4. Compare aggregate Test MAE and learned 4-component MAE (Load, PV, Wind, Battery Power) against A1.
  5. Exclude Battery SOC from learned component evidence because iGT has no learned SOC head.
- **Metrics**: Test MAE (MW), 4-component MAE (Load, PV, Wind, Battery Power), cross-seed mean +/- std, protocol-integrity checks
- **Expected outcome**: Repaired B1/R1-reg trades a small aggregate-MAE cost (<5% vs A1) for substantially better learned component decomposability.
- **Result**: N132 repaired run completed. R0 direct MAE=0.001882. R1 low-LR 3-seed mean MAE=0.001834672 +/- 0.000003128 with component means Load=0.001814825, PV=0.000943461, Wind=0.000301731, BatteryPower=0.001184180. R1-reg 3-seed mean MAE=0.001829685 +/- 0.000002955 with component means Load=0.001808796, PV=0.000898363, Wind=0.000230729, BatteryPower=0.001185496. R2 few-shot degraded aggregate MAE to 0.002255-0.002311. All logs verify canonical checkpoint loading, use_compile=false, cuDNN enabled, and scaler-buffer max diff=0.
- **Baselines**: A1 from-scratch net MSE (aggregate MAE=0.001811; component values treated as C08 cancellation artifacts)
- **Dependencies**: E11, E12, N129, N131, N132

## E14: Dispatch proxy validation - component-aware allocation vs net-only allocation (pending)
- **Verifies**: C13 (operational dispatch value part)
- **Status**: planned / pending execution
- **Setup**:
  - Inputs: saved test-set predictions for A1 and the best repaired decomposable arm (currently R1-reg), plus ground-truth 4-component trajectories.
  - Dispatch target: construct a deterministic net-adjustment requirement from held-out test horizons, such as peak-shaving target, ramp-smoothing target, or reserve-following target. The target must be fixed before evaluation and shared by all arms.
  - Controllable assets: use a simple hierarchy consistent with available data: Battery Power first, then PV/Wind curtailment, then flexible Load adjustment if represented. If hard asset limits are unavailable, estimate conservative per-timestep headroom from observed component histories and mark the result as proxy-only.
  - Baselines: (a) A1 net-only forecast + static allocation by historical component share; (b) A1 net-only + uniform allocation; (c) random feasible allocation as a sanity floor.
  - Component-aware arm: R1-reg component forecasts allocate the required net adjustment according to predicted per-DER availability/headroom.
- **Procedure**:
  1. Define the net target sequence and component feasibility bounds before seeing outcomes.
  2. For each model arm and horizon, compute the adjustment vector that attempts to meet the target under the same bounds and cost weights.
  3. Evaluate realized dispatch using ground-truth components, not predicted components.
  4. Report aggregate residual deviation, infeasible-command rate, ramp/peak penalty, and simple weighted cost.
  5. Run the same protocol across multiple target scenarios (peak shaving, ramp smoothing, reserve tracking) or explicitly scope to one scenario.
- **Metrics**: realized net deviation MAE, infeasible adjustment rate, weighted dispatch cost, ramp/peak penalty, component command error
- **Falsification protocol**:
  - **Claim**: Component-aware R1-reg forecasts improve dispatch proxy outcomes over net-only A1 allocation.
  - **Pass condition**: R1-reg beats both A1 allocation baselines on realized cost/deviation in at least two target scenarios without higher infeasible-command rate.
  - **Fail condition**: A1 net-only + simple allocation matches or beats R1-reg on realized cost/deviation, or R1-reg produces higher infeasible-command rate.
  - **Confounds**: If headroom/bounds are estimated rather than measured, the result supports only a proxy operational claim, not field-ready dispatch validation.
- **Baselines**: A1 net-only + historical-share allocation; A1 net-only + uniform allocation; random feasible allocation
- **Dependencies**: E13, C13, N132
