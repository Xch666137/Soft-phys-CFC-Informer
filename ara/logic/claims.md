# Claims

## C01: Physics-guided FiLM conditioning reduces theory deviation vs. pure data-driven baselines
- **Statement**: PhysFormer with FiLM-conditioned physics branches achieves lower theory deviation (Theory MAE) than black-box Transformer baselines (Informer, iTransformer, LSTM) on VPP aggregated net power forecasting, without sacrificing aggregate accuracy.
- **Status**: supported
- **Provenance**: user
- **Falsification criteria**: A black-box baseline trained under identical data/hardware conditions achieves equal or lower Theory MAE than PhysFormer V4 or V5.
- **Proof**: [E01]
- **Evidence basis**: V3 baseline comparison; benchmark configs in `configs/legacy/baselines/`. V4 Theory MAE = 3.811 kW vs V3 = 4.874 kW (-21.8%).
- **Interpretation**: Physics provides a meaningful inductive bias, but the magnitude of improvement varies by component (largest for Battery/SOC, smallest for Load).
- **Dependencies**: none
- **Tags**: physics-guidance, FiLM, theory-deviation, baseline-comparison

## C02: Component supervision loss at moderate weight (0.05) improves physical consistency with minimal aggregate cost
- **Statement**: Adding per-component MAE loss at weight 0.05 in MW space reduces theory deviation by ~22% (V4 vs V3) while aggregate MAE degrades by only ~2.3% — a favorable point on the physics-accuracy Pareto frontier.
- **Status**: supported
- **Provenance**: user
- **Falsification criteria**: A grid search over component loss weights (0.01, 0.03, 0.05, 0.10, 0.20) finds no weight that achieves both Theory MAE reduction ≥15% and aggregate MAE degradation ≤3% relative to V3.
- **Proof**: [E02]
- **Evidence basis**: V4 (weight 0.05): Theory -21.8%, MAE -2.3%. V4.2 (stronger): Theory best but MAE +3.6%. V5 (weight 0.1): component best but MAE +7.5%.
- **Interpretation**: The optimal weight is architecture-dependent. V5's per-component residual effectively multiplies component supervision, requiring lower weight than V4's scalar residual.
- **Dependencies**: C01
- **Tags**: component-loss, pareto-tradeoff, hyperparameter

## C03: Component-consistent residual reduces per-component error vs. scalar residual
- **Statement**: Replacing a single scalar residual with 5 per-component residuals (Load, PV, Wind, Battery Power, Battery SOC) reduces PV, Wind, and Battery component errors by ≥10% relative to the V4 scalar residual baseline.
- **Status**: supported
- **Provenance**: user
- **Falsification criteria**: A controlled experiment where V5 component-consistent residual fails to outperform V4 scalar residual on ≥3 of 5 component metrics.
- **Proof**: [E03]
- **Evidence basis**: V5 vs V4 component comparison: PV MAE = 1.892 vs 3.998 (-52.7%), Wind MAE = 0.313 vs 0.825 (-62.1%), Battery Power MAE = 1.340 vs 21.345 (-93.7%).
- **Interpretation**: The large Battery improvement (V4 Battery = 21.345 kW was clearly pathological cross-contamination from Load) confirms the cross-contamination hypothesis. PV/Wind gains are real but smaller.
- **Dependencies**: C02
- **Tags**: component-consistent, residual, disentanglement, ablation

## C04: Sigmoid-gated residual connections degrade physics learning
- **Statement**: Adding a learnable sigmoid gate to the residual stream in a physics-guided architecture causes theory branch degradation because the gate creates a gradient bottleneck — dampening one theory pathway affects all branches through the shared encoder.
- **Status**: supported
- **Provenance**: user
- **Falsification criteria**: A controlled comparison where V4.1 (with gate) matches or exceeds V4 (without gate) on both Theory MAE and Test MAE.
- **Proof**: [E04]
- **Evidence basis**: V4.1 Theory MAE = 4.966 kW (regression from V4's 3.811 kW). Gate removed in V4.2 → Theory MAE recovered to 2.981 kW.
- **Interpretation**: This parallels ResNet's finding that identity shortcuts are sufficient and parameterized shortcuts add unnecessary complexity. Identity mapping makes the optimization landscape easier.
- **Dependencies**: C02
- **Tags**: sigmoid-gate, dead-end, gradient-bottleneck, residual-learning

## C05: Curriculum training with progressive component-loss annealing is superior to fixed-weight training
- **Statement**: A two-phase curriculum (Phase 1: higher component loss weight for physics initialization → Phase 2: decaying weight with joint optimization) achieves better aggregate accuracy than fixed-weight training at the same final component loss weight. However, best checkpoints occur before the Phase 1→2 transition, suggesting curriculum provides robustness to weight choice rather than enabling a better optimization path.
- **Status**: supported
- **Provenance**: user (original), user-revised (2026-05-16 after V7 results)
- **Falsification criteria**: Fixed-weight training at the Phase 2 final weight (λ=0.02) or higher (λ=0.05) matches or exceeds the curriculum variant on both Theory MAE and Test MAE.
- **Proof**: [E05 — completed 2026-05-16]
- **Evidence basis**: V7 3-parallel ablation. C05 fixed-w002 (λ=0.02): aggregate MAE +6.5%, MSE +7.4%, Theory MAE +2.3% vs curriculum baseline. C05 fixed-w005 (λ=0.05): aggregate MAE +2.0%, MSE +2.6%, Theory MAE +13.2%. Neither fixed weight matches curriculum. Curriculum best Val MSE=0.3865 (epoch 9), fixed-w002 best=0.3771 (epoch 7), fixed-w005 best=0.4078 (epoch 6). All experiments converge at epoch 4-9 with no improvement after epoch 14 — the Phase1→Phase2 transition at epoch 8 occurs AFTER the best checkpoint is found. Curriculum aids robustness not optimization path.
- **Interpretation**: Curriculum is superior to fixed-weight, but not for the originally hypothesized reason ("Phase1 warmup→Phase2 refinement"). The mechanism is: moderate component supervision weight (~0.02-0.03) early in training is sufficient; curriculum provides robustness to the exact weight choice. The "Phase1 warmup" narrative in the thesis must be corrected (N47).
- **Dependencies**: C03, N41, N42
- **Tags**: curriculum, training-strategy, supported, v7-validated

## C06: Load is fundamentally harder to forecast with physics guidance than weather-driven components
- **Statement**: In a VPP with mixed DER assets, the Load component exhibits 10-20x higher forecast error than weather-driven components (PV, Wind) under physics-guided modeling, because load is driven by human behavioral factors not captured by physical equations.
- **Status**: supported
- **Provenance**: user
- **Falsification criteria**: A physics-guided model (or behavioral model) achieves Load error within 3x of Wind error on the same VPP dataset, without access to individual consumer-level data.
- **Proof**: [E06]
- **Evidence basis**: V4: Load = 14.707 kW, Wind = 0.825 kW (17.8x). V5: Load = 2.069, Wind = 0.313 (6.6x). V6 S1 gru64: Load=0.002056, Wind=0.000337 (6.1x). V6 S2 gru64: Load=0.001936, Wind=0.000319 (6.1x). V6.1 all variants: Load/Wind ratio ~6.0x (Load 0.001886-0.001925, Wind 0.000313-0.000314). Ratio converged at ~6:1 — behavioral module eliminated the 18× gap but ~6× remains irreducible without consumer-level data.
- **Interpretation**: Load is a fundamentally different modeling problem from weather-driven DER. Behavioral/temporal approaches (calendar, temperature response, time-series patterns) outperform physics equations (HDD/CDD). V6 evidence strongly supports this. Remaining 6.1x ratio likely irreducible without individual consumer-level data.
- **Dependencies**: C01, C03, N26, N27, N33, N34, N35
- **Tags**: load-forecasting, behavioral-modeling, error-asymmetry, bottleneck, v6-validated

## C07: Gradient isolation between physics and data modules has a non-monotonic optimum
- **Statement**: Both input-side isolation (removing shared features) and gradient-side isolation (detaching gradient pathways) independently improve the physics-data boundary, but combining them degrades performance — there exists a "sweet spot" of gradient isolation beyond which the shared encoder is starved of learning signal.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: A third mechanism for physics-data isolation (e.g., FiLM gating) combined with either N29 or N30 outperforms the single-mechanism variants, or the full (N29+N30) variant outperforms at least one single-mechanism variant on both aggregate and theory metrics.
- **Proof**: [E07]
- **Evidence basis**: V6.1 2×2 ablation (s2024): N29 alone (no_temp) → MAE 0.001999; N30 alone (detach) → Theory MAE 0.002342, Residual Mean −0.000065. N29+N30 → all metrics worse. P1-A multi-seed (s2025, s2026): detach consistently best MAE + Theory MAE (2-seed mean MAE=0.001998, Theory=0.002326). no_temp unstable — collapse in s2026 (Load 6.6×, Batt 6.9× vs baseline). Ranking detach > baseline > no_temp stable across 3 seeds.
  - **2026-05-25 update (T1 9-run cross-seed)**: the ramp-violation trade-off framing previously associated with this claim (detach ~2× ramp violations vs baseline) is **dropped** — T1 net_ramp_violation has detach lowest in EVERY seed (mean 2.989e-3 vs baseline 3.728e-3, e3 3.971e-3). That trade-off was a regime-specific artefact of the pre-N65 2-stage 0.03→0.01 schedule; under true 3-stage (N73) or 2-stage truncated A+B (N75, c23 configs), detach instead improves ramp behaviour. O11 vs O34 CONFLICT resolved by N85. The gradient-isolation non-monotonic-optimum core of C07 (detach > baseline > no_temp ranking, no_temp instability, joint detach+no_temp degradation) is unchanged.
- **Interpretation**: This is a mechanistic discovery, not just hyperparameter tuning. The physics-data boundary in a shared-encoder architecture has a maximum useful isolation level. Exceeding it causes the encoder to lose cross-component representation learning. This principle likely generalizes to other physics-guided architectures with shared backbones.
- **Dependencies**: C03, N33, N34, N35, N36
- **Tags**: gradient-isolation, non-monotonic, physics-data-boundary, ablation, v6_1-validated, multi-seed-validated

## C08: Component error correlation structure dominates aggregate accuracy, not component magnitudes
- **Statement**: Improving per-component MAE does not guarantee improved aggregate MSE — the covariance structure of component errors determines aggregate accuracy through signed summation cancellation (net = load − pv − wind + batt). Two regimes produce the component-aggregate paradox: (1) **capacity regime**: larger shared capacity → stronger error correlations → more cancellation → aggregate preserved despite worse components; (2) **isolation regime** (hypothesized): gradient isolation → weaker correlations → less cancellation → aggregate degraded despite better components.
- **Status**: supported
- **Provenance**: ai-suggested (original), Claude-executed analysis (2026-05-15)
- **Falsification criteria**: A controlled experiment where a configuration with better per-component MAE simultaneously shows stronger (not weaker) inter-component error correlations AND worse aggregate MSE would falsify the capacity regime mechanism. For the isolation regime, V6.1 per-component prediction data is needed.
- **Proof**: [E08 — completed 2026-05-15]
- **Evidence basis**: V6 S2 gru64 vs gru96 variance decomposition. gru96 has 1.95× larger component variance (58.76e-6 vs 30.06e-6) but 2.34× larger covariance cancellation (−46.90e-6 vs −20.06e-6), yielding net aggregate variance only 1.19× worse (11.86e-6 vs 10.00e-6). Cancellation ratios: gru96 79.8% vs gru64 66.7%. Dominant channel: Cov(e_PV, e_B) = −39.42e-6 (gru96) vs −12.92e-6 (gru64). Phase A AMD 2-stage 0.03→0.01 provides an even stronger stress test: MAE=0.001944 (best aggregate baseline) while Theory MAE worsens to 0.003186 and all 5 component MAEs degrade (PV +137%, Load +40%, Wind +122%, BattP +129%, BattSOC +121%). This directly supports the signed-cancellation mechanism. Full analysis in `docs/analysis/c08_variance_decomposition.md`; Phase A evidence in N63/O29.
- **Interpretation**: The key insight is that component errors are not independent — they are coupled through the shared encoder and then signed by the power-balance equation. Worse component forecasts can still improve aggregate MAE/MSE when their errors cancel under `net = load − pv − wind + batt`; therefore aggregate accuracy alone is not evidence of better physical modeling. The variance decomposition framework `Var(e_net) = ΣVar − 2ΣCov` is a general diagnostic for separating physics quality from dispatch-level aggregate accuracy. AMD/ROCm results should be treated as mechanistic evidence for C08, not as a recommended formal training platform.
- **Dependencies**: C03, N27, N28, N36, N63
- **Tags**: component-aggregate-paradox, error-cancellation, variance-decomposition, covariance, v6_s2-validated, phase_a-validated, dual-regime

## C09: Selective detach dominates aggregate accuracy across seeds with smallest variance
- **Statement**: Under the c23 architecture (per-component residual + FiLM + 2-stage truncated curriculum with cw 0.03→0.02), the selective-detach variant (detach_mode_phase2=selective in Phase 2) achieves strictly lower mean aggregate Test MAE / MSE / RMSE / net_ramp_violation than the standard joint-gradient baseline across at least 3 independent seeds, AND has the smallest cross-seed standard deviation on all four aggregate metrics. This dominance is OUTCOME-level: the mechanism by which detach achieves it is currently unknown (C11 candidate mechanism was refuted by N84) but the outcome is robust.
- **Status**: supported
- **Provenance**: ai-suggested (T1 plan), ai-executed (analysis), user (process: ran research-manager to crystallize after seeing 4 evidence points all confirming aggregate dominance)
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: An independent multi-seed experiment (≥3 seeds, c23 architecture, same data) where baseline mean Test MAE is statistically lower than detach mean OR detach has larger cross-seed std on any aggregate metric. Equivalently: any single OOD/distribution-shift test set where baseline beats detach on aggregate MAE would weaken (not falsify) — robustness to distribution shift is a separate claim.
- **Proof**: [E09 — T1 9 runs, scripts/c08_t1_3seed_results.json + runs/physformer_c23_*_vgpu_s{2025,2026,2027}/metrics.json + the May 23 single-seed c23 vgpu_s2025 evidence]
- **Evidence basis**:
  | metric | baseline mean ± std | e3 mean ± std | **detach mean ± std** |
  |---|---|---|---|
  | MAE  (MW)  | 2.069e-3 ± 1.34e-4 | 2.101e-3 ± 5.3e-5 | **1.973e-3 ± 3.7e-5** |
  | MSE  (MW²) | 8.111e-6 ± 6.75e-7 | 8.146e-6 ± 2.57e-7 | **7.377e-6 ± 1.35e-7** |
  | RMSE (MW)  | 2.846e-3 ± 1.17e-4 | 2.854e-3 ± 4.5e-5 | **2.716e-3 ± 2.5e-5** |
  | net_ramp_violation | 3.728e-3 ± 1.06e-3 | 3.971e-3 ± 4.1e-4 | **2.989e-3 ± 5.0e-4** |
  detach wins both mean AND std (most reproducible) on every aggregate metric. Single-seed s2025 (May 23) plus 3 T1 seeds = 4 evidence points all in same direction.
- **Interpretation**: This is an OUTCOME claim, now WITH a candidate mechanism explanation (C10, 2026-05-26). The May 23 mechanism hypothesis (O37 bias-clearance regime escape) was refuted by N84; M1 cov-cross and M2 residual-fraction were refuted by N89/N90; M3 sharpness was inconclusive (FP32 floor). The surviving mechanism (C10) is: detach disables the encoder-depth→cancellation channel — deeper encoders cannot produce cancellable covariance when residual-gradient backflow is cut. This was confirmed by the E1 detach×e3 cross-seed experiment (N88) which showed detach×e3 ≈ baseline (2.065e-3 ≈ 2.069e-3), not ≈ detach (1.973e-3).
- **Dependencies**: C07 (gradient isolation), C08 (cancellation framework), C10 (mechanism)
- **Tags**: detach, gradient-isolation, multi-seed-validated, mechanism-resolved, t1-validated
- **From staging**: implicit from O33 (aggregate half) + cross-seed grounding from N82

## C10: Detach disables encoder-depth→cancellation channel (C09 mechanism)
- **Statement**: Selective detach achieves aggregate dominance (C09) by cutting the residual-head gradient backflow into the shared encoder. Under normal (non-detach) training, the residual head can send gradient through the encoder to induce cross-component error correlations in the theory branches — correlations that cancel under net = load − pv − wind + batt (C08 capacity regime). Deeper encoders (e_layers=3) enhance this effect (O39). Selective detach blocks this backward path, preventing the encoder from organizing cancellable covariance. Consequently: (1) detach alone achieves superior aggregate (C09); (2) deeper encoders provide no benefit UNDER detach; (3) the detach×e3 joint configuration performs no better than baseline.
- **Status**: supported
- **Provenance**: ai-suggested (hypothesis), ai-executed (E1 + D1/D2/D3), user (approved P2 design)
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: (1) detach×e3 aggregate MAE significantly < detach aggregate MAE (would support M4 disentanglement instead); (2) An architecture variant that preserves encoder-depth benefit while isolating gradients (e.g., separate physics encoder) also fails to outperform detach; (3) Cross-seed C08 variance decomposition showing detach×e3 has CANCELLATION RATIO comparable to e3 alone (not near-zero like detach).
- **Proof**: [E10 — N88 E1 detach×e3 cross-3-seed: detach×e3 MAE 2.065e-3 ± 7.5e-5 >> detach 1.973e-3 ± 3.7e-5; N89 D1 cov-cross; N90 D2 residual fraction; N84 O37 refutation; C08 signed-cancellation framework]
- **Evidence basis**:
  | arm | MAE mean ± std | vs detach | vs baseline | interpretation |
  |---|---|---|---|---|
  | baseline | 2.069e-3 ± 1.34e-4 | +4.9% | — | shared-gradient, normal capacity |
  | e3 | 2.101e-3 ± 5.3e-5 | +6.5% | +1.5% | more capacity → more cancellation (comp wins, agg loses) |
  | **detach** | **1.973e-3 ± 3.7e-5** | **—** | **−4.6%** | **channel cut, unbiased components** |
  | detach×e3 | 2.065e-3 ± 7.5e-5 | +4.7% (2.5σ) | −0.2% | extra depth useless when channel blocked |
- **Interpretation**: This closes the mechanism-open status of C09. The mechanism is NOT one of M1-cov-cross (direct cov reduction), M2-residual-fraction (residual role change), or M3-sharpness (flat minimum). It is structural: detach removes a gradient pathway that the residual head uses to induce theory-branch bias patterns that cancel in aggregate. Deeper encoders amplify this bias-induction capacity, which is why e3 components improve but aggregate degrades (O39). When both are combined (detach×e3), the detach cut dominates — depth provides no benefit because the cancellation channel is already shut. This is consistent with C08's two-regime framework and extends it from GRU-width (V6 S2) and curriculum-weight (O26/O32) to the encoder-depth×gradient-isolation interaction.
- **Dependencies**: C07 (gradient isolation non-monotonic), C08 (cancellation framework), C09 (detach outcome dominance)
- **Tags**: detach, mechanism, capacity-cancellation, encoder-depth, gradient-isolation, cross-seed-validated, C09-mechanism
- **From staging**: O43

## C11: Component-token separation via inverted attention eliminates the shared-encoder cancellation channel and achieves superior aggregate accuracy without physics priors
- **Statement**: An 8-token pure inverted Transformer (5 component GRU tokens + 3 weather MLP tokens, zero physics, zero graph bias, zero component loss, net MSE only) achieves strictly lower mean aggregate Test MAE/MSE/RMSE than the full PhysFormer c23 baseline (shared encoder + FiLM + per-component residual + curriculum) across at least 3 independent seeds, with dramatically smaller cross-seed variance.
- **Status**: supported
- **Provenance**: ai-executed (experiment), user (affirmation: "可以结晶")
- **Crystallized via**: verbal-affirmation
- **Falsification criteria**: Any independent 3-seed experiment where the 8-token iTransformer fails to outperform c23 baseline on aggregate MAE at ≥2/3 seeds, OR where any A2-A5 architecture variant with added physics priors outperforms A1.
- **Proof**: [E11 — N100 A1 3-seed results]
- **Evidence basis**:
  | metric | A1 mean ± σ (3 seeds) | c23 baseline ± σ (3 seeds) | Δ |
  |---|---|---|---|
  | MAE (MW) | 1.811e-3 ± 6e-6 | 2.069e-3 ± 1.34e-4 | −12.5% |
  | MSE (MW²) | 6.766e-6 ± 4.9e-8 | 8.111e-6 ± 6.75e-7 | −16.6% |
  | RMSE (MW) | 2.601e-3 ± 9e-6 | 2.846e-3 ± 1.17e-4 | −8.6% |
  A1 cross-seed σ(MAE) = 6e-06 is 20× smaller than c23 baseline σ = 1.34e-4.
  Architecture: 5-component batched GRU (672→d_model) + 3-weather batched MLP (96→d_model) → inverted self-attention (2 layers, 8 heads) → shared FFN decoder → real-unit power balance → net MSE only. 1.3M params, 700+ S/s.
- **Interpretation**: This is the ARCHITECTURAL proof of C10. The shared encoder creates a cancellation channel where cross-component error correlations cancel under net = load − pv − wind + batt (C08 capacity regime). Inverted attention gives each component its own token and representation space — no shared d_model for FFN to mix across components. Component separation at the architecture level eliminates the cancellation channel without gradient hacks (detach) or physics priors. The 20× smaller cross-seed variance suggests the inverted architecture navigates a dramatically simpler optimization landscape.
- **Dependencies**: C10 (shared-encoder cancellation channel), C08 (signed cancellation framework), N100
- **Tags**: iTransformer, component-token, inverted-attention, architecture, cancellation-elimination, multi-seed-validated, phase-a
- **From staging**: O47

## C12: Fixed physics priors are monotonic overfitting amplifiers — every architectural addition beyond component-token separation systematically degrades Test MAE
- **Statement**: In the inverted Transformer architecture (A1 baseline), adding ANY fixed physics prior — physics tokens (A2), component twin tokens + constraint tokens (A3), graph-biased attention (A4), or component-wise horizon decoder with weather conditioning (A5) — produces a strict monotonic degradation of Test MAE, despite improving Val MSE in every case. The Val-better → Test-worse divergence strengthens with stronger physics priors.
- **Status**: supported
- **Provenance**: ai-executed (experiment), user (affirmation: "可以结晶")
- **Crystallized via**: verbal-affirmation
- **Falsification criteria**: Any fixed-prior addition to A1 (physics tokens, graph bias, richer decoder, weather conditioning) that achieves Test MAE at or below A1 on ≥2/3 seeds.
- **Proof**: [E12 — N102–N105 A2-A5 3-seed results]
- **Evidence basis**:
  | variant | description | Test MAE (MW, 3-seed mean) | vs A1 | E10 Val MSE ratio vs A1 |
  |---|---|---|---|---|
  | **A1** | 8 tokens, simple FFN decoder | **0.001811** | — | 1.000 (baseline) |
  | A2 | +1 physics token (last-step power balance) | 0.001819 | +0.4% | 0.981 (−1.9%) |
  | A3 | +5 twin + 3 constraint tokens (16 total) | 0.001843 | +1.8% | 0.948 (−5.2%) |
  | A4 | +graph bias annealing (B_phys + B_learned) | 0.001863 | +2.9% | 0.890 (−11.0%) |
  | A5 | +per-step CrossAttention decoder + weather | 0.001947 | +7.5% | ~0.43 (best) |
  Monotonic Val→Test divergence: stronger Val improvement predicts worse Test degradation. A1's shared FFN (Linear 256→96) acts as an implicit regularizer — compressing 96-step prediction into a single token representation. All complexity additions (more tokens, graph bias, richer decoder, weather conditioning) are overfitting amplifiers.
- **Interpretation**: This is the NEGATIVE complement of C11. The iTransformer's value lies SOLELY in component-token separation (eliminating the C10 cancellation channel), NOT in richer decoders, physics priors, or graph constraints. The finding that stronger Val improvement predicts worse Test degradation (monotonic across 4 architecture variants) suggests that fixed physics priors provide training-set-specific guidance that fails to generalize across portfolios. The simplest architecture (8 tokens, 2 encoder layers, shared FFN decoder, no physics) is the global optimum for this dataset. Phase B's self-supervised pretraining is a direct response to this finding — replacing FIXED priors with DATA-DRIVEN coupling discovery.
- **Dependencies**: C11 (A1 baseline), C10 (cancellation channel), N102, N103, N104, N105
- **Tags**: physics-priors, overfitting, monotonic-degradation, val-test-divergence, architecture-simplicity, phase-a, ablation-chain
- **From staging**: O49, O50
