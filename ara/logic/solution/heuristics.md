# Heuristics

## H01: Default to identity (non-gated) residual connections for physics-guided learning
- **Rationale**: Identity shortcuts provide unimpeded gradient flow from component loss to theory branches. Sigmoid gates (V4.1) create gradient bottlenecks — dampening one pathway affects all theory branches through the shared encoder. This mirrors ResNet's finding: identity shortcuts are sufficient for residual learning.
- **Provenance**: user
- **Sensitivity**: high — gating causes theory MAE regression (V4.1: +30% vs V4)
- **Bounds**: Applies whenever multiple theory branches share an encoder. If each theory branch had its own encoder (not the case in PhysFormer), gating might be safe but unnecessary.
- **Code ref**: [physformer/models/physformer.py](../../physformer/models/physformer.py) — residual connection (no gate)
- **Source**: V4.1 dead end (N06 in exploration tree)

## H02: Use MAE (L1) not MSE (L2) for component supervision loss
- **Rationale**: Component errors span 2 orders of magnitude (Load ~15 kW, Wind ~0.8 kW). MSE would weight Load errors 324× (18²) more than Wind errors, causing Load to dominate component gradient. MAE treats all kW errors linearly, giving each component proportionate influence.
- **Provenance**: user
- **Sensitivity**: medium — switching to MSE would shift optimization toward Load at the expense of other components
- **Bounds**: Applies when component error scales differ by ≥5×. If all components have similar error magnitudes, MSE is acceptable.
- **Code ref**: [physformer/utils/losses.py](../../physformer/utils/losses.py)
- **Source**: V4 design decision (N16 in exploration tree)

## H03: Use modest component loss weight (0.03–0.05) to stay near the Pareto-optimal knee
- **Rationale**: Component loss creates a physics-accuracy trade-off. Weight ≤0.03 gives too little physics signal; weight ≥0.10 pushes past the Pareto knee into aggregate degradation. The sweet spot depends on architecture — V5's per-component residual multiplies effective supervision, requiring lower nominal weight than V4's scalar residual.
- **Provenance**: user
- **Sensitivity**: high — V5 at weight 0.1 shows +7.5% aggregate MAE degradation
- **Bounds**: V4 scalar residual: 0.05 was near the early Pareto knee. Later per-component residual variants favor moderate schedules around 0.02-0.03 for physics/aggregate balance; 0.01 can improve aggregate through error cancellation but degrades component physics, and 0.10 over-emphasizes component consistency.
- **Code ref**: [physformer/utils/losses.py](../../physformer/utils/losses.py) — component_loss_weight
- **Source**: V4→V4.2→V5 empirical comparison

## H04: Use moderate component-loss curriculum for robustness, not as proven phase refinement
- **Rationale**: C05 supports curriculum over tested fixed-weight baselines, but the best checkpoints occur around epochs 4-9, near or before the Phase1→Phase2 transition. The supported mechanism is robustness to moderate component-loss weights early in training, not a demonstrated "Phase 1 physics warmup enables Phase 2 refinement" path. V5 Phase 3 (pure net_mse fine-tune) produced no validation improvement.
- **Provenance**: user
- **Sensitivity**: low — Phase 3 is simply unnecessary, not harmful
- **Bounds**: Two-phase schedules remain the supported default. Prior C23 "3-stage" conclusions were invalidated by the config_to_args propagation bug (N64); true 3-stage still requires NVIDIA validation after the fix.
- **Code ref**: [physformer/utils/losses.py](../../physformer/utils/losses.py) — curriculum schedule
- **Source**: V5 curriculum dead end (N12), C05 fixed-weight ablations (N41/N42), narrative correction (N47), config propagation bug (N64)

## H05: Per-component residual heads should use low initialization std (0.01–0.05)
- **Rationale**: Residual corrections should start small — the theory branches provide the initial forecast, and residuals should learn to correct only where needed. Too-large initialization causes early training instability.
- **Provenance**: user
- **Sensitivity**: medium — V5 used std=0.01; V5.5 tests std=0.05. Too small (0.001) slows residual learning; too large (0.1) may cause early divergence.
- **Bounds**: Empirical. The optimal value depends on the scale of theory branch outputs.
- **Code ref**: [physformer/models/physformer.py](../../physformer/models/physformer.py) — residual head init
- **Source**: V5 design, V5.5 tuning plan

## H06: Time-condition the residual decoder with calendar features
- **Rationale**: Residual corrections are time-dependent — Load residual differs by hour and weekday; PV residual differs by season and time of day. Adding `time_proj(y_mark)` to the temporal decoder gives residuals access to this structure without relying on the encoder to pass it through the bottleneck.
- **Provenance**: user
- **Sensitivity**: medium — without time conditioning, residuals must learn temporal patterns indirectly through encoder representations, which is slower and less reliable.
- **Bounds**: Applies to VPP forecasting where load patterns are strongly calendar-driven. May be less important for pure generation forecasting.
- **Code ref**: [physformer/models/temporal_decoder.py](../../physformer/models/temporal_decoder.py)
- **Source**: V5 TemporalDecoder design (N10 in exploration tree)

## H07: Remove shared input features to enforce physics-data boundary (input-side isolation)
- **Rationale**: When a feature (e.g., temperature) flows to both a physical module (Temp MLP) and a data-driven module (GRU), the data module can "steal" the physical response function, duplicating the physics module's computation and degrading component accuracy. Removing the shared feature from the data module's input forces the physical module to be the sole pathway for that physical quantity, sharpening the physics-data boundary.
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Sensitivity**: medium — removing temp from GRU improves aggregate MAE (0.001999 vs 0.002017 baseline) but causes earlier overfitting (Val MSE peak at ep12 vs ep26), suggesting the shared feature also acts as an implicit regularizer.
- **Bounds**: Applies when a physical module (physics equation or constrained MLP) and a data-driven module (GRU, TCN) share input features. Remove the feature from the data module only if the physical module can capture its effect. The removed feature may have a secondary regularization role — monitor generalization gap.
- **Code ref**: [physformer/models/physformer/physical_layer.py](../../physformer/models/physformer/physical_layer.py) — LoadTemporalModule.use_temp_input
- **Source**: V6.1 N29 (no_temp) experiment (N33), crystallized from O09
- **From staging**: O09

## H08: Val MSE (normalized) and Test MSE (MW²) differ by target_std² — always compare in same units
- **Rationale**: Val MSE is computed in z-scored normalized space (`F.mse_loss(pred_norm, target_norm)`), while Test MSE is computed after denormalization in real MW². The conversion factor is `target_std² ≈ 2.0×10⁻⁵` (target_std ≈ 0.0045 MW = 4.5 kW). So Val MSE 0.38 corresponds to Test MSE ~7.6×10⁻⁶. Any Val/Test comparison must first multiply Val MSE by target_std². The "d512 Val breakthrough" narrative (N57) was an artifact of comparing different units — once converted, d512 shows perfect Val/Test consistency, ruling out overfitting.
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Sensitivity**: high — incorrect cross-scale comparison led to false "overfitting" narrative in N57 (d512) and misleading N44 encoder bottleneck interpretation
- **Code ref**: [physformer/train/physformer_exp.py:202,412-418](../../physformer/train/physformer_exp.py), [scripts/check_val_test_scale.py](../../../scripts/check_val_test_scale.py), [scripts/v3_verify_d512.py](../../../scripts/v3_verify_d512.py)
- **From staging**: O22

## H09: Push tools that mirror project trees must default-deny ML output directories
- **Rationale**: When a tool tar+scp's a project directory to remote, any local `runs/`, `checkpoints/`, `results/`, `tensorboard/`, or `wandb/` dirs that exist locally (because of a previous pull) will be round-tripped to remote — overwriting whatever was there. If the trainer auto-resumes from a checkpoint file in the run dir (PhysFormer does: `Path(run_dir)/training_state.pth` exists check at physformer/train/physformer_exp.py:272), the freshly launched training will silently resume from old state instead of starting fresh, invalidating multi-seed reproducibility. The failure mode is invisible at launch time (PIDs spawn normally) and only surfaces when the first epoch logs `Resumed from checkpoint | start_epoch=N`.
- **Provenance**: ai-suggested (diagnosis after Batch 1 launch incident N78)
- **Crystallized via**: empirical-resolution
- **Sensitivity**: high — single missing exclude entry silently invalidates an entire experiment batch
- **Bounds**: Any sync tool with whitelist-style includes/excludes. Tools using opt-in "what to ship" lists (e.g., explicit file list, git-tracked-only) do not have this issue. Trainer auto-resume on ckpt presence is a separate latent footgun: a defensive trainer would either require explicit `--resume` flag or fail-loud on found-ckpt-but-not-requested.
- **Code ref**: `~/.claude/skills/autodl-push/config.yaml` (added `runs` to exclude list after N78 incident). Future generalisation: also add `checkpoints`, `tensorboard`, `wandb`, `mlruns` to default-exclude when seeding a new project.
- **Source**: T1 Batch 1 ckpt-pollution incident (N77, N78)
- **From staging**: ai-suggested directly; no prior staged observation

## H10: SSH remote daemons must use one-shot checker + local polling, not `cmd &`
- **Rationale**: Remote background processes launched via `ssh ... cmd &` receive SIGHUP when the SSH session ends and die silently. The correct pattern: write a one-shot checker script on the remote, poll it from local via Monitor's while+sleep loop. Sub-rules: (1) Windows local operations should use PowerShell, not Bash — each Bash invocation creates a bash.exe process that may linger. (2) Monitor while/until loop conditions must be dry-run tested first (the first Monitor attempt used `until ... grep -E "epoch|ALL_DONE"` which exited immediately because grep matched "epoch" on the first iteration).
- **Provenance**: ai-executed
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: memory: feedback_shell_discipline.md, feedback_remote_daemon.md
- **From staging**: O44

## H11: Batch multi-variate tokenization yields super-linear throughput gains on cuDNN RNNs
- **Rationale**: When tokenizing multiple variates (e.g., 5 component histories) with per-variate GRU encoders, batching into a single (B*C, T, 1) GRU call reduces cuDNN kernel launch overhead by C× (5× for PhysFormer). Paired with batch_size doubling (128→256), the combined effect yields 9.3× throughput improvement (75→700+ S/s). GPU utilization improved from 17% to 25%. The key insight: cuDNN GRU benefits disproportionately from batched calls — kernel launch overhead dominates small-batch GRU latency. This pattern generalizes to any architecture that independently encodes multiple variates from separate time series (multi-asset forecasting, multi-sensor fusion, multi-channel biological signals).
- **Provenance**: ai-executed
- **Crystallized via**: verbal-affirmation
- **Sensitivity**: medium — the speedup magnitude depends on the number of variates and batch size, but the pattern (batch → reduce kernel launches → super-linear speedup) is general
- **Bounds**: Applies when multiple RNN/GRU/LSTM encoders process independent variates with the same sequence length. Requires that all variates share the same input dimension — padding or separate embedders needed if dims differ. Not compatible with per-variate custom preprocessing (use serial path for those).
- **Code ref**: [physformer/models/physformer/igt_model.py:BatchedComponentEmbedding](../../physformer/models/physformer/igt_model.py), [physformer/models/physformer/igt_model.py:BatchedWeatherEmbedding](../../physformer/models/physformer/igt_model.py)
- **From staging**: O48
