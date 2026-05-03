# Peer Review Consensus Report

- File: `V4.1 regression — Theory MAE degradation root cause`
- Reviewers: **claude** vs **codex**
- Matching rule: loose (v2: category equivalents + range overlap), tolerance=2

## Summary
| Bucket | Count |
|---|---:|
| Consensus (both flagged) | 0 |
| Only claude | 5 |
| Only codex | 13 |
| **Total unique** | **18** |

`claude` reported 5 raw issues; `codex` reported 13 raw issues.

## 1. Consensus Issues (0)
_Both reviewers independently flagged the same location + (compatible) category._

_(none)_

## 2. Only from claude (5)

- **V4.1 — Gradient coupling through shared encoder (most likely root cause)** `[P1/architecture]` Theory MAE 退化 30% 的最可能根因是共享编码器的梯度耦合。数据流：Encoder → TemporalDecoder → WeatherFusion → FiLM → UnifiedResidualHead。theory_net 的计算路径是：PhysicalLayer(weather, calendar, battery) → theory_net。但理论层的 weather features 来自 Encoder 输出（经过 WeatherFusion 的 latent），而 Encoder 的梯度主要来自 residual head 的 loss。当 sigmoid(0)=0.5 将 residual 梯度减半，Encoder 的参数更新方向和幅度发生了改变。这间接影响了下游 WeatherFusion 产生的 weather_latent 质量，进而影响 PhysicalLayer 需要的 weather features 表示。关键证据：V4.1 的 Residual std 也退化（4.31→5.08），说明残差本身的修正能力也减弱了——如果没有 gate，residual 至少应该保持相同的修正精度。这表明问题是系统性的（Encoder 表示质量下降），而非仅 residual 被限制。
  - Fix: `(1) 将 residual 的 gate 移除，改用不改变梯度的替代方案（如仅在推理时做 threshold）；(2) 或者在 encoder 和 physical layer 之间断开梯度——PhysicalLayer 的参数梯度不应依赖 residual path。具体：将 weather_latent 的 detach 版本传给 PhysicalLayer，使 theory_net 的训练完全独立于 residual head。`
  - Why: 当前架构中 theory_net 和 residual_net 共享 Encoder + TemporalDecoder + WeatherFusion。任何对 residual path 的修改都会通过共享参数反向传播到 theory 的输入表示层。这不是 architecture bug，而是激励结构问题——residual 的训练目标（net MSE）与 theory 的训练目标（component MAE）可能冲突。

- **V4.1 — TemporalDecoder time_proj 可能是有害引入** `[P1/training_dynamics]` time_proj 以 Xavier(gain=0.1) 初始化——输出幅度约为 query_pos 的 10%。但从 V4 checkpoint 恢复训练时，V4 的 query_pos 已经过 11 epoch 训练，time_proj 的随机初始化可能破坏了 query 分布。具体机制：query = query_pos + time_proj(y_mark)。如果 time_proj 的输出与 query_pos 的分布不一致（均值偏移、方差不同），cross-attention 的 attention pattern 会偏移，影响整个 decoder 输出。这会影响 weather_latent → PhysicalLayer 的输入表示。V3/V4 中 decoder 没有 time_proj，解码器 query_pos 已经学会了最优的 attention pattern。添加 time_proj 破坏了这一模式。
  - Fix: `(1) 短期：如果 time_proj 对性能有害，将其移除——Phase 2 重新引入时从头训练；(2) 中长期：如果保留 time_proj，必须从零训练而非从 V4 checkpoint 恢复，让 query_pos 和 time_proj 共同学习。`
  - Why: 从预训练 checkpoint 添加新的前向路径（time_proj → query → cross-attention → decoder output）时，下游的所有权重（query_pos, cross_attn, FFN）都针对旧 query 分布优化过。注入新信息源可能导致 attention 失效。

- **V4.1 — Loss 变更与 checkpoint 恢复的不兼容** `[P2/training_dynamics]` V4.1 从 V4 checkpoint 恢复模型权重，但使用了不同的 loss 函数（归一化 component MAE vs MW 空间 MAE）。V4 的 theory_net 参数经过了 11 epoch 的特定 loss landscape 优化。新 loss 改变了每个 theory branch（load/pv/wind/battery）的相对梯度权重。尤其是 battery_power_mae 在归一化后与其他 component MAE 量级相当（之前 battery 在 MW 空间可能较大），改变了理论分支之间的梯度平衡。从 V4 checkpoint 开始，optimizer 虽已重置，但模型权重初始位置在 V4 的 local minimum——这个 minimum 在新 loss 下可能不是好起点。
  - Fix: `从零训练（非 V4 checkpoint）来隔离 loss 变更的影响。或者保持 V4 的 MW 空间 component loss，仅降低 weight。`
  - Why: Transfer learning 的前提是目标函数不变。当 loss 函数改变时，旧的最优权重在新 loss 下可能处于 sharp minimum，optimizer 的第一步就会跳到一个更差的 basin。

- **V4.1 vs V4 — Val MSE 改善但 Test 退化暗示过拟合** `[P2/training_dynamics]` Val MSE 0.381→0.379 (-0.5%) 但 Test MSE 7.35e-6→7.68e-6 (+4.5%)。Val 与 Test 方向相反——这是典型的过拟合信号。可能原因：(a) early_stop_start=20 使得 model selection 完全基于 Val 表现，而 V4 使用 epoch 10 开始的计数；(b) 新增参数（time_proj, alpha）增加了模型容量，可能在 Val 上过拟合；(c) component loss 的归一化可能在 Val 分布上有更好的表现但泛化到 Test 时变差。
  - Fix: `对比 V4 和 V4.1 的 per-component test metrics 以识别哪个分量的泛化差距最大。如果 load 分量的 Val-Test 差距最大，说明 calendar embedding + component loss 对 load 的训练集特定模式过拟合。`
  - Why: 论文中 Val→Test 的一致性需要被验证。如果 Val 的改善不能转化为 Test 改善，实验设置需要修正。

- **V4.1 — 隔离实验的建议** `[P1/missing_mechanism]` 当前有 4 个同时变更（gate + loss 归一化 + time_proj + schedule），无法确定哪个导致 Theory MAE 退化。需要消融实验：(A) V4 baseline + 仅改 schedule（patience=20, early_stop=20），其他不变；(B) V4 baseline + 仅改 loss（归一化 component MAE）；(C) V4 baseline + 仅加 time_proj（从头训练）。对比三者的 Theory MAE 变化即可定位根因。
  - Fix: `快速消融策略：每个 variant 只跑 15 epoch（~2.5h），比较 epoch 15 的 Val MSE 和 Theory MAE。最可能的结果：time_proj 是主因（query distribution shift），loss 归一化是次因（梯度平衡变化），gate 是第三因（间接梯度效应）。`
  - Why: 同时变更多个变量使得回归无法归因。消融是唯一可靠的诊断手段。

## 3. Only from codex (13)

- **physformer/models/temporal_decoder.py:18-42** `[medium/architecture]` Temporal decoding uses independent learned future queries plus an ungated time projection, with no future-step self-attention or multi-scale temporal structure.
  - Original: `self.query_pos = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
self.time_proj = nn.Linear(time_enc_in, d_model)
...
query = self.query_pos.expand(B, -1, -1)
if y_mark is not None:
    query = query + self.time_proj(y_mark)
attn_out, _ = self.cross_attn(query, memory, memory, need_weights=False)`
  - Fix: `Normalize y_mark before projection, gate or residual-scale time_proj, add decoder self-attention across future steps, and consider multi-resolution or patch-based horizon embeddings for daily cycles and ramp events.`
  - Why: VPP trajectories have strong intra-horizon dependencies. Independent queries can make each future step attend to history separately without explicitly modeling ramp continuity, peak timing, or correlated forecast errors.

- **physformer/models/physical_layer.py:53-64; physformer/exp/exp_physformer.py:97-105** `[medium/physics_fidelity]` Battery numerical integration uses a fixed 0.25 hour step and the experiment hard-codes the same value in the loss.
  - Original: `dt_hours=0.25,
...
self.dt_hours = dt_hours
...
dt_hours=0.25,`
  - Fix: `Derive dt_hours from the dataset frequency or actual timestamp deltas, pass it through config/data metadata into both ExplicitVPPPhysicalLayer and PhysAwareBaseLoss, and support variable dt for missing or irregular samples.`
  - Why: SOC integration is dimensionally tied to the time step. A fixed 15-minute assumption silently breaks when the data frequency, forecast horizon, daylight-saving alignment, or resampling strategy changes.

- **configs/drivers/train_three_stage.yaml:3-45; physformer/runner/config.py:54-72; physformer/runner/drivers.py:21-80; physformer/exp/exp_physformer.py:89-93** `[high/implementation]` The declared three-stage curriculum is not fully wired: keys such as training_mode, freeze_backbone, use_aux_supervision, theory_aux_weight, phys_layer_lr_scale, aux_weight_stageA, and init_from_run are not applied by the experiment training logic.
  - Original: `TRAINING_KEYS = (
    "batch_size", "train_epochs", "learning_rate", "weight_decay", "grad_clip",
    "patience", "use_amp", "seed", "log_interval", "warmup_epochs",
    "warmup_start_factor", "early_stop_metric", "early_stop_start_epoch",
    "soc_weight", "component_loss_weight", "restart_t0", "restart_t_mult",
)
...
return optim.AdamW(self.trainable_parameters, lr=self.args.learning_rate, weight_decay=wd)`
  - Fix: `Add the stage-specific keys to config parsing and driver overrides; load init_from_run checkpoints before training; call freeze_for_physics_warmup or freeze_backbone as configured; create optimizer param groups with phys_layer_lr_scale; map stage aux/SOC weights into PhysLoss.`
  - Why: A curriculum only helps if it changes optimization. As written, the YAML suggests staged physics warmup and operational fine-tuning, but the code mostly trains a single objective with one optimizer group, making regression attribution unreliable.

- **physformer/models/physformer.py:77-83; physformer/layers/revin.py:11-77** `[medium/missing_mechanism]` RevIN is implemented but not used by PhysFormer; the model relies on fixed global scaler buffers only.
  - Original: `self.register_buffer("target_mean", self._to_buffer(target_mean, 1, 0.0))
self.register_buffer("target_std", self._to_buffer(target_std, 1, 1.0))
self.register_buffer("aux_mean", self._to_buffer(aux_mean, 5, 0.0))
self.register_buffer("aux_std", self._to_buffer(aux_std, 5, 1.0))`
  - Fix: `Add optional RevIN or adaptive instance normalization on the neural history/residual path, while keeping the explicit physics layer in calibrated real units. Denormalize the residual/output consistently and run per-portfolio ablations.`
  - Why: VPP load, PV, wind, and battery distributions shift by season, weather regime, portfolio size, and asset mix. RevIN-style stationarization is a standard time-series mechanism for this non-stationarity.

- **physformer/models/physformer.py:90-112; configs/physformer_default.yaml:34-35** `[medium/scalability]` The default architecture uses full self-attention over 672 history steps and cross-attention from 96 future queries, with distillation disabled.
  - Original: `attn_cls = ProbAttention if attn == "prob" else FullAttention
self.encoder = Encoder(... use_distillation=distil ...)
...
self.temporal_decoder = TemporalDecoder(seq_len=seq_len, pred_len=pred_len, d_model=d_model, ...)`
  - Fix: `Use patching or multi-resolution encoders, ProbAttention/linear attention for long histories, memory compression, and horizon chunking. Benchmark memory and latency for seq_len > 672 and pred_len > 96.`
  - Why: Full attention scales quadratically with history length and cross-attention scales with pred_len times seq_len. Longer lookbacks, sub-5-minute data, or multi-day horizons will degrade speed and memory before model quality can be evaluated.

- **physformer/models/conditioning.py:94-109** `[medium/missing_mechanism]` The output head is deterministic and emits only one residual value per horizon step; there is no uncertainty quantification or scenario consistency.
  - Original: `self.net = nn.Sequential(
    nn.Linear(d_model + theory_proj_dim, d_model),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_model, 1),
)`
  - Fix: `Add quantile, Gaussian/mixture, or distributional heads trained with pinball loss, CRPS, or NLL. For operations, generate temporally correlated scenarios and constrain them through the same physics/feasibility layer.`
  - Why: VPP scheduling and market bids require prediction intervals and risk-aware scenarios, especially under renewable and load uncertainty. Point forecasts hide tail risk and cannot support reserve or dispatch decisions.

- **physformer/models/conditioning.py:102-109; physformer/models/physformer.py:251-252** `[high/architecture]` Residual correction is controlled by one global scalar gate, so all horizons, portfolios, seasons, and operating regimes get the same residual capacity.
  - Original: `self.alpha = nn.Parameter(torch.tensor(0.0))
...
gate = torch.sigmoid(self.alpha)
return gate * residual`
  - Fix: `Replace the scalar alpha with a context-dependent confidence gate g_t = f(theory_net, battery_feats, weather_latent, horizon, portfolio_id), initialized near 1.0 or with a staged schedule. Add diagnostics for gate by horizon and operating regime.`
  - Why: VPP forecast errors are regime-dependent: PV ramps, wind cut-in/cut-out, battery saturation, and load peaks need different reliance on physics versus learned residuals. A scalar gate also halves residual-path gradients at initialization, which can underfit corrections and destabilize V4 to V4.1 comparisons.

- **physformer/models/physical_layer.py:133-158** `[medium/scalability]` Per-portfolio adaptation is an embedding table that cannot generalize to unseen portfolios or changing asset compositions.
  - Original: `self.portfolio_embed = nn.Embedding(self.num_portfolios, per_portfolio_dim)
self.portfolio_delta = nn.Linear(per_portfolio_dim, 16)
...
emb = self.portfolio_embed(portfolio_ids)
return self.portfolio_delta(emb)`
  - Fix: `Condition physics parameters on portfolio metadata and asset-level descriptors such as capacities, technology mix, region, inverter/turbine types, and battery ratings. Use hierarchical asset encoders or graph/set encoders for variable-size portfolios.`
  - Why: VPP fleets evolve. An ID lookup memorizes training portfolios and does not transfer to new aggregations, asset additions, retirements, or regional changes.

- **physformer/models/physical_layer.py:178-207; physformer/models/physical_layer.py:284-286** `[medium/physics_fidelity]` The load branch is a static temperature/calendar formula plus a recent net-injection average; historical weather is denormalized but unused, and there is no thermal state-space memory.
  - Original: `recent_net_avg = x_net_hist_real[:, -24:, :].mean(dim=1, keepdim=True)
autoreg_correction = self.load_autoreg_proj(recent_net_avg)
...
load_theory = F.softplus(load_pre)
...
hist_weather_real = self._denorm_weather(x_weather_hist)`
  - Fix: `Replace the static load prior with a differentiable RC/thermal state model or at least lagged temperature and load-state features. Use decomposed load history when available, or estimate latent load separately instead of using net injection as a proxy.`
  - Why: Net injection confounds load with PV, wind, and battery behavior. Without thermal inertia or lagged weather, the load prior cannot represent HVAC dynamics, occupancy effects, or delayed response during heat/cold events.

- **physformer/utils/losses.py:79-83; physformer/utils/losses.py:186-198; physformer/exp/exp_physformer.py:296-298** `[high/training_dynamics]` The training objective uses fixed scalar weights and early-stops on net MSE, while theory quality is only diagnostic and physical validation metrics are not used for checkpoint selection.
  - Original: `total_loss = terms["net_mse"]
...
total_loss = total_loss + self.component_loss_weight * (
    terms["component_load_mae"]
    + terms["component_pv_mae"]
    + terms["component_wind_mae"]
    + terms["battery_power_mae"]
)
...
stop_value = vali_stats["net_mse"] if early_stop_metric == "net_mse" else vali_stats["loss"]`
  - Fix: `Use dynamic multi-objective balancing such as uncertainty weighting, GradNorm, or PCGrad; include ramp, SOC pre-clip violation, component error, and theory_net error in a composite validation metric; log gradient norms per branch.`
  - Why: Fixed weights make the physics branches compete unpredictably with net MSE. This can explain theory MAE regressions even when validation MSE improves, because model selection ignores whether the physical prior remains calibrated.

- **physformer/models/physical_layer.py:245-264; physformer/utils/losses.py:91-95** `[high/physics_fidelity]` Battery SOC is hard-clipped inside the forward recurrence, so the SOC bounds loss observes post-clipped states and cannot teach the model to avoid infeasible pre-clipped dispatch.
  - Original: `soc_next = soc_prev + (eta_charge * charge - discharge / eta_discharge) * self.dt_hours
soc_next = torch.minimum(torch.maximum(soc_next, torch.zeros_like(soc_next)), capacity)
...
soc_bounds_loss = F.relu(-soc).mean() + F.relu(soc - capacity_real).mean()`
  - Fix: `Track pre_clip_soc and penalize violations before projection; use differentiable projection or constrained action parameterization based on current SOC headroom; add terminal SOC, battery ramp, degradation, and reserve constraints.`
  - Why: The current structure can report zero SOC violations while the neural dispatch repeatedly asks for impossible charge/discharge actions. This weakens physical fidelity and hides battery errors that directly affect net injection.

- **physformer/models/physformer.py:247-252** `[high/physics_fidelity]` The final net injection is an unconstrained additive residual over theory_net, with no feasibility projection for aggregate capacity, ramp limits, or component balance.
  - Original: `conditioned = self.physics_film(weather_latent, physics_features)
residual = self.unified_head(conditioned, theory_net)
pred_net = theory_net + residual`
  - Fix: `Add a differentiable feasibility layer or constrained residual parameterization: bound residual by available upward/downward flexibility, enforce aggregate generation/load/battery limits, and add ramp/energy penalties to training rather than only reporting ramp metrics at test time.`
  - Why: A learned residual can cancel the physical prior and produce operationally impossible VPP injections. For dispatch or market use, net forecasts must respect asset capacities, battery energy, and ramp constraints, not only minimize point MSE.

- **physformer/models/physical_layer.py:291-312; physformer/models/physical_layer.py:327-335** `[medium/physics_fidelity]` PV and wind physics are first-order scalar curves with no solar geometry, inverter clipping, curtailment, air-density correction, hub-height transform, wake effects, or turbine fleet heterogeneity.
  - Original: `solar_energy = (weather_real[..., 1:2].clamp_min(0.0) / 3600000.0).clamp_min(0.0)
pv_temp_factor = (1.0 - pv_temp_coeff * 0.01 * (temp - 25.0)).clamp(0.5, 1.5)
pv_theory = (pv_scale + F.softplus(pv_cap)) * solar_energy * pv_temp_factor
...
wind_curve = plateau + (1.0 - plateau) * rising_curve
wind_theory = (w_scale + F.softplus(w_cap)) * wind_curve * running_mask`
  - Fix: `Add solar-position/clear-sky features, inverter AC clipping, availability/curtailment latent states, hub-height wind conversion, air-density correction, and mixture-of-curves or asset-cluster parameters for heterogeneous portfolios.`
  - Why: VPP renewable output is often limited by clipping, curtailment, weather forecast bias, and heterogeneous asset fleets. Scalar curves may fit average behavior but fail during ramps and extremes.
