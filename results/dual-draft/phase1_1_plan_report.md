# Peer Review Consensus Report

- File: `Phase 1.1 fix plan — V4 critical fixes`
- Reviewers: **claude** vs **codex**
- Matching rule: loose (v2: category equivalents + range overlap), tolerance=2

## Summary
| Bucket | Count |
|---|---:|
| Consensus (both flagged) | 0 |
| Only claude | 5 |
| Only codex | 11 |
| **Total unique** | **16** |

`claude` reported 5 raw issues; `codex` reported 11 raw issues.

## 1. Consensus Issues (0)
_Both reviewers independently flagged the same location + (compatible) category._

_(none)_

## 2. Only from claude (5)

- **Fix 1 — Loss normalization approach** `[P2/implementation]` Fix 1 方案（用 aux_std 归一化理论分量）是正确的方向，但有一个微妙问题：`component_theory_real` 和 `y_aux` 的归一化统计量来自不同的 scaler——component_theory 使用的是 target scaler（通过 `_denorm_target` 得到），而 y_aux 使用的是 aux scaler。这两个 scaler 各自由 StandardScaler 独立拟合。如果 component_theory_real 的分布与 y_aux_real 的分布差异很大（例如 Load 的均值远大于 net injection），即使归一化后它们在同一空间，MAE 的量级仍可能不同。建议：在两个 scaler 空间都做归一化验证——确保 aux_std 在 Load 维度上的值与 net injection 的 target_std 在同一数量级。
  - Fix: `在 compute_terms() 中添加 `component_load_mae_normalized` 到 debug dict 用于运行时监控；首次训练后检查归一化后的 MAE 是否在 0.05-2 范围（与 net_mse ~0.4 可比）。`
  - Why: 归一化消除单位不匹配，但不保证量级匹配。如果某个 aux_std 极小（如 battery_soc 的 std ~ 0.01），归一化后 MAE 会爆炸。

- **Fix 2 — TemporalDecoder time projection initialization** `[P3/implementation]` `time_proj` 是一个新的 nn.Linear(10, d_model) 层，默认使用 Kaiming 初始化。`query_pos` 是一个 nn.Parameter 从 N(0,1) 初始化。两者相加后，如果 time_proj 的初始输出幅度与 query_pos 不同，可能会导致 cross-attention 的 query 分布偏移。在训练的 epoch 1，解码器输出会有短暂的退化。
  - Fix: `对 time_proj 使用较小的初始化（例如 `nn.init.xavier_uniform_(self.time_proj.weight, gain=0.1)`）或使用零初始化使 y_mark 的影响在训练初期为中性。`
  - Why: 从 V4 checkpoint 恢复训练时（V4 的解码器 query_pos 已训练过），新加的 time_proj 如果随机初始化可能破坏已有的解码器行为。建议从零初始化开始逐步学习时间模式。

- **Fix 3 — early_stop_start_epoch=20 可能过激** `[P2/training_dynamics]` V4 的最优在 epoch 11。如果 early_stop_start_epoch=20，意味着 epoch 20 之前即使 9 个 epoch 无改善也不会触发早停计数。这确实给了 WarmRestarts 更多空间。但 V4 的 epoch 12-20 全部无改善（counter 从 1 数到了 9 但未触发因为 start_epoch=10），实际在 epoch 20 WarmRestart 时 counter=9/25。提高到 20 后 epoch 20 的 restart 仍有 35-(20-11)=26 个 epoch 的空间——这应该足够了。
  - Fix: `保持 early_stop_start_epoch=20。额外建议：在论文中报告 best epoch=11, early_stop_start=20, 总 epoch=100, patience=35。`
  - Why: 实验设计需要可复现。固定 schedule 参数确保论文结果的透明性。

- **Fix 2 — TemporalDecoder 是否需要 time_enc_in 默认值** `[P3/implementation]` TemporalDecoder.__init__ 新增 `time_enc_in` 参数，默认值应为 10（当前 time_feat_dim）。但如果有其他调用者（如 baseline 模型）使用 TemporalDecoder 且不传此参数，默认 10 会导致 shape 不匹配（如果他们的 time_features 输出不是 10 维）。不过当前代码中 TemporalDecoder 仅在 PhysFormer 中使用。
  - Fix: `设置默认值 `time_enc_in=10`，但在 baseline 模型中也更新 TemporalDecoder 构造。或设为 `time_enc_in=None` 并在 forward 中检查——如果 time_proj 不存在，跳过 y_mark 注入（保持向后兼容）。`
  - Why: 向后兼容性。当前看似不必要但能防止将来的事故。

- **Fix 3 — 是否需要从 V4 checkpoint 恢复训练** `[P1/training_dynamics]` Loss 归一化改变了 total_loss 的组成——component_loss_weight 从 0.05（MW 空间）变为 0.02（归一化空间），component MAE 的量级也变了。如果从 V4 checkpoint 恢复，optimizer 和 scheduler 状态来自旧的 loss landscape。新的 loss landscape 可能与旧的不连续，导致 epoch 12 出现 loss 突变。
  - Fix: `两种方案：(A) 从 V4 epoch 11 checkpoint 恢复模型权重但重置 optimizer 和 scheduler；(B) 从头训练（清空 checkpoint，让新 loss 自然收敛）。推荐方案 A：保留已训练的物理参数，仅重置优化器状态。在 exp_physformer 中添加 `--reset-optimizer` CLI 选项。`
  - Why: Loss 函数变更后继续使用旧 optimizer momentum/velocity 状态（如 Adam 的 m/v）会导致更新方向错误。重置 optimizer 是必须的。

## 3. Only from codex (11)

- **configs/drivers/train_three_stage.yaml:13-28** `[high/implementation]` The three-stage driver contains stale curriculum keys that are not consumed by the current Exp_PhysFormer/PhysLoss path, so the intended staged optimization is partly inert.
  - Original: `- name: stageA_net_first
  ...
  phys_layer_lr_scale: 0.1
  aux_weight_stageA: 0.2
  soc_weight_stageA: 0.1
  overlap_weight_stageA: 0.01`
  - Fix: `Either remove stale keys or implement them end-to-end. Add optimizer parameter groups for phys_layer_lr_scale, map stage-specific weights to actual PhysLoss arguments, and implement init_from_run loading in Exp_PhysFormer instead of only setting args.init_from_run in the driver.`
  - Why: Curriculum learning and staged physics warmup only help if the code actually changes optimization behavior. Inert config values create misleading experiments and make convergence failures hard to diagnose.

- **physformer/models/temporal_decoder.py:15-39** `[high/architecture]` TemporalDecoder accepts y_mark but ignores it, so future horizon queries are only static learned positions and cannot adapt to calendar/weekend/holiday conditions.
  - Original: `def forward(self, memory, y_mark=None):
    B = memory.shape[0]
    query = self.query_pos.expand(B, -1, -1)  # (B, pred_len, d_model)
    attn_out, _ = self.cross_attn(query, memory, memory, need_weights=False)`
  - Fix: `Add time conditioning to the decoder: pass time_feat_dim into TemporalDecoder, define self.time_proj = nn.Linear(time_feat_dim, d_model), initialize it near zero, and use query = self.query_pos.expand(B, -1, -1) + self.time_proj(y_mark). For long horizons, add relative horizon embeddings and a small future self-attention/TCN block.`
  - Why: VPP net injection has strong calendar-dependent load, PV, and battery scheduling patterns. Static horizon queries force the model to learn one generic horizon template and weakens generalization to weekends, holidays, and seasonal transitions.

- **physformer/models/physformer.py:77-83** `[medium/missing_mechanism]` RevIN is implemented in physformer/layers/revin.py but PhysFormer uses only fixed dataset scaler buffers, leaving the neural stream exposed to non-stationarity across portfolios and seasons.
  - Original: `self.register_buffer("target_mean", self._to_buffer(target_mean, 1, 0.0))
self.register_buffer("target_std", self._to_buffer(target_std, 1, 1.0))
self.register_buffer("aux_mean", self._to_buffer(aux_mean, 5, 0.0))
self.register_buffer("aux_std", self._to_buffer(aux_std, 5, 1.0))`
  - Fix: `Add optional RevIN/adaptive normalization on the data-driven history stream and residual output path, while keeping the explicit physics layer in calibrated real units through dataset scalers. Evaluate per-portfolio RevIN ablations.`
  - Why: Load and renewable distributions shift by season, region, portfolio size, and penetration. RevIN-style stationarization is a standard forecasting mechanism for this type of non-stationarity.

- **physformer/models/conditioning.py:94-107** `[medium/missing_mechanism]` The model is deterministic and outputs only a scalar residual/net forecast; there is no uncertainty quantification for weather error, renewable variability, or dispatch risk.
  - Original: `self.net = nn.Sequential(
    nn.Linear(d_model + theory_proj_dim, d_model),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_model, 1),
)
...
residual = self.net(inp)`
  - Fix: `Add probabilistic heads: quantile regression, Gaussian/Student-t NLL with heteroscedastic scale, or conformal calibration on residuals. Condition predictive variance on weather forecast uncertainty and component-level physics features.`
  - Why: VPP operation needs risk-aware forecasts for reserves, bids, and grid constraint margins. Point forecasts alone cannot express tail risk during ramps, cloud events, or battery saturation.

- **physformer/utils/losses.py:103-199** `[high/training_dynamics]` The total loss mixes normalized net_mse with real-MW component MAEs under one fixed component_loss_weight, causing scale imbalance and unstable multi-objective tradeoffs.
  - Original: `y_aux_real = self.denorm_aux(y_aux)
component_load_mae = F.l1_loss(load_theory_comp, y_aux_real[..., 0:1])
component_pv_mae = F.l1_loss(pv_theory_comp, y_aux_real[..., 1:2])
...
total_loss = terms["net_mse"]
...
total_loss = total_loss + self.component_loss_weight * (...)`
  - Fix: `Compute component losses in normalized auxiliary space: component_norm = (component_theory_real - aux_mean) / aux_std and compare to raw y_aux. Add per-component or uncertainty-based weights, include battery SOC supervision when available, and log each weighted contribution.`
  - Why: If component losses are in MW while net_mse is normalized, the auxiliary objective can dominate or vanish depending on scaler values. This can improve theory MAE while degrading final net MAE.

- **physformer/models/physical_layer.py:185-206** `[high/physics_fidelity]` The load branch uses recent net injection as an autoregressive load proxy, which entangles load with PV, wind, and battery behavior.
  - Original: `recent_net_avg = x_net_hist_real[:, -24:, :].mean(dim=1, keepdim=True)
autoreg_correction = self.load_autoreg_proj(recent_net_avg)
...
load_theory = F.softplus(load_pre)`
  - Fix: `Use historical load auxiliary data when available, or reconstruct a load proxy from component histories instead of raw net. A better fix is a latent thermal/load state-space branch with calendar, temperature, and lagged load state, separate from renewable and battery components.`
  - Why: Low net injection during high PV periods can be misread as low load. This damages component identifiability and can make the physics layer learn compensating errors rather than real load response.

- **physformer/models/physformer.py:242-245** `[high/architecture]` The explicit physics layer computes load/PV/wind/battery components, but the neural path receives only scalar theory_net plus battery features, discarding component-level structure.
  - Original: `theory_net_real = physics_states["theory_net_real"]
battery_feats = physics_states["battery_feats_real"]
theory_net = self._norm_target(theory_net_real)
physics_features = torch.cat([theory_net, battery_feats], dim=-1)  # (B, P, 5)`
  - Fix: `Feed normalized component_theory_real components into PhysicsFiLM or a physics cross-attention block: [load, pv, wind, battery_power, soc, capacity, eta, theory_net]. Optionally predict residuals per component and reconstruct final net as load_res - pv_res - wind_res + batt_res.`
  - Why: Different physical states can produce the same net injection. Collapsing components hides whether the error comes from load, PV, wind, or battery dispatch, limiting residual correction and physical interpretability.

- **physformer/models/physical_layer.py:245-258** `[medium/physics_fidelity]` Battery dispatch is generated by an MLP from weather/calendar/SOC context, but it lacks VPP control drivers such as price, dispatch target, forecast net imbalance, grid limits, and terminal SOC objective.
  - Original: `step_context = torch.cat([
    base_context,
    weather_phys[:, step, :],
    y_mark[:, step, :],
    soc_prev / (capacity + 1e-6),
    (soc_prev - 0.5 * capacity) / (0.5 * capacity + 1e-6),
], dim=-1)
hidden = self.battery_step_proj(step_context)
power_raw = self.battery_power_head(hidden)`
  - Fix: `Condition battery policy on load/PV/wind theory, net forecast imbalance, market price or schedule if available, interconnection limit, and terminal SOC target. For stronger fidelity, replace the MLP with a differentiable MPC/optimization layer or imitation policy trained from dispatch traces.`
  - Why: A VPP battery is a controlled asset, not a weather-driven generator. Without operational objectives, learned battery power may fit correlations but fail under new dispatch regimes or high-renewable conditions.

- **physformer/models/physformer.py:247-252** `[high/physics_fidelity]` The final residual is unconstrained and can overwrite the physical prior, so physics is conditioning rather than an enforced consistency mechanism.
  - Original: `conditioned = self.physics_film(weather_latent, physics_features)

residual = self.unified_head(conditioned, theory_net)
pred_net = theory_net + residual`
  - Fix: `Use a constrained residual gate, e.g. pred_net = theory_net + alpha * residual with alpha predicted from physics confidence and bounded to [0,1]. Add scheduled residual magnitude/smoothness penalties, or project final predictions onto net = load - pv - wind + battery component consistency.`
  - Why: For VPP forecasting, a high-capacity residual can make the physics layer diagnostic only. This weakens conservation, makes ablations hard to interpret, and can create physically implausible corrections under distribution shift.

- **physformer/models/physical_layer.py:263-264** `[high/training_dynamics]` SOC is hard-clamped inside the battery recurrence, while losses.py later penalizes only post-clamp SOC bounds. This makes the SOC bounds loss mostly zero and removes gradient information at violations.
  - Original: `soc_next = soc_prev + (eta_charge * charge - discharge / eta_discharge) * self.dt_hours
soc_next = torch.minimum(torch.maximum(soc_next, torch.zeros_like(soc_next)), capacity)`
  - Fix: `Expose soc_unclamped and penalize violations before clamp. Add a transition residual loss on soc[t] - soc[t-1] - (eta_c*charge - discharge/eta_d)*dt, plus optional terminal SOC and operating-band penalties. Use a soft barrier or straight-through clamp if hard feasibility is required.`
  - Why: Battery feasibility is central to VPP dispatch. A hard clamp guarantees reported feasibility but hides infeasible power commands, so the optimizer does not learn to avoid boundary-saturating trajectories.

- **physformer/models/physical_layer.py:291-312** `[medium/physics_fidelity]` The PV branch is a simple irradiance times temperature factor and does not encode solar geometry, daylight constraints, panel orientation, or inverter clipping.
  - Original: `solar_energy = (weather_real[..., 1:2].clamp_min(0.0) / 3600000.0).clamp_min(0.0)
pv_temp_factor = (1.0 - pv_temp_coeff * 0.01 * (temp - 25.0)).clamp(0.5, 1.5)
pv_theory = (pv_scale + F.softplus(pv_cap)) * solar_energy * pv_temp_factor`
  - Fix: `Add a PVWatts-like differentiable prior: solar position from timestamp/region, clear-sky index, capacity and inverter clipping, panel temperature correction, and explicit zero-at-night/daylight masks.`
  - Why: PV generation is dominated by solar geometry and clipping effects. Without them, the model must learn basic physical shape from data, which weakens extrapolation across seasons and regions.
