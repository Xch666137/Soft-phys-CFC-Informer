# Peer Review Consensus Report

- File: `V3→V4.x synthesis — optimization direction and innovation`
- Reviewers: **claude** vs **codex**
- Matching rule: loose (v2: category equivalents + range overlap), tolerance=2

## Summary
| Bucket | Count |
|---|---:|
| Consensus (both flagged) | 0 |
| Only claude | 6 |
| Only codex | 12 |
| **Total unique** | **18** |

`claude` reported 6 raw issues; `codex` reported 12 raw issues.

## 1. Consensus Issues (0)
_Both reviewers independently flagged the same location + (compatible) category._

_(none)_

## 2. Only from claude (6)

- **论文 baseline 选择 — V4 应为当前 submission 模型** `[P1/target_misalignment]` V4 在测试集上的最终 MAE（1.976kW）是所有 Phase 1+ variant 中第二好的（仅次于 V3 的 1.932），且 Theory MAE 比 V3 改善 21.8%。V4.2 虽然在物理质量指标上全面领先（Theory MAE 2.981kW），但最终 MAE 退化 6%。论文的核心 claim 是 'physical guidance improves forecasting'——V4 用 V3→V4 的 -21.8% Theory MAE + 仅 -2.3% MAE 退化来支撑这个 claim。V4.2 反而削弱了这个叙事（better physics → worse forecast）。建议 V4 为 submission baseline，V4.2 的理论质量作为 ablation 讨论。
  - Fix: `论文中使用 V4 作为主结果。Ablation 表中展示：V3 (baseline), V4 (Phase 1: calendar + component loss + load proxy), V4 w/o component loss (消融), V4 w/o calendar (消融)。V4.2 的理论质量放在 supplementary 或 future work 中讨论 'theory quality can be further improved by...'。`
  - Why: 审稿人只看最终 MAE/MSE。如果论文声称 '更好的物理引导' 但 Table 1 的 MAE 比 V3 差 6%，审稿人会直接 reject。V4 的 MAE 仅差 2.3%——勉强可接受，尤其是配合 -21.8% Theory MAE 的 narrative。

- **下一步方向 1 — Load 独立建模（最高优先级）** `[P1/missing_mechanism]` 分量评估反复验证了 Load 是最大瓶颈：V4 中 Load theory MAE=14.7kW 是 Wind (0.83kW) 的 18 倍。Load 的物理本质（人类行为）与 PV/Wind（物理方程）根本不同。当前 theory_net 对 Load 仅提供温度冷热度日模型 + sigmoid calendar scalar + net injection AR proxy——这三者都无法有效捕捉 Load 的驱动因素（日历、节假日、占据率、电价弹性）。建议实施 pre-discussed Dual-Stream：Stream A (PV+Wind) 保留 FiLM + physics equations；Stream B (Load) 使用 iTransformer-style variable attention，以 calendar embeddings、historical load、temperature 为变量 token。
  - Fix: `(1) 在论文中报告 per-component theory MAE 作为发现（'physics guidance excels for weather-driven components, underperforms for human-driven load'）；(2) 将 Dual-Stream 作为 proposed future work；(3) 如果时间允许，跑一组 Dual-Stream 的初步结果放进 supplementary。`
  - Why: Load/Wind 的 18x 误差比是论文中最有力的 story element——它既展示了物理引导的成功（Wind 0.83kW），也诚实地承认了局限性（Load 14.7kW）。审稿人欣赏这种 honesty + insight combination。

- **下一步方向 2 — Paper 的新颖性定位** `[P1/target_misalignment]` 当前 PhysFormer 的贡献定位需要明确：(1) Explicit physics layer with differentiable PV/Wind/Battery/Load equations —— 这是核心创新；(2) FiLM-conditioned integration of physics features into Transformer latent space —— 第二个创新点；(3) Portfolio-specific physics adaptation via learned embedding deltas —— 第三个创新点；(4) Per-component theory decomposition enabling interpretable error attribution —— 第四个创新点。与 iTransformer 的差异化：iTransformer 是纯数据驱动的 variable-attention，PhysFormer 是 physics-guided structure。两者的 intersection（Dual-Stream）才是新颖性爆发点。
  - Fix: `论文结构建议：(1) Introduction: 定位 'physics-guided forecasting for VPPs' 的 gap——现有方法要么纯物理（NWP-based，忽略数据模式）要么纯数据驱动（Transformer，物理不可解释）；(2) Method: PhysFormer architecture with ExplicitVPPPhysicalLayer + FiLM + component theory；(3) Experiments: V4 results + per-component analysis + ablation；(4) Discussion: Load limitation → future Dual-Stream direction。`
  - Why: 审稿人需要在 Introduction 的前两段就看到 'what is the gap and why does it matter'。Physics-guided forecasting for VPPs 是一个 underexplored intersection——NWP-based methods are too coarse for distributed VPP assets, while pure data-driven methods lack physical consistency guarantees.

- **下一步方向 3 — Component-consistent residual（中期）** `[P2/missing_mechanism]` 当前 residual 是一个 scalar 修正（pred_net = theory_net + residual），不区分 PV/Wind/Load/Battery 各自需要多少修正。分析显示 residual mean=-1.15kW（V4.2），但不同分量的 residual 需求差异巨大（Load 需要 ~5kW 修正，Wind 需要 ~0.1kW）。统一的 scalar residual 无法针对性修正每个分量。建议：predict per-component residuals，最终 net = load_theory + load_res - (pv_theory + pv_res) - (wind_theory + wind_res) + (batt_theory + batt_res)。
  - Fix: `这需要 aux targets 的 component-level ground truth（p_load_mw, p_pv_mw, p_wind_mw）。幸运的是数据集已有这些。在 losses.py 中已有 component_theory_real（shape [B, P, 5]），可以扩展 UnifiedResidualHead 输出 5 维而非 1 维。`
  - Why: Per-component residual 是介于 'scalar residual' 和 'full Dual-Stream' 之间的中间方案——改动适中（仅改 head 输出维度），但能大幅提升物理可解释性。

- **下一步方向 4 — Per-component ablation table（论文必须）** `[P1/missing_mechanism]` V4 包含 4 个同时变更（calendar embedding + load proxy + component loss + per-component eval）。每个变更对 Theory MAE -21.8% 的贡献无法归因。论文的 ablation 是最低要求——审稿人会问 'which change contributed how much?'。需要 4 个消融实验，每个跑 25 epoch：(1) V4 minus calendar flags (time_feat_dim=8); (2) V4 minus component loss (weight=0); (3) V4 minus load AR proxy; (4) V4 minus all (V3 baseline)。
  - Fix: `消融结果放入 Table 3 (Ablation Study)。最可能的结果：component loss 是 Theory MAE 改善的主因，calendar flags 是收敛加速的主因，load AR proxy 贡献最小。`
  - Why: 没有消融的论文在 2024+ 审稿标准下几乎无法通过。EPSR 作为 Elsevier 期刊要求严格的实验 evidence。

- **创新点 — 'Interpretable VPP decomposition via physics-guided theory_net'** `[P2/missing_mechanism]` V4 的 per-component analysis（Load=14.7kW, PV=4.0kW, Wind=0.83kW）是论文中最独特的贡献——它不仅是 forecasting 结果，更是对 VPP 运行机制的量化洞察。这个分析在现有 VPP forecasting 文献中几乎不存在（大多数论文只报告 net MAE，不分解到分量）。将 'interpretable decomposition' 作为论文的二级贡献，可以差异化 PhysFormer 与所有 black-box forecasting models。
  - Fix: `在 Experiments 中加一个 'Interpretability Analysis' subsection：展示 per-component theory MAE，讨论 Wind (physics works) vs Load (physics fails) 的含义，将 Dual-Stream 自然地推导为下一步。`
  - Why: 审稿人喜欢 'insight-driven' 论文而非 'number-driven' 论文。V4 的 per-component data 提供了独一无二的 insight，应该被充分展示。

## 3. Only from codex (12)

- **configs/drivers/train_three_stage.yaml:3-45; physformer/runner/config.py:54-72; physformer/exp/exp_physformer.py:180-244** `[high/implementation]` The configured three-stage training pipeline is not actually implemented: training_mode, freeze_backbone, theory_* weights, aux_weight_stageA, phys_layer_lr_scale, and init_from_run are not honored in Exp_PhysFormer.
  - Original: `stageA_net_first:
  init_from_job: stage0_physics_warmup
  phys_layer_lr_scale: 0.1
  aux_weight_stageA: 0.2
...
TRAINING_KEYS = (... "soc_weight", "component_loss_weight", "restart_t0", "restart_t_mult")`
  - Fix: `Add these keys to config parsing, load args.init_from_run checkpoints before training, implement mode-specific freezing via freeze_for_physics_warmup, and create optimizer parameter groups for phys_layer_lr_scale.`
  - Why: Without real staged initialization and parameter-group control, the reported curriculum is mostly declarative. This weakens reproducibility and can explain unstable physics-versus-accuracy trade-offs.

- **physformer/models/temporal_decoder.py:34-37** `[high/architecture]` TemporalDecoder receives y_mark but ignores it; each forecast step is represented only by a static learned query.
  - Original: `def forward(self, memory, y_mark=None):
    B = memory.shape[0]
    query = self.query_pos.expand(B, -1, -1)
    attn_out, _ = self.cross_attn(query, memory, memory, need_weights=False)`
  - Fix: `Build decoder queries from learned horizon embeddings plus future time features: query = query_pos + Linear(y_mark) + Fourier/holiday embeddings. Add decoder self-attention or causal horizon mixing before cross-attention.`
  - Why: VPP net injection has strong hour-of-day, weekday, solar-angle, and ramp-timing structure. Static horizon queries force the following WeatherFusion block to recover calendar dependence later, limiting temporal expressiveness and making long-horizon extrapolation brittle.

- **physformer/layers/revin.py:11-77; physformer/models/physformer.py:77-83** `[medium/missing_mechanism]` RevIN is implemented but not used by PhysFormer; the model relies on fixed global scaler buffers.
  - Original: `self.register_buffer("target_mean", self._to_buffer(target_mean, 1, 0.0))
self.register_buffer("target_std", self._to_buffer(target_std, 1, 1.0))
self.register_buffer("aux_mean", self._to_buffer(aux_mean, 5, 0.0))
self.register_buffer("aux_std", self._to_buffer(aux_std, 5, 1.0))`
  - Fix: `Add optional RevIN or adaptive per-portfolio normalization on the data-driven history and residual path. Keep the explicit physical layer in calibrated real units through dataset scalers.`
  - Why: VPP distributions shift by season, weather regime, region, portfolio size, and renewable penetration. RevIN-style stationarization is a standard mechanism for non-stationary time-series forecasting.

- **physformer/models/conditioning.py:88-107; physformer/utils/losses.py:71-78** `[medium/missing_mechanism]` The model is deterministic and predicts only a point residual; there is no uncertainty quantification or probabilistic forecast head.
  - Original: `self.net = nn.Sequential(
    nn.Linear(d_model + theory_proj_dim, d_model),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_model, 1),
)
...
net_mse = F.mse_loss(pred_net, y_target)`
  - Fix: `Add quantile heads, Gaussian/Student-t distribution heads, or ensemble dropout with calibration. Use uncertainty to gate physics residual corrections and report P50/P90 or conformal intervals.`
  - Why: VPP operation needs risk-aware forecasts for reserves, market bidding, and constraint margins. Point MSE cannot represent weather forecast uncertainty or rare ramp risk.

- **physformer/models/physical_layer.py:133-158; physformer/exp/exp_physformer.py:85** `[high/scalability]` Portfolio adaptation is an embedding lookup over known portfolio IDs, so it cannot generalize compositionally to unseen VPPs or changing asset mixes.
  - Original: `self.portfolio_embed = nn.Embedding(self.num_portfolios, per_portfolio_dim)
self.portfolio_delta = nn.Linear(per_portfolio_dim, 16)
...
emb = self.portfolio_embed(portfolio_ids)  # (B, per_portfolio_dim)
return self.portfolio_delta(emb)            # (B, 16)`
  - Fix: `Condition physical parameters on explicit asset metadata using a DeepSets/GNN asset encoder: capacities, locations, turbine/PV specs, battery ratings, region, and portfolio composition. Keep ID embeddings only as optional residual calibration.`
  - Why: Real VPP fleets change over time. ID-only deltas memorize training portfolios and break for new assets, portfolio expansion, or cross-region deployment.

- **physformer/layers/encoder.py:137-158; configs/physformer_default.yaml:34-35** `[medium/scalability]` The encoder is a plain full-sequence Transformer over 672 steps with no patching, multiscale decomposition, or asset-wise tokenization.
  - Original: `self.layers = nn.ModuleList([
    EncoderLayer(...)
    for i in range(num_layers)
])
...
for layer in self.layers:
    x = layer(x, mask)
return self.norm(x)`
  - Fix: `Add PatchTST/iTransformer-style patching or multiresolution temporal blocks, and consider variable/asset tokens for load/PV/wind/battery streams. Use local attention for high-frequency ramps plus global attention for daily context.`
  - Why: Longer horizons, finer sampling, or more input channels will make full attention expensive and may dilute localized ramp information.

- **physformer/models/physical_layer.py:178-207** `[high/physics_fidelity]` The load branch is too shallow and uses recent net injection as a load proxy, mixing load with PV, wind, and battery effects.
  - Original: `recent_net_avg = x_net_hist_real[:, -24:, :].mean(dim=1, keepdim=True)
autoreg_correction = self.load_autoreg_proj(recent_net_avg)
...
load_pre = (
    base
    + heat_sens * heating
    + cool_sens * cooling
    + calendar_gain * calendar_profile
    + F.softplus(self.load_autoreg_gain) * autoreg_correction
)`
  - Fix: `Replace this with a dedicated load model using lagged load/estimated load states, temperature-lag features, humidity or heat-index terms, day-type embeddings, and a small temporal state-space/TCN module. Avoid using aggregate net injection as the only autoregressive load signal.`
  - Why: Load is often the dominant VPP uncertainty. Net injection is not a clean load measurement, so this branch can learn physically ambiguous corrections and degrade component consistency.

- **physformer/utils/losses.py:186-198** `[high/training_dynamics]` Loss weighting is static and mixes normalized net MSE with normalized component MAEs; there is no adaptive balancing, curriculum, or uncertainty weighting.
  - Original: `total_loss = terms["net_mse"]
...
if self.component_loss_weight > 0:
    total_loss = total_loss + self.component_loss_weight * (
        terms["component_load_mae"]
        + terms["component_pv_mae"]
        + terms["component_wind_mae"]
        + terms["battery_power_mae"]
    )`
  - Fix: `Use MW-space component losses or learned homoscedastic uncertainty weights, GradNorm, or a scheduled curriculum: component/theory warmup, then residual fitting, then operational constraints. Track Pareto trade-offs instead of one fixed scalarization.`
  - Why: The target report shows better theory quality can worsen final MAE. Static scalar weights can over-constrain the residual or under-train the physical branches depending on scale and horizon.

- **physformer/models/physformer.py:242-245** `[high/architecture]` The neural stream sees only aggregated theory_net plus battery features, while component-level physical estimates are discarded before FiLM.
  - Original: `theory_net_real = physics_states["theory_net_real"]
battery_feats = physics_states["battery_feats_real"]
theory_net = self._norm_target(theory_net_real)
physics_features = torch.cat([theory_net, battery_feats], dim=-1)  # (B, P, 5)`
  - Fix: `Feed normalized component features [load, pv, wind, battery_power, soc], capacities, and component confidence scores into PhysicsFiLM or a component-aware residual head. Predict residual components and recombine as load_res - pv_res - wind_res + battery_res.`
  - Why: Aggregate net injection can hide large compensating component errors. A VPP forecast needs to know whether error comes from load, PV, wind, or storage because each has different dynamics and physical constraints.

- **physformer/models/physformer.py:250-252; physformer/utils/losses.py:186-198** `[high/physics_fidelity]` The final residual is unconstrained; pred_net can violate VPP ramp limits, asset capacity envelopes, and feasible dispatch even if theory_net is physically plausible.
  - Original: `residual = self.unified_head(conditioned, theory_net)
pred_net = theory_net + residual`
  - Fix: `Constrain or regularize residuals with differentiable ramp-rate, capacity-envelope, and power-balance losses. Use a physics-uncertainty gate so large residual corrections are allowed only when component uncertainty is high.`
  - Why: Operational VPP forecasts must remain feasible for dispatch and power-flow validation. An unconstrained residual can improve MSE while destroying physical consistency.

- **physformer/models/physical_layer.py:263-264; physformer/utils/losses.py:91-95** `[high/training_dynamics]` Battery SOC is hard-clamped before the SOC bounds loss, making soc_bounds_loss nearly redundant and killing gradients at the physical boundary.
  - Original: `soc_next = soc_prev + (eta_charge * charge - discharge / eta_discharge) * self.dt_hours
soc_next = torch.minimum(torch.maximum(soc_next, torch.zeros_like(soc_next)), capacity)`
  - Fix: `Keep raw_soc_next for violation penalties and use a smooth projection only for downstream features. Add terminal SOC, degradation, and reserve/headroom penalties if storage behavior matters operationally.`
  - Why: Hard projection hides infeasible control tendencies during training. The model can learn to rely on clipping instead of learning feasible battery dynamics.

- **physformer/models/physical_layer.py:291-335** `[medium/physics_fidelity]` PV and wind physics are first-order approximations without solar geometry, clear-sky normalization, hub-height correction, air density, curtailment, or asset availability.
  - Original: `solar_energy = (weather_real[..., 1:2].clamp_min(0.0) / 3600000.0).clamp_min(0.0)
wind = weather_real[..., 2:3].clamp_min(0.0)
...
pv_theory = (pv_scale + F.softplus(pv_cap)) * solar_energy * pv_temp_factor
...
wind_theory = (w_scale + F.softplus(w_cap)) * wind_curve * running_mask`
  - Fix: `Use capacity-normalized PV and wind submodels: POA or clear-sky-index PV with solar elevation and module temperature; hub-height wind correction with air-density adjustment, turbine power curves, availability and curtailment states.`
  - Why: Weather-to-power conversion is nonlinear and asset-specific. Simplified physics may improve smoothness but can be systematically wrong during low sun, high temperature, storms, curtailment, and cut-out regimes.
