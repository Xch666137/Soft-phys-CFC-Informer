# Peer Review Consensus Report

- File: `results/physformer_v4 — V4 Phase 1 results vs V3`
- Reviewers: **claude** vs **codex**
- Matching rule: loose (v2: category equivalents + range overlap), tolerance=2

## Summary
| Bucket | Count |
|---|---:|
| Consensus (both flagged) | 0 |
| Only claude | 7 |
| Only codex | 13 |
| **Total unique** | **20** |

`claude` reported 7 raw issues; `codex` reported 13 raw issues.

## 1. Consensus Issues (0)
_Both reviewers independently flagged the same location + (compatible) category._

_(none)_

## 2. Only from claude (7)

- **V4 metrics — Theory MAE 改善 21.8% 但 Test MAE 轻微退化** `[P1/metric_validity]` Theory MAE 从 4.87kW→3.81kW 大幅改善 (-21.8%)，但最终 Test MAE 从 1.93kW→1.98kW 轻微退化 (+2.3%)。Residual std 从 6.55kW→4.31kW (-34.1%) 说明残差修正更精准了，但 residual mean 为 +2.26kW（系统性偏置）。可能的原因：(a) theory_net 改善后 residual 的修正需求减少，但 residual 有正偏置导致整体 MAE 未同步改善；(b) component_loss=0.05 可能对 theory_net 施加了过强的组件监督，使得 theory_net 在 per-component 上表现更好但在 net injection 聚合时 suboptimal；(c) 早停 epoch 36 比 V3 的 epoch 51 少训练了 15 epoch，可能未达到最佳 residual 权重。
  - Fix: `(1) 用 V4 checkpoint 继续训练（higher patience=50），观察 residual mean 是否随时间向零收敛；(2) 降低 component_loss_weight 从 0.05 到 0.02，减少组件监督对 net MSE 的竞争；(3) 分析 residual mean 的时间演化——如果 bias 在增加说明 theory 和 residual 在互相推搡而非协同。`
  - Why: Theory MAE -21.8% 是显著改善，但最终 MAE 没跟上说明 residual 没有充分利用 theory 的改进。需要区分是训练不充分还是 loss 设计问题。

- **V4 component metrics — Load = 14.71kW vs Wind = 0.83kW 的不对称性** `[P1/physics_fidelity]` 分量评估揭示了一个极端的不对称：Wind theory 误差仅 0.83kW（风速→功率的物理映射几乎完美），PV theory 误差 4.00kW（辐照度→功率映射良好），Load theory 误差 14.71kW（温度冷暖度日模型极弱）。Load/Wind 误差比 = 17.7:1。这直接验证了 Phase 1 的核心假设——Load 不能用与 PV/Wind 同等的物理引导建模。但当前 Load 分支的改善手段（calendar flags + AR proxy）将 error 从 V3 的不可观测降到了 14.71kW，但相比 Wind 的 0.83kW 仍有数量级差距。
  - Fix: `Load 15kW 的 theory MAE 与 V3 整体 MAE 1.93kW 之间的关系——Load 是 net injection = PV+Wind-Load 中的减项，15kW 的 load error 在聚合时被 PV/Wind 的准确预测部分抵消。但这在操作层面不可接受：VPP 运营需要知道负载的确切值来决定电池调度。建议 Phase 2 为 Load 建立独立的数据驱动分支（iTransformer-style variable attention）。`
  - Why: 分量评估是 V4 最重要的新信息。它提供了量化证据支持 Dual-Stream 架构决策。Wind 0.83kW 也证明了物理引导对可再生能源分量的有效性——这是一种有价值的负向验证。

- **V4 — Val SOC 始终为零的含义** `[P3/training_dynamics]` 所有 34 个 epoch 的 Val SOC = 0.000000。这是好消息——电池 SOC 约束在任何验证集样本上均未被违反。但这也意味着 soc_weight=0.1 仅是一个安全网，永不被触发。SOC=0 的完美合规可能来自：(a) battery branch 的结构性先验（softplus 功率限制 + tanh 归一化 + soc clamp）足够强，无需 loss 约束；(b) 验证集的 battery 行为相对温和，soc 极少接近边界。可以将此表述为论文的正面发现：'物理结构性先验足以保证 SOC 合规，无需在 loss 中显式惩罚'。
  - Fix: `在论文中：(1) 报告 SOC violation=0；(2) 运行 soc_weight=0 的消融实验证明 battery branch 的结构性先验（非 loss 约束）是 SOC 合规的来源；(3) 可选：报告 battery SOC 的最大/最小值（soc_min=0.0, soc_max=0.046 MWh）展示 SOC 范围合理。`
  - Why: SOC=0 是强正面信号，需要正确归因（结构性先验而非 loss 约束）。

- **V4 — 收敛速度 3.4x 加速** `[P2/training_dynamics]` V4 epoch 11 达到 Val MSE=0.381，V3 epoch 37 达到 0.387。V4 用 3.4x 更少的 epoch 达到了更好的 Val MSE。可能原因：(a) calendar embeddings (is_weekend/is_holiday) 提供了有价值的离散时间信号，加速了时间模式的识别；(b) component loss 为 theory branches 提供了直接的梯度，加速了物理参数的学习；(c) AR load proxy 给 load branch 提供了一个 persistency baseline。需要消融来确认各因素的贡献。
  - Fix: `消融实验：(1) V4 minus calendar flags (time_feat_dim=8, no is_weekend/is_holiday)；(2) V4 minus component loss (component_loss_weight=0)；(3) V4 minus AR proxy (no load_autoreg)。各跑 15 epoch 比较收敛曲线。`
  - Why: 3.4x 收敛加速是重要的论文卖点，但需要归因到具体改动。审稿人会问 '哪个改动贡献了加速'。

- **V4 — Battery component metrics 的解读** `[P3/metric_validity]` Battery power MAE=21.35kW 和 battery SOC MAE=0.020MWh 不在传统 forecasting metrics 中。这些值需要在上下文中解读：net injection 的总 scale 是多少？如果 net injection 均值是 MW 级别（如 ~5kW），则 21kW 的电池功率误差非常大。但 battery_soc_mae=0.020MWh (20kWh) 在典型的电池容量下（如 0.046MWh = 46kWh max soc）占比约 43%——相当大。这说明 theory_net 对电池行为的建模仍然粗糙。
  - Fix: `在论文中：(1) 用相对误差（battery_mae / mean_battery_power）而非绝对值报告；(2) 将电池指标放在 VPP 运行的上下文中——电池调度是 VPP 的核心操作，21kW 的电池预测误差直接影响经济调度优化。`
  - Why: 绝对数值 21kW 没有上下文无法判断好坏。需要相对化。

- **V4 train.log — 早停 epoch 36，WarmRestarts 未充分探索** `[P2/training_dynamics]` V4 在 epoch 11 达到最优 0.381 后，连续 25 个 epoch（12-36）未能刷新纪录。两次 WarmRestarts（epoch 20 和 35）均未产生效果——restart 后的 Val MSE 停留在 0.397-0.398，远高于最优 0.381。对比 V3：restart #2（epoch 37）产生了全局最优 0.387。V4 的 restart 失效可能因为：(a) 最优 0.381 已经接近此架构的天花板；(b) 从零训练的模型（V4 从 epoch 2 开始而非 V3 checkpoint epoch 37）缺少 V3 的 37 个 epoch 预训练权重，导致 restart 探索的 loss landscape 不同；(c) component loss 改变了 loss landscape，WarmRestarts 的探索方向不再有效。
  - Fix: `(1) 或者从 V3 的 epoch 37 checkpoint 重新启动 V4 训练（保留物理参数），观察是否能从 0.381 继续改善；(2) 或者增加 patience 到 50 + 调低 component_loss_weight 到 0.02 后重新训练 100 epoch；(3) 评估是否需要调整 T_0 参数（当前 T_0=15 可能对 V4 的 loss landscape 不是最优）。`
  - Why: 36/100 epochs = 仅使用了 36% 的训练预算。如果最优在 epoch 11 就出现了，说明早期过拟合或 scheduler 失配。论文中需要解释训练停止的原因。

- **V4 — 训练不完整（36/100 epoch）** `[P2/training_dynamics]` V4 使用 patience=25 且 early_stop_start_epoch=10。epoch 11 是最优后，epoch 12→36 的 25 个 epoch 均无改善，触发早停。但 epoch 11 恰好在 early_stop_start_epoch+1 处，这意味着 early_stop_start_epoch=10 可能设置得太低——best epoch 恰好在 start_epoch 之后 1 个 epoch，导致后续 25 个 epoch 没有足够机会。建议 early_stop_start_epoch=15 或 20，给 early training 更多探索空间。
  - Fix: `(1) 调整 config: early_stop_start_epoch=15, patience=30；(2) 从 epoch 11 checkpoint 恢复训练，继续跑剩余 54 epoch。`
  - Why: 36/100 epoch 的使用率（36%）太低了。早停应该在模型真正收敛时触发，而非在 early training 阶段。

## 3. Only from codex (13)

- **configs/drivers/train_three_stage.yaml:25-28** `[high/implementation]` Stage-specific curriculum controls are declared but not consumed by config_to_args/apply_job_overrides/Exp_PhysFormer, so the three-stage training design is mostly nominal.
  - Original: `phys_layer_lr_scale: 0.1
aux_weight_stageA: 0.2
soc_weight_stageA: 0.1
overlap_weight_stageA: 0.01`
  - Fix: `Add training_mode, freeze_backbone, use_aux_supervision, theory_aux_weight, phys_layer_lr_scale, and stage-specific weights to TRAINING_KEYS and apply_job_overrides; call freeze_for_physics_warmup() for stage0 and use optimizer param groups for physics vs backbone.`
  - Why: If curriculum knobs are ignored, physics warmup, net-first training, and operational fine-tuning do not actually impose the intended optimization sequence.

- **physformer/models/temporal_decoder.py:34-40** `[high/architecture]` TemporalDecoder ignores y_mark and uses a static learned query table; horizon queries are not calendar-, holiday-, portfolio-, or regime-conditioned.
  - Original: `def forward(self, memory, y_mark=None):
    B = memory.shape[0]
    query = self.query_pos.expand(B, -1, -1)  # (B, pred_len, d_model)
    attn_out, _ = self.cross_attn(query, memory, memory, need_weights=False)
    query = self.ln1(query + attn_out)
    query = self.ln2(query + self.ffn(query))
    return query`
  - Fix: `Construct queries as query_pos + MLP(y_mark) + portfolio/region embedding, and add decoder self-attention or temporal convolution over future steps so ramps and horizon-to-horizon coupling are modeled before weather fusion.`
  - Why: VPP net injection has strong hour-of-day, weekend/holiday, and operating-regime dependence; static horizon queries limit expressiveness, especially for rare ramps and calendar shifts.

- **physformer/models/physical_layer.py:53-64** `[medium/physics_fidelity]` Battery numerical integration uses a fixed dt_hours=0.25 rather than deriving time step from the data frequency or timestamps.
  - Original: `dt_hours=0.25,
...
self.dt_hours = dt_hours`
  - Fix: `Infer dt from dataset timestamps/freq, pass dt as a tensor when gaps or mixed resolutions exist, and use a validated integration scheme for SOC updates across variable horizons.`
  - Why: The model silently assumes 15-minute data. Longer horizons, resampled data, missing intervals, or mixed markets will produce incorrect battery energy accounting.

- **physformer/models/physformer.py:77-113** `[medium/missing_mechanism]` RevIN exists in physformer/layers/revin.py but is not wired into PhysFormer; the model relies on global scaler buffers only.
  - Original: `self.register_buffer("target_mean", self._to_buffer(target_mean, 1, 0.0))
self.register_buffer("target_std", self._to_buffer(target_std, 1, 1.0))
...
self.stat_embedding = DataEmbedding(
    c_in=enc_in, d_model=d_model, embed_type=embed, freq=freq, dropout=dropout,
    time_enc_in=time_feat_dim,
)
...
self.flatten_head = FlattenHead(seq_len, d_model, pred_len, dropout)`
  - Fix: `Add RevIN or another reversible stationarization block on the data-driven net/history stream, while keeping the physical layer in real units via dataset scalers. Denormalize only the final residual/prediction path consistently.`
  - Why: Load and renewable portfolios are non-stationary across seasons and assets. Without per-instance normalization, the Transformer must learn distribution shift that RevIN-style methods handle directly.

- **physformer/models/physformer.py:90-101** `[medium/scalability]` The default path uses full self-attention over the historical sequence; this scales poorly for longer histories and higher-resolution horizons.
  - Original: `attn_cls = ProbAttention if attn == "prob" else FullAttention
self.encoder = Encoder(
    num_layers=e_layers,
    d_model=d_model,
    n_heads=n_heads,
    d_ff=d_ff,
    attn_cls=attn_cls,
    dropout=dropout,
    use_distillation=distil,
    use_rope=use_rope,
    rope_base=rope_base,
)`
  - Fix: `Use patching or multi-resolution encoders, make ProbAttention/FlashAttention/chunked attention the long-sequence default, and add temporal pyramids for week-to-month histories.`
  - Why: At seq_len=672 this is manageable, but multi-week histories, sub-15-minute data, or many exogenous channels will increase memory and latency quadratically.

- **physformer/models/conditioning.py:94-107** `[high/missing_mechanism]` The output head predicts only a point residual; there is no uncertainty quantification, quantile forecast, scenario generation, or calibration mechanism.
  - Original: `self.net = nn.Sequential(
    nn.Linear(d_model + theory_proj_dim, d_model),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_model, 1),
)
...
residual = self.net(inp)
return residual`
  - Fix: `Add probabilistic heads for quantiles or distribution parameters, train with pinball/CRPS/NLL, calibrate with conformal methods, and generate temporally coherent scenarios conditioned on weather uncertainty.`
  - Why: VPP scheduling, reserve bidding, and risk-aware control require calibrated uncertainty, not only a mean net-injection trajectory.

- **physformer/models/physical_layer.py:133-158** `[medium/scalability]` Per-portfolio physics adaptation is a fixed embedding table plus 16 deltas; it cannot cold-start new portfolios or represent many heterogeneous assets inside a VPP.
  - Original: `if self.num_portfolios > 0:
    self.portfolio_embed = nn.Embedding(self.num_portfolios, per_portfolio_dim)
    self.portfolio_delta = nn.Linear(per_portfolio_dim, 16)
...
def _get_portfolio_delta(self, portfolio_ids):
    if self.num_portfolios == 0 or portfolio_ids is None:
        return None
    emb = self.portfolio_embed(portfolio_ids)
    return self.portfolio_delta(emb)`
  - Fix: `Replace table-only deltas with metadata-conditioned hypernetworks or DeepSets/graph encoders over assets, including capacity, technology type, location, tariff, and network constraints.`
  - Why: A real VPP scales by adding/removing assets. A closed portfolio-id table learns IDs rather than transferable physical structure.

- **physformer/models/physical_layer.py:178-207** `[high/physics_fidelity]` The load branch is too weak and uses recent net injection as a proxy for load, which confounds load with PV, wind, and battery behavior.
  - Original: `def _load_branch(self, temp, y_mark, x_net_hist_real, dx=None):
    heating = F.relu(self.load_comfort_low - temp)
    comfort_high = self.load_comfort_low + F.softplus(self.load_comfort_gap)
    cooling = F.relu(temp - comfort_high)

    calendar_profile = torch.sigmoid(self.load_calendar_proj(y_mark))

    recent_net_avg = x_net_hist_real[:, -24:, :].mean(dim=1, keepdim=True)
    autoreg_correction = self.load_autoreg_proj(recent_net_avg)
...
    load_theory = F.softplus(load_pre)`
  - Fix: `Replace this with a thermal/occupancy state-space branch: latent indoor temperature or RC thermal state, lagged outdoor temperature, holiday/weekend effects, and either true historical load if available or a learned disaggregation state rather than raw net injection.`
  - Why: The V4 result reports load component MAE far larger than wind. A confounded load prior can improve theory_net superficially while degrading the physical decomposition that VPP operators need.

- **physformer/utils/losses.py:187-199** `[high/training_dynamics]` The loss mixes normalized net MSE with real-unit component MAEs using one fixed component_loss_weight, causing scale imbalance and unstable tradeoffs.
  - Original: `total_loss = terms["net_mse"]
...
if self.component_loss_weight > 0:
    total_loss = total_loss + self.component_loss_weight * (
        terms["component_load_mae"]
        + terms["component_pv_mae"]
        + terms["component_wind_mae"]
        + terms["battery_power_mae"]
    )`
  - Fix: `Normalize each component loss by aux_std or capacity, use uncertainty weighting/GradNorm/PCGrad for multi-objective balancing, and monitor a composite validation metric instead of net_mse only.`
  - Why: The V4 report shows theory MAE improved while final MAE regressed. Fixed mixed-unit weights can overfit component priors without improving the actual operational forecast.

- **physformer/exp/exp_physformer.py:187-205** `[medium/training_dynamics]` Warm-restart scheduling is hard-coded and early stopping is not cycle-aware; V4 peaked early and later restarts did not recover.
  - Original: `restart_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, restart_scheduler],
                         milestones=[warmup_epochs])
...
early_stopping = EarlyStopping(
    patience=self.args.patience, verbose=True, logger=self.logger,
    metric_name=metric_name, start_epoch=early_stop_start_epoch,
)`
  - Fix: `Make T_0/T_mult configurable, compare against OneCycleLR/cosine decay/ReduceLROnPlateau, use EMA/SWA checkpoints, and either reset or reinterpret patience at restart boundaries.`
  - Why: The reported best validation epoch was 11, followed by restarts at epochs 20 and 35 without improvement. A fixed restart cycle can waste training budget or trigger early stopping before useful later adaptation.

- **physformer/models/physical_layer.py:245-264** `[high/physics_fidelity]` Battery dispatch is a neural signed-power recurrence with hard SOC clipping; the SOC bounds loss then sees post-clipped states, so violations are hidden rather than optimized away.
  - Original: `for step in range(pred_len):
    ...
    power_raw = self.battery_power_head(hidden)  #  (B, 1)
    power = power_limit * torch.tanh(power_raw)

    charge = F.relu(power)
    discharge = F.relu(-power)

    soc_next = soc_prev + (eta_charge * charge - discharge / eta_discharge) * self.dt_hours
    soc_next = torch.minimum(torch.maximum(soc_next, torch.zeros_like(soc_next)), capacity)`
  - Fix: `Parameterize feasible charge/discharge bounds from current SOC before choosing power, use differentiable barrier penalties instead of hard clipping, and add terminal SOC, throughput/degradation, and dispatch-objective terms.`
  - Why: Hard clipping makes Val SOC violation look perfect while masking infeasible control intent. For VPP operation, battery forecasts must respect feasible power, energy, and terminal-state constraints.

- **physformer/models/physformer.py:247-252** `[high/architecture]` Physics is only a soft additive prior; the unconstrained residual can overwrite theory_net, so final pred_net is not guaranteed to be component-consistent.
  - Original: `conditioned = self.physics_film(weather_latent, physics_features)

# ---- Step 6: Residual head -> final prediction ----
residual = self.unified_head(conditioned, theory_net)
pred_net = theory_net + residual`
  - Fix: `Predict physically meaningful component residuals and recompute net injection as load - pv - wind + battery, or add a constrained residual layer with learned physics confidence, residual magnitude limits, ramp limits, and component-level diagnostics for the final prediction.`
  - Why: For VPP forecasting, improving net MAE by arbitrary residual correction can destroy physical interpretability and make downstream dispatch/risk decisions inconsistent with load, renewable, and battery behavior.

- **physformer/models/physical_layer.py:291-312** `[medium/physics_fidelity]` PV physics is reduced to irradiance times a scalar temperature factor; it omits solar geometry, plane-of-array conversion, inverter clipping, and asset nameplate constraints.
  - Original: `solar_energy = (weather_real[..., 1:2].clamp_min(0.0) / 3600000.0).clamp_min(0.0)
...
pv_temp_factor = (1.0 - pv_temp_coeff * 0.01 * (temp - 25.0)).clamp(0.5, 1.5)
pv_theory = (pv_scale + F.softplus(pv_cap)) * solar_energy * pv_temp_factor`
  - Fix: `Use a differentiable PVlib-style block: solar position, clear-sky or POA transposition, module temperature, nameplate capacity, inverter clipping, night mask, and metadata-conditioned parameters.`
  - Why: PV errors dominate net injection during daylight ramps. A scalar irradiance model will not extrapolate well across seasons, regions, cloud regimes, or differently oriented portfolios.
