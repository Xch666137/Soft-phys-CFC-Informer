# Peer Review Consensus Report

- File: `physformer/exp/exp_physformer.py`
- Reviewers: **claude** vs **codex**
- Matching rule: loose (v2: category equivalents + range overlap), tolerance=2

## Summary
| Bucket | Count |
|---|---:|
| Consensus (both flagged) | 1 |
| Only claude | 9 |
| Only codex | 8 |
| **Total unique** | **18** |

`claude` reported 10 raw issues; `codex` reported 9 raw issues.

## 1. Consensus Issues (1)
_Both reviewers independently flagged the same location + (compatible) category._

### L95 — `physics_fidelity` (severity: P1/P1)
- **claude**: SOC transition loss 与 SOC bounds loss 部分重叠，浪费 soc_weight 预算。implied_soc（无 clamp 的 cumsum）与 soc（有 clamp 的模型输出）仅在 SOC 触边界时才产生非零差异。而 soc_bounds_loss 直接以 ReLU 惩罚越界。两者惩罚的是同一物理现象——模型产出的充放电序列会推 SOC 出界——相当于用不同数学形式双重计数。
  - Fix: `保留 soc_bounds_loss（直接、可解释），移除 soc_transition_loss。将释放的 soc_weight 预算分配给更有信息量的约束，如 battery_power 与 ground truth aux 的 MAE（当前被 if False 跳过）。`
- **codex**: SOC consistency is self-referential: it compares `soc` to a cumsum reconstructed from the same `charge/discharge/eta` emitted by the physical layer.
  - Fix: `If SOC quality matters, compare predicted SOC against `y_aux[..., e_battery_soc_mwh]` or compute an unclamped recurrence residual before hard clamping. If primary MSE is the goal, disable this default loss or normalize it and verify it improves validation MSE.`

## 2. Only from claude (9)

- **physformer/models/conditioning.py:70-71** `[P2/design_hallucination]` FiLM scale=0.2 将 gamma 限制在 [0.8, 1.2]，beta 限制在 [-0.2, 0.2]。这意味物理特征对数据驱动 latent 的最大调制幅度仅 20%。考虑到 theory_net MAE=17 kW 而最终 MAE=2.2 kW（物理层贡献仅 13%），当前的 FiLM 约束可能太保守——物理先验几乎没有能力「塑造」数据驱动的预测。
  - Original: `gamma = 1.0 + self.film_scale * torch.tanh(raw_gamma)
beta = self.film_scale * torch.tanh(raw_beta)`
  - Fix: `将 film_scale 从 0.2 提高到 0.5~1.0，或改为可学习参数 `self.film_scale = nn.Parameter(torch.tensor(0.2))`，让模型自己决定物理调制强度。需配合 ablation 实验验证：film_scale=0 (no physics) vs 0.2 vs 0.5 vs 1.0。`
  - Why: 如果 FiLM 的物理调制对最终预测几乎没有影响，则「物理引导」变成了装饰。当前 residual_std=2.615 说明残差修正幅度远大于理论信号，暗示 FiLM 可能未充分利用物理信息。

- **physformer/models/physical_layer.py:88-92** `[P3/physics_fidelity]` PV 温度系数初始值 0.2（经 softplus 后 ≈ 0.6%/°C）低于典型晶硅面板值 -0.35~-0.5%/°C。且公式 `(1.0 - pv_temp_coeff * 0.01 * relu(temp - 25.0))` 在 temp<25°C 时不施加温度修正——低于 25°C 时 PV 效率实际会上升（负温度系数），当前实现忽略了这一物理效应。
  - Original: `pv_temp_factor = (1.0 - pv_temp_coeff * 0.01 * torch.relu(temp - 25.0)).clamp_min(0.0)`
  - Fix: `改为 `(1.0 - pv_temp_coeff * 0.01 * (temp - 25.0)).clamp_min(0.5)` 使温度效应在 25°C 两侧对称，或至少将初始 pv_temp_coeff 调整为 0.04（softplus 后 ≈ 0.4%/°C，接近物理真实值）。`
  - Why: PV 是 VPP 中的主要分布式资源之一。温度效应的物理准确性直接影响 theory_net 在高温/低温场景下的表现。当前 theory_net MAE=17 kW，PV 模型的温度效应错误可能贡献了其中一部分。

- **physformer/models/conditioning.py:95-96** `[P3/data_flow_break]` UnifiedResidualHead 将 d_model=256 维 latent 与单标量 theory_net 拼接。256:1 的维度比例意味着 theory_net 信号可能被 latent 淹没。虽然后续 Linear 层可以学习重新缩放，但初始阶段 theory_net 的信息贡献微乎其微。
  - Original: `def forward(self, conditioned, theory_net):
    inp = torch.cat([conditioned, theory_net], dim=-1)
    residual = self.net(inp)`
  - Fix: `在拼接前对 theory_net 做轻量投影：`theory_proj = nn.Linear(1, 32); theory_expanded = theory_proj(theory_net)`，然后 `inp = torch.cat([conditioned, theory_expanded], dim=-1)`。或对两部分分别做 LayerNorm 再拼接。`
  - Why: theory_net 单标量在 256 维空间中信息密度极低。一个 256×1 的 Linear 权重矩阵（256 个参数）承载全部 theory→residual 映射，容易欠拟合。扩展投影维度给予模型更多表达物理修正的能力。

- **physformer/utils/losses.py:98-101** `[P2/metric_validity]` soc_transition_loss 使用 L1 loss 在归一化 latent 空间比较 soc（已 clamp）与 implied_soc（未 clamp），但两者本身就是同一物理层的输出——该指标从不反映「模型 SOC 与真实 SOC」的差距，只反映「模型内部 self-consistency」。将其标记为物理一致性指标有误导性。
  - Original: `implied_soc = torch.cumsum(
    (eta_c * charge - discharge / eta_d) * self.dt_hours, dim=1
) + last_soc_real
soc_transition_loss = F.l1_loss(soc, implied_soc)`
  - Fix: `将 soc_transition_loss 重命名为 soc_clamp_penalty 或注释说明其仅衡量 clamp 截断量。真正的 SOC ground truth 监督需要从数据集中提供 battery SOC aux 标签。`
  - Why: 指标命名误导（transition loss 暗示时间序列转移一致性，实际是 clamp penalty），使日志解读和消融实验归因产生偏差。

- **physformer/utils/losses.py:104-107** `[P2/implementation_bug]` battery_power_mae 计算被硬编码 `if False` 跳过，导致该指标始终为零。代码意图是仅在 aux ground truth 可用时计算，但实现上有两个问题：(1) 无论如何都是 False；(2) 即使改为 True，直接用 `y_target.expand(-1,-1,5)` 扩展目标维度到 5 也是语义错误——y_target 是聚合净出力，不是各分量的 ground truth。
  - Original: `battery_power_mae = F.l1_loss(
    physics_states.get("battery_power_theory_real", charge),
    self.denorm_aux(y_target.expand(-1, -1, 5))[..., 3:4],
) if False else charge.new_tensor(0.0)`
  - Fix: `删除此段 dead code，或在 PhysLoss.forward 中从 batch_data 传入 y_aux（真正的分量 ground truth），然后计算 battery_power_mae = F.l1_loss(battery_power_theory_real, y_aux_battery_real)。`
  - Why: Dead code 降低代码可维护性，且 `if False` 明确表明作者知道这个检查需要做但跳过了。分量级 battery power supervision 是提升 theory_net 精度（当前仅 17 kW MAE）最直接有效的方法。

- **physformer/exp/exp_physformer.py:179-189** `[P1/training_dynamics]` Cosine annealing 单调衰减 LR 导致 plateau 无法逃脱。训练日志显示 epoch 13 达到最优 (Val MSE=0.420)，此后 25 个 epoch 无任何改善，但 train loss 从 0.110 持续降至 0.036——模型在训练集上继续过拟合但无法跳出验证集局部最优。Cosine annealing 在 epoch 13 时 LR=9.83e-05，到 epoch 38 已降至 7.33e-05，衰减幅度不足以提供逃逸动量。
  - Original: `cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(self.args.train_epochs - warmup_epochs, 1),
                             eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                         milestones=[warmup_epochs])`
  - Fix: `替换为 CosineAnnealingWarmRestarts(T_0=15, T_mult=1, eta_min=1e-6)，每 15 个 epoch 将 LR 重置回 base_lr，给模型多次逃脱局部最优的机会。或将 patience 从 25 降至 15，plateau 开始后 5 个 epoch 即停止，节省约 2 小时 GPU 时间。`
  - Why: 25 个 plateau epoch × 8 分钟/epoch ≈ 3.2 小时 GPU 时间零收益。WarmRestarts 可让模型在更高 LR 下探索新的 loss landscape 区域，可能找到比 0.420 更优的极小值。

- **physformer/utils/losses.py:184-191** `[P2/target_misalignment]` 物理约束 loss (SOC + anti_overlap) 占总 loss 比例极小。训练日志 epoch 2: Val Loss=3.249, Val MSE=3.225，差值仅 0.024（0.74%）。soc_weight=0.1 和 overlap_weight=0.01 的设置使物理约束在优化中被 MSE 主导。如果目标是让物理层学到有意义的参数，当前权重不足以产生有竞争力的梯度信号。
  - Original: `total_loss = terms["net_mse"]
if not self.no_battery_physics_loss:
    if not self.no_soc_consistency:
        total_loss = total_loss + self.soc_weight * (
            terms["soc_transition_loss"] + terms["soc_bounds_loss"]
        )
    total_loss = total_loss + self.overlap_weight * terms["anti_overlap_loss"]`
  - Fix: `在 warmup 阶段将 soc_weight 提高到 1.0~5.0（物理参数初步校准），然后在主训练阶段线性退火至 0.1。这样既保证早期物理参数学到合理值，又避免后期物理约束拖累 MSE。`
  - Why: 物理参数（16 个 global params + per-portfolio embedding/delta）的梯度主要来自主 loss 通过 theory_net 的间接路径。在 FiLM scale=0.2 下这条路径已经很弱，物理 loss 如果再没有足够权重，物理参数几乎完全由 data-driven MSE 间接训练，失去了「物理引导」的意义。

- **physformer/exp/exp_physformer.py:268** `[P2/training_dynamics]` 训练日志包含两次独立运行（行 1-30 和行 31-114），第一次在 epoch 11 中断后从头重训，浪费约 1.5 小时 GPU。训练循环不支持 checkpoint resume——每次 `train()` 调用都重新实例化 optimizer/scheduler，即使存在 checkpoint.pth。
  - Original: `self.model.load_state_dict(torch.load(self.checkpoint_path(), map_location=self.device))
return self.model`
  - Fix: `在 train() 开头检查 checkpoint 是否存在，若存在则加载 model/optimizer/scheduler/epoch 状态并继续训练：`if os.path.exists(self.checkpoint_path()): checkpoint = torch.load(...); self.model.load_state_dict(...); optimizer.load_state_dict(...); start_epoch = checkpoint['epoch']`。`
  - Why: 训练中断（ssh 断开、OOM、硬件故障）在生产环境中常见。支持 resume 可节省 GPU 时间和费用，对于长时间训练（>10 小时）尤其重要。

- **physformer/models/physical_layer.py:270-282** `[P1/gradient_pathology]` Battery 分支使用 Python for-loop 逐 step 迭代 96 步 (pred_len=96)，每步都通过 tanh 和 MLP。96 层的展开计算图可能导致：(1) 梯度消失——tanh 在饱和区梯度接近零，96 步连乘后指数衰减；(2) 梯度爆炸——MLP 权重在 96 步中反复使用，梯度累积放大；(3) 训练速度下降——96 次 sequential MLP forward 完全不可并行。
  - Original: `for step in range(pred_len):
    step_context = torch.cat([base_context, weather_phys[:, step, :], ...], dim=-1)
    hidden = self.battery_step_proj(step_context)
    power_raw = self.battery_power_head(hidden)
    power = power_limit * torch.tanh(power_raw)
    ...
    soc_prev = soc_next`
  - Fix: `用卷积/attention 一次性预测全部 96 步 power 序列，然后用 cumsum 计算 SOC 轨迹。或至少将 MLP 改为 GRU/LightLinear RNN 以利用并行 scan 加速（如 torch.compile + associative_scan）。`
  - Why: 96 步 sequential loop 是训练速度的隐藏瓶颈（当前无法测量因为 dataloader 是瓶颈），但当 batch_size 增大或用更快的 GPU 时将成为主要瓶颈。更重要的是，深层展开的梯度质量随步数增加衰减，导致 battery 物理参数学习不充分。

## 3. Only from codex (8)

- **physformer/utils/metrics.py:33-46** `[P2/metric_validity]` Ramp violation only checks differences inside each predicted horizon and ignores the jump from the last observed net output to the first forecast point.
  - Original: `if pred.ndim == 3:
    diff = np.abs(pred[:, 1:, :] - pred[:, :-1, :])
...
violations = diff > limits.reshape((1,) * (diff.ndim - 1) + (-1,))
return float(violations.mean() * 100.0)`
  - Fix: `Pass the last historical target for each sample into `compute_forecast_metrics` and prepend `abs(pred[:, 0] - last_hist)` to the ramp-diff tensor before computing violations.`
  - Why: A model can violate the physical boundary exactly at forecast start while still reporting 0% ramp violation, so the fallback physical metric is incomplete.

- **physformer/exp/exp_physformer.py:78-80** `[P1/design_hallucination]` The no_temporal_decoder ablation flag is never applied; ablation runs still instantiate TemporalDecoder.
  - Original: `no_future_weather=bool(getattr(self.args, "ablation_no_future_weather", False)),
battery_meta=self.battery_meta,
use_temporal_decoder=getattr(self.args, "use_temporal_decoder", True),`
  - Fix: `Set use_temporal_decoder from both config and ablation, e.g. `use_temporal_decoder=bool(getattr(args, "use_temporal_decoder", True)) and not bool(getattr(args, "ablation_no_temporal_decoder", False))`.`
  - Why: The reported `no_temporal_decoder`/flatten-head control does not test the intended mechanism, so any ablation conclusion for TemporalDecoder is invalid.

- **physformer/exp/exp_physformer.py:120-147** `[P2/design_hallucination]` `y_aux` is loaded and returned but never passed into the loss; auxiliary component/battery targets are mandatory in the dataset but provide no supervision.
  - Original: `x_weather_future, y_target, y_aux,
...
loss, debug, terms = self.criterion(
    outputs, y_target, batch_context, collect_debug=collect_debug,
)`
  - Fix: `Either remove the mandatory aux-target requirement, or pass `y_aux` to `PhysLoss` and add explicitly weighted component/SOC/battery supervision terms. Keep these weights default-off if primary MSE is the only objective.`
  - Why: The code shape implies auxiliary supervision, but the actual gradient path ignores it. Theory MAE and component diagnostics therefore cannot be attributed to component-level training.

- **physformer/models/physical_layer.py:243-246** `[P2/gradient_pathology]` The anti-overlap loss is identically zero because charge and discharge are derived from opposite ReLU sides of the same signed power.
  - Original: `power = power_limit * torch.tanh(power_raw)

charge = F.relu(power)
discharge = F.relu(-power)`
  - Fix: `Remove `anti_overlap_loss` for this signed-power parameterization, or change the battery head to predict separate smooth charge/discharge channels if overlap is a real constraint to learn.`
  - Why: `charge * discharge` is zero almost everywhere by construction, so `overlap_weight` and the `no_battery_physics_loss` comparison do not measure the claimed constraint.

- **physformer/exp/exp_physformer.py:307-314** `[P2/metric_validity]` `residual_std` in test metrics is computed in normalized target space, while MSE/MAE/Theory MAE are reported in MW units.
  - Original: `preds_real = self._denorm_target_np(preds)
trues_real = self._denorm_target_np(trues)
theory_real = self._denorm_target_np(theory_nets)
...
metrics["theory_mae"] = float(np.mean(np.abs(theory_real - trues_real)))
metrics["residual_std"] = float(np.std(residuals))`
  - Fix: `Compute `residual_std_real = np.std(preds_real - theory_real)` and either report that in MW or rename the current metric to `residual_std_norm`.`
  - Why: The current number cannot be compared to MW-scale errors and can overstate/understate the residual correction depending on the train target scaler.

- **physformer/models/physical_layer.py:309-315** `[P2/data_flow_break]` Trainable wind cut-in/rated/cut-out thresholds are used inside boolean comparisons, blocking gradient through the regime-selection logic.
  - Original: `rated = w_cut_in + w_rated_delta
cut_out = rated + w_cut_out_delta
rising_curve = ((wind - w_cut_in) / (rated - w_cut_in + 1e-6)).clamp(0.0, 1.0) ** 3
running_mask = ((wind >= w_cut_in) & (wind <= cut_out)).float()
plateau = ((wind > rated) & (wind <= cut_out)).float()
wind_curve = plateau + (1.0 - plateau) * rising_curve
wind_theory = (w_scale + F.softplus(w_cap)) * wind_curve * running_mask`
  - Fix: `Use differentiable soft gates, e.g. sigmoid ramps around cut-in/rated/cut-out with a temperature parameter, or make thresholds fixed non-trainable constants.`
  - Why: `w_cut_out_delta` receives no useful gradient except through non-differentiable masks, and threshold learning is largely illusory.

- **physformer/data/data_factory.py:425-552** `[P1/data_flow_break]` Portfolio IDs are local per split (`group_idx`), but the physical layer uses them as indices into train-time portfolio embeddings.
  - Original: `self._portfolio_id_to_idx = {gid: i for i, gid in enumerate(self.group_ids)}
...
portfolio_idx = torch.tensor(group_idx, dtype=torch.long)`
  - Fix: `Persist a train-split `portfolio_id -> embedding_idx` mapping in scaler/model metadata and use it for all splits. For unseen val/test portfolios, use a reserved unknown/global embedding or disable per-portfolio deltas for portfolio-holdout evaluation.`
  - Why: With `portfolio_manifest` splits such as `train_portfolio_01`, `val_portfolio_01`, and `test_portfolio_01`, validation/test index 0 reuses the embedding learned for a different train portfolio. This silently corrupts physics parameters and makes MSE/ablation attribution unreliable.

- **physformer/data/data_factory.py:545** `[P1/metric_validity]` The model consumes realized future weather columns as `x_weather_future`; if these are observations rather than forecast-at-issue-time covariates, test MSE has future-information leakage.
  - Original: `x_weather_future = seq_y_raw[-self.pred_len :, weather_start:weather_end]`
  - Fix: `Use weather forecast columns generated before the prediction time, or run/report the primary benchmark with `ablation_no_future_weather=True`. If perfect future weather is an intentional conditional-forecast setting, label the metric as conditional on realized weather.`
  - Why: Future temperature/irradiance/wind strongly determine VPP net output. Using actual future weather can explain very low test MSE and is not comparable to black-box baselines that do not receive the same future information.
