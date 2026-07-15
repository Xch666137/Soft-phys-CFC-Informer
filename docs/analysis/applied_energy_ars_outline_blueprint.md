# Applied Energy 论文提纲与论证蓝图 v0.1

日期：2026-06-01

状态：AE-Gate-0 已满足；AE-Gate-1 未满足。因此本文件是 Applied Energy detailed outline + argument blueprint，不是完整投稿稿。

输入：

- `docs/analysis/applied_energy_ars_writing_adapter.md`
- `docs/analysis/applied_energy_journal_profile.md`
- `docs/analysis/applied_energy_paper_source_pack.md`
- `docs/analysis/applied_energy_pdf_extraction_matrix.md`
- `ara/logic/claims.md`

## 1. Paper Configuration Record

| 字段 | 当前选择 |
|---|---|
| Target journal | Applied Energy |
| Article type | Research article |
| Field | Virtual power plant forecasting / energy forecasting / power systems ML |
| Manuscript language | English |
| Working mode | Outline + argument blueprint only |
| Full-draft status | Blocked until AE-Gate-1 |
| Main evidence layer | ARA C08, C11, C12, with C09/C10 as secondary mechanism support |
| Journal-style layer | AE01-AE15 Applied Energy distillation |

## 2. Working Title

推荐标题：

> Robust aggregate net-power forecasting for virtual power plants via component-token separation under heterogeneous resource coupling

备选标题：

> Component-token separation for robust virtual power plant aggregate net-power forecasting

暂不推荐作为主标题：

> When fixed physics priors overfit: component-token separation for virtual power plant aggregate forecasting

原因：第三个标题冲击力强，但会让 C12 成为论文第一入口。当前更稳妥的 Applied Energy 写法是先建立 VPP aggregate forecasting 的能源系统问题，再把 fixed-prior overfitting 放入结果和讨论。

## 3. Core Mainline

英文主线：

> In VPP aggregate net-power forecasting, component-token separation improves generalization stability, while fixed physics-prior additions can amplify validation-test overfitting under heterogeneous portfolio coupling.

中文解释：

虚拟电厂聚合净功率预测不是一个普通 scalar forecasting 问题。净功率由 load、PV、wind、battery 的 signed composition 构成，component errors 会通过 covariance 和 cancellation 影响 aggregate accuracy。当前 ARA 证据支持：将异构组件分离成 component tokens 比叠加固定物理先验更稳健；固定 physics-prior additions 在当前 VPP aggregate setting 中呈现 Val 更好但 Test 更差的过拟合链条。

## 4. Contribution Statement

推荐贡献写法：

1. We formulate VPP aggregate net-power forecasting as a signed multi-component forecasting problem, where aggregate accuracy depends on component-error covariance rather than component MAE alone.
2. We introduce a component-token separated inverted Transformer that assigns heterogeneous VPP resources to independent token representations before aggregate forecasting.
3. Across three seeds, the component-token separated model improves aggregate Test MAE over the full PhysFormer c23 baseline and substantially reduces cross-seed variance.
4. We provide a controlled fixed-prior ablation chain showing that physics tokens, twin/constraint tokens, graph bias, and horizon/weather decoder additions improve validation loss but monotonically degrade Test MAE.
5. We connect these findings to Applied Energy-style operational concerns through AE-Gate-1 metrics: ramp-event error, peak/valley error, deviation penalty proxy, and reserve requirement proxy.

第 5 点目前是待补实验证据，不应在正式摘要和结论中写成已完成结果。

## 5. Detailed Outline

### Abstract

当前只能写 provisional abstract。结构如下：

1. VPP aggregate net-power forecasting supports dispatch, market participation, reserve allocation, and renewable integration.
2. The forecast target is a signed composition of heterogeneous components, so component-error coupling can dominate aggregate behavior.
3. Existing Transformer and physics-informed forecasting approaches often do not separate component representation from fixed prior injection.
4. This paper studies component-token separation for robust aggregate net-power forecasting.
5. Use C11 numbers: A1 vs c23 MAE, MSE/RMSE, and 20x lower cross-seed MAE std.
6. Use C12 numbers: A1-A5 fixed-prior ablation chain and Val/Test divergence.
7. Add operational implication only after AE-Gate-1.

禁止在当前摘要中写：

- operational cost/reserve/market gains；
- “physics-informed forecasting is ineffective”；
- “the proposed method generalizes to all VPPs”。

### 1. Introduction

#### 1.1 Energy-system motivation

目标：先让 Applied Energy 读者看到能源系统问题，而不是模型。

内容：

- VPPs aggregate distributed load, PV, wind, and battery resources.
- Forecasting aggregate net power is necessary for dispatch, reserve, market bidding, and renewable integration.
- AE02/AE03 支撑 VPP operation、resource coordination、uncertainty。
- AE04/AE05 支撑 market、forecasting error、ramping/flexible product。

#### 1.2 Why aggregate net-power forecasting is structurally different

核心句：

> VPP net power is not a single homogeneous time series, but a signed composition of heterogeneous component processes.

必须引入：

```text
P_net(t) = P_load(t) - P_pv(t) - P_wind(t) + P_battery(t)
```

解释：

- Load is behavior-driven.
- PV and wind are weather-driven.
- Battery is control-driven.
- Signed summation means component errors can cancel or amplify aggregate errors.

证据：ARA C08；AE06/AE07。

#### 1.3 Gap in existing forecasting designs

目标：不是说已有方法不行，而是指出它们没有充分处理这个结构问题。

内容：

- Transformer forecasting 在 Applied Energy 中已被接受，但不是天然优势。
- Physics-informed forecasting 在 Applied Energy 中也有成功案例，但物理约束必须和预测目标直接对应，并经 out-of-sample 验证。
- 对 VPP aggregate forecasting，shared encoder + fixed prior additions 可能混合 component coupling 与 portfolio-specific patterns。

支撑：

- AE08/AE10/AE11：Transformer forecasting 的正向案例。
- AE09：Transformer 适用性需要实证验证。
- AE12/AE13：physics-informed 成功先例，限制 C12 的过度泛化。

#### 1.4 Proposed direction

推荐写法：

> We study component-token separation as a structural inductive bias for VPP aggregate net-power forecasting.

解释：

- 不是“加更多 physics priors”；
- 不是“pure data-driven always better”；
- 是“将异构 component representation 分离，再通过 aggregate objective 学习 net behavior”。

#### 1.5 Contributions

贡献段应按上文第 4 节写，注意第 5 点要标为 AE-Gate-1 后加入。

### 2. Related Work

建议四个小节。

#### 2.1 VPP operation and uncertainty-aware forecasting

功能：

- 建立 VPP 与 operation / uncertainty / market / reserve 的连接。
- 让文章不是 energy-flavored ML。

主要文献：

- AE02：VPP operations review。
- AE03：VPP uncertainty。
- AE04/AE05：forecasting error into market/ramping strategy。

写作要点：

- 不要把 VPP 只写成数据集来源。
- 要写成一个 operation problem。

#### 2.2 Aggregate net-load and renewable forecasting

功能：

- 连接 `net = Load - PV - Wind + Battery`。
- 引出 peak/valley、ramp、flexibility requirement。

主要文献：

- AE06：peak and valley short-term net-load forecasting。
- AE07：aggregated net-load forecasting with uncertainty/correlation。

写作要点：

- 强调 aggregate forecast 的系统价值。
- 为 C08 的 component-error covariance 做铺垫。

#### 2.3 Transformer and modern time-series forecasting in energy systems

功能：

- 说明 Transformer 不是 novelty 本身。
- 强调 architecture-problem fit。

主要文献：

- AE08：multimodal Transformer for solar irradiance。
- AE09：Transformer applicability analysis。
- AE10：graph networks and Transformer architectures。
- AE11：probabilistic Transformer solar forecasting。

写作要点：

- 不能写“Transformer 已经成功，所以本文也先进”。
- 应写“component-token separation matches VPP heterogeneous structure”。

#### 2.4 Physics-informed and physics-constrained forecasting

功能：

- 给 C12 找到安全位置。
- 避免反 physics-informed 的过度表述。

主要文献：

- AE12：physics-informed RL for probabilistic wind power forecasting。
- AE13：physics-constrained wind power forecasting。

写作要点：

- 先承认 physics-informed forecasting 在合适设定中有效。
- 再指出 fixed, hand-specified priors must be validated under heterogeneous portfolio coupling。

### 3. Problem Formulation

#### 3.1 Forecasting target

定义：

```text
Given historical component and context observations X_{t-L+1:t},
forecast aggregate net power Y_{t+1:t+H}.
```

核心方程：

```text
P_net(t) = P_load(t) - P_pv(t) - P_wind(t) + P_battery(t)
```

#### 3.2 Error decomposition

推荐写：

```text
e_net = e_load - e_pv - e_wind + e_battery
```

解释：

- aggregate error depends on component error variances and signed covariances；
- lower component MAE does not necessarily imply lower aggregate MAE；
- worse component predictions can sometimes produce better aggregate accuracy if signed errors cancel。

证据：C08。

#### 3.3 Operational metrics

当前为 AE-Gate-1 待补。建议定义：

| Metric | Definition idea | Why Applied Energy cares |
|---|---|---|
| Ramp-event MAE | MAE on intervals where `abs(delta P_net)` is in top q% | flexible ramping, reserve stress |
| Peak/valley MAE | MAE on top/bottom q% net-power periods | scheduling and peak-valley operation |
| Deviation penalty proxy | thresholded or asymmetric penalty for forecast deviation | market imbalance / bidding penalty |
| Reserve requirement proxy | high-quantile absolute error, e.g. P95/P99 | reserve margin planning |

正式稿中不要只定义而无结果；如果 AE-Gate-1 未完成，这节只能作为 experiment plan。

### 4. Method

#### 4.1 Design principle

核心句：

> The proposed design separates heterogeneous VPP components before temporal aggregation, rather than injecting additional fixed priors into a shared representation.

#### 4.2 Component-token separated inverted Transformer

应写内容：

- 5 component tokens：Load、PV、Wind、Battery power、Battery SOC；
- 3 weather/context tokens；
- token-wise encoders；
- inverted self-attention over component/weather tokens；
- shared FFN decoder；
- real-unit power balance；
- net MSE objective。

避免写法：

- 不把 A1 说成 physics-guided model；
- 不把 fixed priors 写成最终方法的一部分；
- 不把旧 c23 作为主角。

#### 4.3 Comparison to shared-encoder PhysFormer c23

功能：

- 解释为什么 c23 是重要 baseline。
- 连接 C10/C11：shared encoder cancellation channel 与 component-token separation。

安全写法：

> The c23 baseline represents the earlier shared-encoder physics-guided design. It is used as a strong internal reference to test whether component-token separation can outperform a more heavily guided architecture.

#### 4.4 Fixed-prior ablation variants

列出：

| Variant | Addition | Purpose |
|---|---|---|
| A1 | component/weather tokens + simple decoder | base component-token separation |
| A2 | physics token | test simple fixed-prior injection |
| A3 | twin/constraint tokens | test explicit component/constraint representation |
| A4 | graph bias | test hand-specified coupling prior |
| A5 | horizon decoder + weather conditioning | test richer decoder and weather coupling |

这节为 C12 服务。

### 5. Experiments

#### 5.1 Dataset and setting

必须交代：

- VPP portfolio composition；
- sampling interval；
- forecasting horizon；
- train/val/test split；
- whether data are private；
- reproducibility constraints。

如果无法公开数据，必须强化：

- config disclosure；
- split description；
- metric script；
- mean ± std；
- ablation completeness。

#### 5.2 Baselines

当前必须补齐。建议最低 baseline suite：

| Category | Baseline |
|---|---|
| Naive | persistence / last-value |
| Linear time-series | DLinear or NLinear |
| Recurrent DL | GRU / LSTM |
| Modern Transformer-style | PatchTST or TFT |
| Internal strong baseline | PhysFormer c23 |
| Proposed family | A1-A5 |

如果资源不足，优先级：

1. persistence；
2. DLinear/NLinear；
3. PatchTST/TFT；
4. GRU/LSTM。

#### 5.3 Metrics

基础指标：

- MAE；
- MSE；
- RMSE；
- mean ± std over seeds。

Applied Energy 指标：

- ramp-event MAE；
- peak/valley MAE；
- deviation penalty proxy；
- reserve requirement proxy。

#### 5.4 Main comparison

目标表：A1 vs c23 vs strong baselines。

当前可写 C11：

| Metric | A1 | c23 baseline | Interpretation |
|---|---:|---:|---|
| Test MAE | 1.811e-3 ± 6e-6 | 2.069e-3 ± 1.34e-4 | A1 improves by about 12.5% |
| Test MSE | 6.766e-6 ± 4.9e-8 | 8.111e-6 ± 6.75e-7 | A1 lower aggregate error |
| Test RMSE | 2.601e-3 ± 9e-6 | 2.846e-3 ± 1.17e-4 | A1 lower RMSE |
| Cross-seed MAE std | 6e-6 | 1.34e-4 | about 20x lower variance |

注意：这个表仍需加入 external baselines 后才能成为正式 Applied Energy 主表。

#### 5.5 Fixed-prior ablation

目标表：A1-A5 Val/Test divergence。

当前可写 C12：

| Variant | Addition | Test MAE | vs A1 |
|---|---|---:|---:|
| A1 | no fixed prior addition | 0.001811 | baseline |
| A2 | physics token | 0.001819 | +0.4% |
| A3 | twin/constraint tokens | 0.001843 | +1.8% |
| A4 | graph bias | 0.001863 | +2.9% |
| A5 | horizon decoder + weather conditioning | 0.001947 | +7.5% |

写作重点：

- stronger validation performance hides poorer test generalization；
- this is setting-specific fixed-prior overfitting；
- do not generalize to all physics-informed forecasting。

#### 5.6 Operational-value analysis

当前是 AE-Gate-1 block。正式稿需要回答：

- A1 的普通 MAE 改善是否转化为 ramp-event 改善？
- A1 是否在 peak/valley 时段更稳？
- fixed-prior variants 虽然 Test MAE 更差，是否可能在 reserve/deviation proxy 上更好？
- simple baselines 是否在 operational proxy 上接近或超过 A1？

如果这些结果未补齐，不进入 full draft。

#### 5.7 Robustness and limitations

至少写：

- multi-seed stability；
- single private VPP dataset limitation；
- deterministic forecasting limitation；
- no full dispatch-cost optimization yet；
- target adaptation / data-scarce scenario as future or added experiment。

### 6. Results and Argument Flow

建议按 research questions 组织。

#### RQ1: Does component-token separation improve aggregate forecasting?

Claim：

> Component-token separation improves aggregate Test MAE/MSE/RMSE and cross-seed stability over the c23 full PhysFormer baseline.

Evidence：C11。

Allowed strength：强，但限于当前 dataset/split/baseline family。

#### RQ2: Are fixed physics-prior additions automatically beneficial?

Claim：

> In the tested inverted-Transformer family, fixed prior additions monotonically degrade Test MAE despite improving validation loss.

Evidence：C12。

Allowed strength：强，但限于 A1-A5 design space。

Forbidden expansion：

> Physics-informed learning is generally harmful.

#### RQ3: Why can component-level and aggregate behavior diverge?

Claim：

> Aggregate accuracy depends on signed component-error covariance, not only component MAE.

Evidence：C08；C10 可作为 mechanism support。

Allowed strength：强。

#### RQ4: Does the forecasting gain matter operationally?

Claim：

> Pending AE-Gate-1.

Evidence：尚未补齐。

Allowed strength：当前只能写成 planned analysis 或 motivation，不能写成 result。

### 7. Discussion

建议四段。

#### 7.1 Energy-system implication

说明 component-token separation 不是普通 ML trick，而是对应 VPP 异构资源结构。

#### 7.2 Boundary condition for fixed priors

推荐句：

> The results do not reject physics-informed forecasting. They identify a boundary condition: fixed, hand-specified priors should be validated against portfolio heterogeneity and out-of-sample aggregate behavior.

#### 7.3 Practical implication for VPP forecasting systems

当前只能保守写：

- A1 improves deterministic aggregate forecast accuracy；
- operational value still needs ramp/peak/deviation/reserve evaluation；
- architecture simplicity may help deployment and reproducibility。

#### 7.4 Limitations

必须承认：

- private/single VPP data；
- deterministic forecasts only；
- limited external baselines until AE-Gate-1；
- no full market/dispatch optimization loop yet。

### 8. Conclusion

结论只写三件事：

1. VPP aggregate forecasting has signed component coupling.
2. Component-token separation is a robust inductive bias in the current evidence.
3. Fixed priors require data-adaptive validation; more physics is not automatically better.

不要写：

- market profit improved；
- reserve cost reduced；
- universal VPP generalization；
- physics-informed forecasting is ineffective。

## 6. Claim-Evidence Map

| Paper claim | Evidence | Status | Manuscript location |
|---|---|---|---|
| VPP net power is a signed composite of heterogeneous components | ARA problem facts, C08 | Supported | Introduction, Problem Formulation |
| Aggregate accuracy depends on component error covariance/cancellation | C08 | Supported | Problem Formulation, Discussion |
| Component-token separation improves aggregate forecasting vs c23 | C11 | Supported | Method, Results RQ1 |
| A1 reduces cross-seed MAE std by about 20x vs c23 | C11 | Supported | Results RQ1 |
| Fixed-prior additions monotonically degrade Test MAE | C12 | Supported | Results RQ2 |
| Fixed priors can amplify portfolio-specific overfitting | C12 + AE profile | Plausible scoped interpretation | Discussion |
| A1 improves operational value | AE-Gate-1 pending | Not yet supported | Experiment plan only |
| A1 generalizes to all VPPs | none | Forbidden | Nowhere |
| Physics-informed learning is generally harmful | none; contradicted by AE12/AE13 as broad claim | Forbidden | Nowhere |

## 7. Figure and Table Plan

| Item | Purpose | Status |
|---|---|---|
| Fig. 1 VPP signed net-power composition | Explain `Load - PV - Wind + Battery` and error covariance | Can draft now |
| Fig. 2 Component-token separated architecture | Show 5 component tokens + 3 weather/context tokens + inverted attention | Can draft now |
| Fig. 3 A1-A5 Val/Test divergence | Visualize C12 monotonic overfitting chain | Can draft now from ARA numbers |
| Fig. 4 Operational proxy comparison | Show ramp/peak/deviation/reserve results | Blocked until AE-Gate-1 |
| Table 1 Dataset and split | Applied Energy reproducibility baseline | Needs dataset details checked |
| Table 2 Main performance | A1 vs c23 vs strong baselines | Partially blocked by baseline closure |
| Table 3 Fixed-prior ablation | A1-A5 | Can draft now |
| Table 4 Operational metrics | AE-Gate-1 | Blocked |
| Table 5 Limitations / robustness | seeds, split, adaptation | Partially blocked |

## 8. AE-Gate-1 Experiment TODO

Minimum experiment package:

```text
Compare:
- A1
- c23
- persistence / naive
- DLinear or NLinear
- PatchTST or TFT
- GRU/LSTM

Report:
- MAE / MSE / RMSE
- ramp-event MAE
- peak/valley MAE
- deviation penalty proxy
- reserve requirement proxy
- mean ± std over seeds
```

Pass condition:

- A1 remains competitive or leading on aggregate MAE/RMSE.
- A1 is not worse than c23 and strong baselines on at least two operational proxies.
- C12 fixed-prior caution is not reversed by operational proxies.

Fail condition:

- simple baselines approach or beat A1 on aggregate metrics；
- A1 wins only average MAE but loses operational proxies；
- fixed-prior variants perform worse on MAE but better on operational proxies in a way that changes the paper's mainline。

If AE-Gate-1 fails, manuscript mainline must pivot from “robust Applied Energy contribution” to “architecture diagnostic / cautionary result,” and Applied Energy fit becomes weaker.

## 9. Immediate ARS Use

下一次调用 ARS 时应使用：

```text
ars-outline

Use:
- docs/analysis/applied_energy_ars_writing_adapter.md
- docs/analysis/applied_energy_ars_outline_blueprint.md
- docs/analysis/applied_energy_paper_source_pack.md

Generate a section-level Applied Energy outline and keep all full-draft prose blocked until AE-Gate-1.
```

当前不使用 `ars-full`。
