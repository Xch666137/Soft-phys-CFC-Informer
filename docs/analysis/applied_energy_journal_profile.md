# Applied Energy 期刊画像 v0.1

日期：2026-06-01

依据：AE01-AE15 共 15 篇 `Applied Energy` 论文 PDF 的本地文本抽取与初步蒸馏。

本文档面向 PhysFormer 投稿判断和后续 ARS reviewer simulation。它不是普通 related work，而是一个投稿画像：Applied Energy 期待什么样的能源系统问题、实验强度、指标组合和叙事方式。

## 一句话判断

PhysFormer 有进入 Applied Energy 的潜力，但必须从“模型精度论文”升级成“VPP aggregate forecasting 的能源系统泛化问题论文”。

当前最安全主线：

> In VPP aggregate net-power forecasting, component-token separation improves generalization stability, while fixed physics-prior additions can amplify validation-test overfitting under heterogeneous portfolio coupling.

不安全主线：

> A physics-guided Transformer improves VPP forecasting because adding more physics priors helps.

不安全原因是 ARA 的 C12 已经证明固定 physics-prior additions 在当前实验中单调恶化 Test MAE。Applied Energy 中确实存在 physics-informed 成功论文，因此我们不能泛化为“物理先验无效”，只能说“未经数据适配的固定先验在 VPP aggregate setting 中可能过拟合”。

## Applied Energy 的投稿口味

### 1. 能源系统问题必须先于模型问题

15 篇文章的共同写法是先定义能源系统痛点，再引出模型：

- VPP operation 受到 stochastic resource、heterogeneous information、market interaction 影响。
- Net load forecasting 服务 optimal generation scheduling 和 flexibility requirement estimation。
- Renewable forecasting 支撑 reserve allocation、market participation、risk management。
- Probabilistic forecasting 支撑 uncertainty-aware optimization、unit commitment、economic dispatch。

对 PhysFormer 的要求：

- 引言不能从 Transformer 或 physics-guided learning 开始。
- 应从 VPP 聚合净功率预测的 operation consequences 开始：reserve、ramp、imbalance、dispatch、market bidding、resource coordination。

### 2. MAE/MSE 是基础指标，不是完整贡献

核心文献中 MAE、RMSE、MAPE、NRMSE、NMAE 很常见，但 Applied Energy 更重视这些精度指标如何转化为系统价值。

常见增强指标：

- ramping / peak-valley error；
- reserve 或 flexibility requirement；
- profit / operating cost；
- trading deviation penalty；
- CVaR / risk level；
- CRPS / pinball loss / reliability / skill score；
- scenario generation quality；
- dispatch or scheduling cost。

对 PhysFormer 的要求：

- 保留 MAE/MSE/RMSE。
- 必须补至少一组 operational metric，优先级如下：
  1. `ramp / peak-valley error`：最贴近已有 net_ramp_violation 和 AE05/AE06。
  2. `imbalance or deviation penalty proxy`：贴近 AE04 spot-market 逻辑。
  3. `reserve requirement proxy`：预测误差转化为 reserve margin。
  4. `dispatch cost toy model`：最强，但实现成本更高。

### 3. Baseline 需要覆盖“简单、传统、深度、Transformer”

Applied Energy 论文常见 baseline 不是只和同类 Transformer 比。可接受的组合一般包括：

- naive / persistence；
- statistical：ARIMA、Grey、MCMC/copula 等；
- classical ML：SVR、ANN、BP；
- recurrent DL：LSTM、GRU；
- modern DL/forecasting：Informer、Transformer、TFT、DeepAR、DLinear/NLinear；
- domain-specific 或 physics-informed variants。

对 PhysFormer 的最低要求：

- 已有 Informer / iTransformer / LSTM 还不够稳。
- 建议补 `persistence/naive`、`GRU/LSTM`、`DLinear/NLinear`、`PatchTST or TFT`、`iTransformer`。
- 如果资源有限，至少补 `persistence + DLinear/NLinear + PatchTST/TFT`，因为这些能覆盖“简单强基线”和“近年时序预测强基线”。

### 4. Transformer 不是天然卖点，适用性才是卖点

AE09 明确显示 Transformer 在某些 wind forecasting 设定中低于 GRU；AE10/AE08 则通过结构设计和实验说明 Transformer 在特定 spatio-temporal/multimodal 场景下有效。

对 PhysFormer 的含义：

- 不应说“我们用了 Transformer，所以更先进”。
- 应说“VPP component coupling 导致 shared encoder cancellation channel；component-token separation 是与问题结构对齐的 architecture choice”。
- C11 是正面贡献：8-token inverted Transformer 通过 component-token separation 消除 shared-encoder cancellation channel。
- C12 是负面贡献：固定 physics priors 在 validation 上更好但 Test 上更差，说明复杂先验可能过拟合 portfolio-specific patterns。

### 5. Physics-informed 叙事必须保守

AE12、AE13 说明 Applied Energy 接受 physics-informed/physics-constrained forecasting，但这些论文通常满足：

- 物理机制清晰，例如 wind power curve、extreme event physical expression；
- 约束与预测目标直接相关；
- 通过 reliability、skill score、noise robustness、extreme event 等指标证明收益。

对 PhysFormer 的含义：

- 不能把 C12 写成“physics-informed 方法无效”。
- 应写成：
  - fixed priors are not automatically beneficial；
  - physics priors need data-adaptive validation；
  - in heterogeneous VPP aggregate forecasting, component representation separation was more robust than hand-specified prior additions。

### 6. 单数据集风险真实存在

多篇论文使用多站点、多区域、多 horizon、多场景、极端天气或 data-scarce setting 来支撑泛化。Applied Energy 对单一私有数据集会有天然疑问。

对 PhysFormer 的缓解策略：

- 多 seed 必须保留。
- 多 portfolio / target-portfolio adaptation 比单一时间 split 更有说服力。
- 如果不能公开数据，需要更强的 reproducibility package：模型配置、split 描述、指标脚本、完整消融表。
- 可以把 B1-R2 few-shot target adaptation 作为 Applied Energy 补强实验，而不是作为旁支。

## 对 PhysFormer 的推荐论文主线

### 推荐标题方向

保守版：

> Component-token separation for robust virtual power plant aggregate net-power forecasting

更有冲击力版：

> When fixed physics priors overfit: component-token separation for virtual power plant aggregate forecasting

Applied Energy 风格版：

> Robust aggregate net-power forecasting for virtual power plants via component-token separation under heterogeneous resource coupling

### 推荐贡献点

1. 提出 VPP aggregate net-power forecasting 中的 component-error coupling 问题：aggregate accuracy 不只由 component MAE 决定，还受 signed error covariance 和 cancellation 影响。
2. 证明 shared encoder physics-guided architecture 存在 cancellation channel，导致更复杂的先验可能改善 validation 但恶化 Test。
3. 提出 8-token component-separated inverted Transformer，在三 seed 上优于 full PhysFormer c23 baseline，并显著降低 cross-seed variance。
4. 给出固定 physics prior 的系统性负结果：physics token、twin/constraint token、graph bias、horizon decoder/weather conditioning 均单调恶化 Test MAE。
5. 用 ramp/peak/deviation/reserve proxy 说明该结构改进对 VPP operation 有实际意义。

### 推荐摘要逻辑

1. VPP aggregate forecasting 对 dispatch、reserve、market participation 关键。
2. VPP net power 由 load、PV、wind、battery signed composition 构成，component errors 会通过 covariance/cancellation 影响 aggregate prediction。
3. 现有 physics-guided Transformer 容易把 fixed priors 和 shared encoder coupling 混在一起，导致 validation/test divergence。
4. 本文提出 component-token separation，把 heterogeneous components 放入独立 token representation。
5. 实验显示 simple component-separated model 比复杂 physics-guided baseline 更稳；固定先验 additions 反而放大 overfitting。
6. 结果提示：VPP forecasting 中更重要的是 representation separation 和 data-adaptive coupling，而不是静态叠加更多先验。

## 实验缺口清单

### 必补

| 缺口 | 原因 | 建议实现 |
|---|---|---|
| Operational metric | Applied Energy 预测论文通常连接 dispatch/market/risk/cost。 | 先做 ramp/peak/deviation penalty proxy，成本最低。 |
| Strong simple baselines | 需要证明不是复杂模型压过弱 baseline。 | persistence、DLinear/NLinear、GRU/LSTM、PatchTST/TFT。 |
| 多 horizon 或 peak/valley 分析 | AE06/AE07 强调 net-load 的 peak/valley 和 scheduling value。 | 按 horizon、peak hours、ramp events 分组报错。 |
| 多 seed 表格 | C11/C12 已有优势，必须显式展示。 | 保留 mean ± std，并突出 A1 20x lower variance。 |
| 单数据集风险说明 | Applied Energy 可能质疑 private VPP generalization。 | 用 multi-portfolio split 或 B1-R2 target adaptation 缓解。 |

### 可选但加分

| 加分项 | 价值 | 成本 |
|---|---|---|
| Probabilistic extension | 对齐 AE11/AE14/AE15。 | 中高。 |
| Scenario generation | 强 Applied Energy 风格。 | 高。 |
| Dispatch cost simulation | 最能证明 operational value。 | 中高。 |
| Extreme/weather subset analysis | 对齐 AE12/AE14。 | 中。 |
| Data-scarce adaptation | 对齐 AE01，并服务 DVPP 部署叙事。 | 中。 |

## C12 的安全写法

可写：

- Fixed physics priors are not automatically beneficial in heterogeneous VPP aggregate forecasting.
- Stronger validation performance can hide poorer test generalization when priors encode portfolio-specific coupling.
- Component-token separation provided a more robust inductive bias than hand-specified physics tokens or graph biases in our setting.

不要写：

- Physics-informed learning is ineffective.
- Physics priors hurt forecasting in general.
- Pure data-driven models are always better.

原因：

- AE12/AE13 是 Applied Energy 内部的 physics-informed 成功样例。
- 我们的证据只支持“当前 VPP aggregate setting + fixed prior additions + 当前数据 split 下过拟合”，不支持跨领域否定。

## ARS Reviewer Mode 输入建议

后续给 ARS 的 reviewer prompt 应包括：

```text
Target journal: Applied Energy
Manuscript claim:
Component-token separation improves VPP aggregate net-power forecasting generalization, while fixed physics-prior additions can amplify validation-test overfitting.

Evidence constraints:
- Use ARA C11/C12 as primary claims.
- Do not claim physics priors are generally harmful.
- Treat ARS as reviewer/writing layer only; ARA remains source of facts.

Review tasks:
1. Identify Applied Energy desk-reject risks.
2. Identify missing operational metrics.
3. Identify unavoidable baselines.
4. Judge whether single private VPP data is acceptable.
5. Suggest safest title/abstract framing.
```

## 当前投稿适配性判断

适配潜力：中高。

当前不宜直接投：还缺 Applied Energy 读者期待的 operational value 和强 baseline closure。

最小补强后可进入写作：

1. 增加 ramp/peak/deviation proxy 指标。
2. 补 persistence + DLinear/NLinear + PatchTST/TFT 或同等级强基线。
3. 用 ARA C11/C12 改写主线，不再以“physics-guided Transformer”作为核心卖点。
4. 准备 single-dataset limitation 的防御：multi-seed、multi-portfolio 或 target adaptation。

