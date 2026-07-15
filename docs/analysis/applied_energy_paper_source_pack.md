# Applied Energy Paper Source Pack for ARS Plan A

日期：2026-06-01

用途：这是方案 A 的 ARA-derived source pack。它只承载可追溯事实、证据边界和禁用声明，供 ARS reviewer / paper writer 使用。ARS 只能基于本文件和 `applied_energy_journal_profile.md` 做写作或审稿判断，不能自行补事实。

输入来源：

- `ara/logic/problem.md`
- `ara/logic/claims.md`
- `ara/evidence/tables/version_comparison.md`
- `ara/evidence/tables/v5_component_metrics.md`
- `docs/analysis/applied_energy_journal_profile.md`
- `docs/analysis/applied_energy_pdf_extraction_matrix.md`

## Target Journal Constraint

目标期刊：`Applied Energy`

期刊画像结论：

- Applied Energy 的 forecasting 论文通常要求能源系统问题先于模型问题。
- MAE/MSE/RMSE 是基础指标，但通常需要连接到 scheduling、dispatch、market、reserve、ramping、risk 或 cost。
- Transformer 不是天然卖点，必须证明 architecture 与能源系统结构对齐。
- Physics-informed 叙事必须保守。不能把单一数据集上的 fixed-prior failure 扩展成 physics-informed learning 的一般否定。
- 单一私有 VPP 数据集存在外部有效性风险，需要多 seed、多 portfolio、target adaptation 或 operational analysis 缓解。

## Proposed Manuscript Mainline

推荐主线：

> In VPP aggregate net-power forecasting, component-token separation improves generalization stability, while fixed physics-prior additions can amplify validation-test overfitting under heterogeneous portfolio coupling.

推荐标题方向：

> Robust aggregate net-power forecasting for virtual power plants via component-token separation under heterogeneous resource coupling

备选标题：

> When fixed physics priors overfit: component-token separation for virtual power plant aggregate forecasting

## Problem Facts

| ID | Fact | Source | Paper Use |
|---|---|---|---|
| PF01 | VPP net power is a signed composite: `net = Load - PV - Wind + Battery`. | `ara/logic/problem.md:O2` | Problem definition and component coupling formulation. |
| PF02 | Components have heterogeneous dynamics: Load is behavior-driven; PV/Wind weather-driven; Battery control-driven. | `ara/logic/problem.md:O2` | Justifies component-aware representation. |
| PF03 | Physics equations exist for PV/Wind/Battery but are incomplete and should be used as soft priors, not hard guarantees. | `ara/logic/problem.md:O3` | Supports cautious physics-informed framing. |
| PF04 | Load remains the hardest component; historical evidence shows Load/Wind error ratio converging near 6x even after behavioral modeling. | `ara/logic/claims.md:C06` | Explains component heterogeneity and residual difficulty. |

## Core ARA Claims

| Claim | Status | Evidence | Manuscript Role |
|---|---|---|---|
| C08 | supported | Component error covariance determines aggregate accuracy through signed summation cancellation. Worse component MAE can still yield better aggregate accuracy if errors cancel. | Mechanistic foundation: aggregate accuracy is not equivalent to physical component correctness. |
| C09 | supported | Selective detach improves c23 aggregate metrics across 3 seeds: MAE 1.973e-3 ± 3.7e-5 vs baseline 2.069e-3 ± 1.34e-4; net_ramp_violation also lower. | Secondary evidence that gradient-path design affects aggregate robustness. |
| C10 | supported | Detach disables encoder-depth to cancellation channel; detach x e3 does not improve over detach. | Mechanistic bridge from shared encoder to component-token separation. |
| C11 | supported | 8-token pure inverted Transformer beats full PhysFormer c23 baseline across 3 seeds: MAE 1.811e-3 ± 6e-6 vs 2.069e-3 ± 1.34e-4, 12.5% better; cross-seed std is 20x smaller. | Primary positive contribution. |
| C12 | supported | Fixed prior additions degrade Test MAE monotonically despite improving Val MSE: A1 0.001811; A2 0.001819; A3 0.001843; A4 0.001863; A5 0.001947. | Primary negative/cautionary contribution. |

## Quantitative Evidence Table

### Best Current Architecture vs Full PhysFormer Baseline

| Metric | A1 8-token iTransformer | c23 full PhysFormer baseline | Delta |
|---|---:|---:|---:|
| Test MAE (MW) | 1.811e-3 ± 6e-6 | 2.069e-3 ± 1.34e-4 | -12.5% |
| Test MSE (MW²) | 6.766e-6 ± 4.9e-8 | 8.111e-6 ± 6.75e-7 | -16.6% |
| Test RMSE (MW) | 2.601e-3 ± 9e-6 | 2.846e-3 ± 1.17e-4 | -8.6% |
| Cross-seed MAE std | 6e-6 | 1.34e-4 | 20x lower |

Source: `ara/logic/claims.md:C11`

### Fixed-prior Addition Chain

| Variant | Addition | 3-seed Test MAE | vs A1 | Val/Test Pattern |
|---|---|---:|---:|---|
| A1 | 8 component/weather tokens, simple FFN decoder, no physics prior | 0.001811 | baseline | best Test |
| A2 | + physics token | 0.001819 | +0.4% | Val improves, Test worsens |
| A3 | + twin/constraint tokens | 0.001843 | +1.8% | Val improves more, Test worsens more |
| A4 | + graph bias | 0.001863 | +2.9% | Val improves more, Test worsens more |
| A5 | + horizon decoder + weather conditioning | 0.001947 | +7.5% | best Val, worst Test |

Source: `ara/logic/claims.md:C12`

### Historical Physics-guided Context

| Evidence | Use |
|---|---|
| V4 component loss reduced Theory MAE by 21.8% vs V3 but aggregate MAE degraded 2.3%. | Physics priors can improve physical consistency, but trade off with aggregate accuracy. |
| Stronger component supervision improved Theory MAE but degraded aggregate MAE. | Supports Pareto-frontier framing, not monotonic physics-help framing. |
| V5 improved PV/Wind/Battery component metrics but aggregate MAE degraded. | Supports component-aggregate paradox and C08. |

Sources: `ara/evidence/tables/version_comparison.md`, `ara/evidence/tables/v5_component_metrics.md`

## Safe Claims

These can be used in an Applied Energy manuscript if tied to the evidence above:

1. VPP aggregate net-power forecasting is a signed multi-component problem, not a scalar time-series problem.
2. Component-token separation via inverted attention improves aggregate accuracy and dramatically reduces cross-seed variance compared with the full PhysFormer c23 baseline.
3. Aggregate accuracy depends on component-error covariance and signed cancellation, so component-level physical consistency and aggregate accuracy can diverge.
4. In this dataset and architecture family, fixed prior additions improved validation loss but degraded test MAE monotonically.
5. The evidence suggests fixed priors should be data-adaptively validated in heterogeneous VPP settings.

## Do-not-claim Boundaries

Do not claim:

- Physics-informed learning is generally ineffective.
- Physics priors hurt forecasting in general.
- Pure data-driven models are always better.
- A1 is globally optimal beyond this dataset and architecture family.
- Current evidence proves operational value in market bidding, dispatch, reserve, or cost terms.
- Current evidence proves probabilistic forecasting quality.
- Current evidence eliminates single-dataset / private-data external validity risk.

## Known Gaps for Applied Energy

| Gap | Current State | Why It Matters |
|---|---|---|
| Operational metric | C09 has net_ramp_violation for detach/c23, but C11 A1 evidence does not yet foreground ramp/peak/deviation/reserve metrics. | Applied Energy expects forecasting accuracy to connect to operation, market, risk, or cost. |
| Strong simple baselines | ARA mentions Informer/iTransformer/LSTM and c23 baseline; Applied Energy profile expects persistence, DLinear/NLinear, PatchTST/TFT or equivalent. | Prevents reviewer claim that gains come from weak baselines. |
| External validity | Evidence is primarily one VPP dataset / current split family. | Applied Energy may challenge single private dataset. |
| Probabilistic uncertainty | Current mainline is deterministic MAE/MSE/RMSE. | AE11/AE14/AE15 show probabilistic/scenario analysis is valued. |
| Manuscript-specific evidence trace | No current Applied Energy manuscript draft exists. | ARS can only run pre-submission evidence review, not line-by-line peer review. |

## Recommended Experiment Add-ons Before Writing

Priority order:

1. Operational metrics for A1 vs c23 baseline:
   - ramp error / ramp violation;
   - peak-hour and valley-hour MAE;
   - deviation penalty proxy;
   - reserve requirement proxy.
2. Strong baseline closure:
   - persistence / naive;
   - DLinear or NLinear;
   - PatchTST or TFT;
   - GRU/LSTM if not already comparable under current split.
3. External validity stress:
   - multi-portfolio split;
   - target-portfolio few-shot adaptation;
   - data-scarce scenario.

## ARS Reviewer Prompt Seed

```text
Mode: pre-submission evidence review, not full manuscript review.
Target journal: Applied Energy.

Inputs:
- Applied Energy journal_profile v0.1.
- ARA paper_source_pack.

Review target:
Assess whether the current PhysFormer evidence package can support an Applied Energy submission.

Hard evidence constraints:
- Use only source_pack facts for research claims.
- Do not claim physics priors are generally harmful.
- Treat fixed-prior overfitting as setting-specific.
- Identify missing experiments before writing.

Reviewer tasks:
1. Estimate desk-reject risk.
2. Identify missing operational metrics.
3. Identify baseline gaps.
4. Evaluate whether single private VPP data is defensible.
5. Recommend safest manuscript framing.
```

