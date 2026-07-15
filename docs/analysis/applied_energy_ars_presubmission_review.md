# Applied Energy ARS Pre-submission Review

日期：2026-06-01

模式：方案 A 试错。使用 ARS `academic-paper-reviewer` 的多视角结构，但由于当前没有完整 manuscript，本报告是 `pre-submission evidence review`，不是逐段 peer review。

输入：

- `docs/analysis/applied_energy_journal_profile.md`
- `docs/analysis/applied_energy_paper_source_pack.md`
- `docs/analysis/applied_energy_pdf_extraction_matrix.md`

## Phase 0: Reviewer Configuration

| Reviewer | Persona | Focus |
|---|---|---|
| EIC | Applied Energy handling editor | Journal fit, energy-system significance, desk-reject risk |
| R1 Methodology | Energy forecasting / time-series ML reviewer | Baselines, metrics, splits, reproducibility |
| R2 Domain | VPP / DER operation reviewer | VPP relevance, dispatch/market/ramping value |
| R3 Cross-disciplinary | Power systems + ML generalization reviewer | Whether negative fixed-prior result is publishable |
| Devil's Advocate | Skeptical reviewer | Strongest counter-argument and overclaim detection |

## Phase 1: Independent Reviews

### EIC Review

Decision tendency: **Not ready for submission; potentially suitable after focused pre-submission revision.**

The evidence package has a plausible Applied Energy angle because it is not merely another Transformer variant. The strongest contribution is the discovery that VPP aggregate forecasting is sensitive to component representation and error-covariance structure. The C11 result is quantitatively strong: an 8-token component-separated model improves Test MAE by 12.5% over the full PhysFormer c23 baseline and reduces cross-seed MAE variance by 20x.

However, the current package would likely face desk-reject risk if submitted as a forecasting paper without operational linkage. The Applied Energy profile from AE01-AE15 shows repeated emphasis on market participation, dispatch, reserve, ramping, risk, uncertainty, and cost. Current evidence is dominated by MAE/MSE/RMSE and architecture ablations. That is enough for a strong ML venue story, but not yet enough for a safe Applied Energy story.

Required before submission:

- Add operational metrics: ramp/peak/deviation/reserve proxy at minimum.
- Reframe title/abstract away from “physics-guided Transformer” and toward VPP aggregate forecasting under heterogeneous resource coupling.
- Avoid general anti-physics claims. C12 should be written as fixed-prior overfitting in this VPP setting.

### R1 Methodology Review

Decision tendency: **Major methodological gaps before journal submission.**

Strengths:

- Multi-seed evidence is unusually clean for the main architecture claim.
- C11/C12 form a coherent positive-negative ablation chain.
- A1 vs A2-A5 isolates the effect of fixed-prior additions better than many forecasting papers isolate architectural complexity.

Major concerns:

1. **Baseline closure is incomplete for Applied Energy.** The source pack names c23 full PhysFormer and mentions historical Informer/iTransformer/LSTM baselines, but the current A1 contribution needs comparison against current strong simple and modern time-series baselines under the same split. At minimum: persistence/naive, DLinear/NLinear, PatchTST or TFT, and GRU/LSTM.

2. **Operational metrics are not yet integrated into the primary C11 table.** C09 has net_ramp_violation evidence for detach/c23, but A1's primary claim currently reports MAE/MSE/RMSE. Since Applied Energy papers often report ramping, reserve, cost, risk, CRPS, or scheduling value, this is a methodological mismatch.

3. **Single split / single dataset risk remains.** The evidence is strong internally, but external validity is underdeveloped. A multi-portfolio or target-adaptation test would substantially improve defensibility.

Recommended methodological revision:

- Build one consolidated benchmark table: A1, c23, persistence, DLinear/NLinear, PatchTST/TFT, GRU/LSTM, iTransformer.
- Report aggregate MAE/MSE/RMSE plus ramp/peak/deviation proxy for every baseline.
- Keep mean ± std across seeds as a required format.

### R2 Domain Review

Decision tendency: **Promising VPP contribution, but current operational relevance is under-specified.**

The paper has a real VPP-specific insight: aggregate net power is a signed composition of heterogeneous components. This matters for VPP operation because dispatch, market declaration, and reserve planning depend on aggregate net-power trajectories. The C08/C11 mechanism is domain-relevant: component errors are not independent, and signed cancellation can hide poor component modeling.

The current weakness is that the domain consequence is asserted more than measured. Applied Energy VPP papers such as AE04/AE05 connect forecast error to spot market deviation, energy storage reserve allocation, flexible ramping products, and risk levels. For this manuscript, the reader will ask: if A1 improves MAE, does it reduce ramping stress, reserve requirement, imbalance penalty, or peak/valley error?

Minimum domain additions:

- `ramp_event_MAE`: error during high net-load ramp intervals.
- `peak_valley_MAE`: error at operationally critical high/low net load periods.
- `deviation_penalty_proxy`: asymmetric or thresholded penalty for market deviation.
- `reserve_requirement_proxy`: quantile or max-error based reserve margin.

Without one of these, the work risks being seen as energy-flavored ML rather than Applied Energy.

### R3 Cross-disciplinary Review

Decision tendency: **The negative result is publishable if framed precisely.**

The most interesting part of the evidence is not simply that A1 performs best. The interesting part is the monotonic validation-test divergence in C12: each added fixed prior improves validation MSE but worsens Test MAE. This is potentially valuable to Applied Energy because the field often assumes more physics guidance improves generalization.

But the claim must be narrowed:

- Strong version that is not supported: physics priors are harmful.
- Defensible version: fixed, hand-specified priors can overfit portfolio-specific coupling in heterogeneous VPP aggregate forecasting.
- Stronger defensible version after more evidence: component-token separation is a more robust inductive bias than fixed prior tokens/graph biases for this dataset family.

The journal profile contains helpful precedents. AE09 shows Applied Energy can accept a result where Transformer is not automatically best. AE12/AE13 show physics-informed methods can work when the physical constraint is direct and validated. Therefore, the manuscript should position itself as a boundary-condition paper: physics guidance needs validation against portfolio heterogeneity and test generalization.

### Devil's Advocate Review

Strongest counter-argument:

The current story may be a well-executed model-selection study on one private dataset, not yet an Applied Energy contribution. The A1 result may simply show that the added physics-prior variants were over-parameterized or poorly tuned, while the simple model regularized better. Without strong external validation, operational metrics, and independent baselines, reviewers may not accept the broader conclusion that fixed priors amplify overfitting in VPP forecasting. The monotonic A1-A5 chain is interesting, but it may be interpreted as a local architecture search artifact rather than a general energy-systems insight.

Critical issues:

| Severity | Issue | Why it matters | Fix |
|---|---|---|---|
| CRITICAL | No Applied Energy-style operational value metric in the primary A1 evidence. | Forecasting-only MAE gains may be insufficient for Applied Energy. | Add ramp/peak/deviation/reserve/cost proxy for A1 and baselines. |
| CRITICAL | Baseline suite is not yet aligned with current forecasting standards. | Reviewers may argue A1 only beats a weak or internally designed baseline. | Add persistence, DLinear/NLinear, PatchTST/TFT, GRU/LSTM under same split. |
| MAJOR | Single private dataset / split family. | Generalization claim is vulnerable. | Add multi-portfolio or target-adaptation evaluation; otherwise explicitly limit scope. |
| MAJOR | Risk of overclaiming C12. | Applied Energy has successful physics-informed forecasting papers. | State fixed-prior overfitting as setting-specific; do not attack physics-informed learning generally. |
| MINOR | Current framing still carries legacy “PhysFormer physics-guided” naming. | It conflicts with C11/C12 evidence. | Retitle and restructure around component-token separation. |

## Phase 2: Editorial Synthesis

### Consensus Findings

All reviewers agree:

1. The best current paper mainline is component-token separation for VPP aggregate net-power forecasting.
2. C11/C12 are stronger and more Applied Energy-relevant than the older “physics-guided Transformer improves physical consistency” story.
3. Current evidence is not yet safe for immediate Applied Energy submission.
4. The most urgent missing piece is operational value measurement.
5. Baseline coverage must be strengthened before writing a submission manuscript.
6. C12 is valuable only if framed as a setting-specific caution about fixed priors, not as a general anti-physics claim.

### Editorial Decision

Current decision if submitted now: **Reject / Desk-reject risk high**.

Decision after focused pre-submission revision: **Potentially competitive, likely Major Revision if manuscript quality is strong**.

### Required Revision Roadmap

Priority 1: Operational value table

- Add A1 vs c23 vs core baselines on:
  - MAE/MSE/RMSE;
  - ramp error or ramp violation;
  - peak/valley MAE;
  - deviation penalty proxy or reserve requirement proxy.

Priority 2: Baseline closure

- Add at least:
  - persistence/naive;
  - DLinear or NLinear;
  - PatchTST or TFT;
  - GRU/LSTM;
  - current iTransformer if not already represented cleanly.

Priority 3: External validity defense

- Preferred: multi-portfolio / target-portfolio few-shot adaptation.
- Acceptable fallback: explicit limitation + data-scarce or split-stability stress test.

Priority 4: Manuscript framing

- Use title/abstract around VPP aggregate net-power forecasting, component-token separation, and heterogeneous resource coupling.
- Do not lead with “physics-guided Transformer”.
- Use C12 as cautionary result: fixed priors can overfit under heterogeneous portfolio coupling.

## Scheme A Trial Outcome

方案 A 的试错结论：

1. **可以配合 ARA 使用。** `journal_profile + paper_source_pack` 能把 ARA 的实验事实转成期刊审稿问题，边界清楚。
2. **痛点已经暴露。** Applied Energy 缺口不是“不会写”，而是当前实验包缺少 operational metric 和强 baseline closure。
3. **ARS 作为 reviewer layer 有价值。** 它能把 C11/C12 从“实验结果”转成“投稿风险和补实验列表”。
4. **正式 ARA-ARS bridge 暂时不必做复杂。** 目前一个 source pack 足够暴露问题；等补实验后再自动化更合理。
5. **下一步不应直接写全文。** 应先补一个最小 Applied Energy experiment patch：operational metrics + baseline closure。

## Next Recommended Action

建议下一步做一个小而硬的 Applied Energy 补实验包：

```text
AE-Gate-1:
Compare A1, c23, persistence, DLinear/NLinear, PatchTST/TFT, GRU/LSTM on:
- MAE / MSE / RMSE
- ramp-event MAE
- peak/valley MAE
- deviation penalty proxy
- reserve requirement proxy
Report mean ± std over seeds.
```

通过条件：

- A1 仍在 aggregate MAE/RMSE 上领先；
- A1 在至少 2 个 operational proxy 上不劣于 c23 和强 baseline；
- C12 的 fixed-prior overfitting 叙事不被 operational metrics 反转。

失败条件：

- A1 只在普通 MAE 上领先，但 operational proxy 弱；
- simple baselines 接近或超过 A1；
- operational metrics 显示 fixed priors 虽然 MAE 差但调度价值更好。

