# Applied Energy ARS 写作适配器 v0.1

日期：2026-06-01

用途：把 AE01-AE15 的 Applied Energy 期刊画像蒸馏成 ARS `academic-paper` 写作流程的目标期刊适配层。这个文件不是审稿报告，也不是普通文献综述；它是给 ARS 用的写作控制面板，用来约束选题主线、章节结构、证据边界、引用策略和“何时允许写完整稿”。

## 一句话定位

我们要做的不是“用 15 篇 Applied Energy 训练 ARS 变成另一个 reviewer”，而是：

> 用 15 篇 Applied Energy 核心文章蒸馏出目标期刊的写作口味、证据门槛和叙事结构，再把 ARA 中已经证实的实验结论转化为一篇 Applied Energy 风格的 VPP aggregate forecasting 论文。

适配后的 ARS 分工：

- **ARA 是事实层**：只提供已经证实、可追溯的研究事实、实验结果、claim 边界和 forbidden claims。
- **Applied Energy 15 篇论文是期刊风格层**：提供目标期刊偏好的问题框架、baseline 组合、operational metrics、figure/table 习惯、审稿风险。
- **ARS 是写作层**：负责 outline、argument blueprint、section draft、abstract、citation audit 和 self-review，但不能自行补事实。

## 输入文件

ARS 写作时必须同时读取以下输入：

| 输入 | 作用 |
|---|---|
| `docs/analysis/applied_energy_literature_screen.md` | AE01-AE15 的题名、DOI、选择理由和候补文献池 |
| `docs/analysis/applied_energy_pdf_extraction_matrix.md` | 15 篇 PDF 的逐篇蒸馏矩阵 |
| `docs/analysis/applied_energy_journal_profile.md` | Applied Energy 期刊画像和 PhysFormer 投稿判断 |
| `docs/analysis/applied_energy_paper_source_pack.md` | ARA-derived 论文事实包、safe claims、do-not-claim boundaries |
| `ara/logic/claims.md` | C08-C12 等核心实验声明 |
| `ara/trace/exploration_tree.yaml` | 研究路径、dead ends、已提交决策 |

禁止 ARS 直接依据记忆、常识或未读文献扩展研究事实。任何实验数字必须能回指到 source pack 或 ARA。

## 适配后的主线

推荐主线：

> In VPP aggregate net-power forecasting, component-token separation improves generalization stability, while fixed physics-prior additions can amplify validation-test overfitting under heterogeneous portfolio coupling.

中文解释：

在虚拟电厂聚合净功率预测中，核心问题不是“Transformer 是否更先进”，也不是“物理先验越多越好”，而是异构资源组合下的 component error coupling。当前证据支持：component-token separation 是更稳健的结构性归纳偏置；未经数据适配的 fixed physics priors 可能把 portfolio-specific coupling 学成过拟合路径，表现为 Val MSE 更好但 Test MAE 更差。

不推荐主线：

> A physics-guided Transformer improves VPP forecasting because adding more physics priors helps.

这个主线与 C12 冲突，也不符合 Applied Energy 对 physics-informed 论述的审稿习惯。

## AE-Gate 写作闸门

### AE-Gate-0：现在已经满足

允许产出：

- Applied Energy 风格的 paper plan；
- detailed outline；
- argument blueprint；
- provisional abstract；
- missing-evidence checklist；
- reviewer-risk map。

依据：

- AE01-AE15 已完成 PDF 蒸馏；
- Applied Energy journal profile 已完成；
- ARA source pack 已完成；
- C08-C12 可作为当前事实层。

### AE-Gate-1：完整投稿稿之前必须补齐

没有通过 AE-Gate-1 时，ARS 不应写 full submission manuscript，只能写“预备稿/提纲/论证蓝图”。

最低证据要求：

| 缺口 | 最低要求 | 原因 |
|---|---|---|
| Operational metrics | A1 vs c23 vs core baselines 至少包含 ramp-event MAE、peak/valley MAE、deviation penalty proxy 或 reserve proxy 中的 2 类 | Applied Energy forecasting 文章通常要把 MAE/RMSE 连接到 dispatch、market、reserve、ramping、risk 或 cost |
| Strong baselines | persistence/naive、DLinear 或 NLinear、PatchTST 或 TFT、GRU/LSTM、当前 iTransformer/c23 | 避免审稿人认为 A1 只是击败了内部弱基线 |
| Multi-seed reporting | 关键表格使用 mean ± std | C11 的 20x lower variance 是强卖点，必须显式保留 |
| Single-dataset defense | multi-portfolio、target adaptation、data-scarce stress 或明确 limitation | 私有单数据集是 Applied Energy 的真实风险 |

AE-Gate-1 通过条件：

- A1 在 aggregate MAE/RMSE 上仍然领先或至少保持竞争力；
- A1 在至少 2 个 operational proxy 上不劣于 c23 和强 baseline；
- C12 的 fixed-prior overfitting 叙事没有被 operational metrics 反转；
- simple baselines 没有接近到让 component-token separation 的贡献失去意义。

### AE-Gate-2：正式投稿前建议满足

增强但不是绝对必须：

- target-portfolio few-shot adaptation；
- horizon-wise / ramp-event / peak-valley subgroup analysis；
- uncertainty 或 probabilistic limitation 说明；
- reproducibility package：config、split、metric script、ablation table。

## ARS academic-paper 阶段适配

### Phase 0：Intake Agent

原始 ARS 目标：确认 paper type、discipline、journal、citation format、output format、language、word count。

Applied Energy 适配：

| 字段 | 固定或推荐值 |
|---|---|
| Target journal | Applied Energy |
| Paper type | Research article |
| Field | Energy forecasting / virtual power plant / power systems ML |
| Citation style | Elsevier numbered style 或 Applied Energy guide 要求 |
| Language | English manuscript；中文讨论稿可作为中间产物 |
| Main contribution | Component-token separation for robust VPP aggregate net-power forecasting |
| Evidence boundary | Use only ARA C08-C12 and source pack facts |
| Full draft permission | Only after AE-Gate-1 |

Phase 0 必须输出一条硬约束：

> ARS may draft outlines and argument blueprints now. ARS must not draft a full Applied Energy submission manuscript until AE-Gate-1 evidence is available.

### Phase 1：Literature Strategist Agent

原始 ARS 目标：检索、筛选、构建 annotated bibliography 和 literature matrix。

Applied Energy 适配：

1. AE01-AE15 不是普通 related work，而是 style/lit seed。
2. 文献使用顺序应该服务于论文结构，而不是按模型类别堆砌。
3. 禁止编造 citation、DOI、结果数字。
4. 如果要补文献，优先从 `applied_energy_literature_screen.md` 的 AE16-AE27 候补池扩展。

AE01-AE15 在写作中的功能分组：

| 组别 | 文献 | 用途 |
|---|---|---|
| VPP operation / uncertainty | AE02, AE03 | 引言中建立 VPP operation、resource coordination、uncertainty 的系统背景 |
| VPP market / ramping | AE04, AE05 | 支撑 operational metrics：deviation penalty、reserve、flexible ramping |
| Net-load forecasting | AE06, AE07 | 支撑 `net = Load - PV - Wind + Battery` 和 peak/valley/ramp 重要性 |
| Transformer forecasting | AE08, AE09, AE10, AE11 | 说明 Transformer 必须靠结构适配和强 baseline 证明，不是天然优势 |
| Physics-informed forecasting | AE12, AE13 | 约束 C12 写法：不能否定 physics-informed learning，只能讨论 fixed-prior overfitting |
| Probabilistic / scenario / scheduling | AE14, AE15 | 支撑 limitation 和 future work：uncertainty、scenario、dispatch cost |
| Data-scarce VPP adaptation | AE01 | 支撑后续 B1-R2 / target adaptation 作为 Applied Energy 加强实验 |

### Phase 2：Structure Architect Agent

原始 ARS 目标：设计 paper structure、outline、word allocation、evidence map。

Applied Energy 推荐结构：

#### 1. Introduction

推荐 5 段式：

1. VPP aggregate net-power forecasting 对 dispatch、reserve、market participation、renewable integration 的重要性。
2. VPP net power 是 signed heterogeneous composition：`net = Load - PV - Wind + Battery`，component errors 会通过 covariance/cancellation 影响 aggregate accuracy。
3. 现有 energy forecasting 方法包括 statistical、DL、Transformer、physics-informed，但它们通常没有充分区分 component representation 与 fixed prior coupling。
4. 关键 gap：在 heterogeneous VPP portfolio 中，更多固定物理先验未必提升 test generalization；需要验证何种 inductive bias 真正稳健。
5. 本文贡献：component-token separation、C08 covariance mechanism、C11 multi-seed gain、C12 fixed-prior caution、operational metrics（AE-Gate-1 后加入）。

禁止开头：

- “Transformer has achieved great success...”
- “Physics-guided learning always improves forecasting...”
- “We propose a novel deep learning model...” 后面才讲能源问题。

#### 2. Related Work

推荐四条线：

1. VPP operation and uncertainty-aware forecasting；
2. Aggregate net-load / renewable forecasting and operational metrics；
3. Transformer and modern time-series forecasting in energy systems；
4. Physics-informed or physics-constrained forecasting and its boundary conditions。

写作目标不是“证明别人不行”，而是建立这个 gap：

> Existing work validates forecasting architectures or physics constraints in specific settings, but less attention has been paid to how component representation and fixed priors affect test generalization under heterogeneous VPP aggregate coupling.

#### 3. Problem Formulation

必须显式写：

```text
P_net(t) = P_load(t) - P_pv(t) - P_wind(t) + P_battery(t)
```

并解释：

- aggregate error 不是 component MAE 的简单相加；
- signed error covariance can cancel or amplify aggregate error；
- 因此 VPP aggregate forecasting 需要同时看 aggregate metrics、component structure、operational proxies。

这部分承接 C08，是整篇论文的理论入口。

#### 4. Method

方法部分推荐以 A1 作为主角，而不是旧 PhysFormer c23。

推荐命名：

- Component-separated inverted Transformer；
- Component-token separated forecasting model；
- 8-token component/weather inverted attention model。

不要把方法包装成“更强 physics-guided Transformer”。安全写法：

> The method uses component-token separation as a structural inductive bias. It does not rely on fixed physics-prior tokens, graph biases, or hand-specified component constraints in the final best-performing configuration.

需要说明：

- 5 个 component tokens：Load、PV、Wind、Battery power、Battery SOC；
- 3 个 weather/context tokens；
- inverted self-attention；
- shared FFN decoder；
- real-unit power balance；
- net MSE training objective；
- 与 c23 full PhysFormer 和 A2-A5 prior additions 的关系。

#### 5. Experiments

实验部分必须按 Applied Energy 审稿逻辑组织：

1. Dataset and VPP portfolio description；
2. Baselines；
3. Metrics；
4. Main comparison；
5. Ablation on fixed prior additions；
6. Operational-value analysis；
7. Robustness / seed / split / portfolio analysis；
8. Limitation and reproducibility。

核心表格建议：

| 表格 | 内容 |
|---|---|
| Table 1 | Dataset / VPP portfolio / horizon / split |
| Table 2 | A1 vs c23 vs strong baselines on MAE/MSE/RMSE, mean ± std |
| Table 3 | Operational metrics：ramp、peak/valley、deviation/reserve proxy |
| Table 4 | A1-A5 fixed-prior ablation chain with Val/Test contrast |
| Table 5 | Subgroup or adaptation analysis：portfolio/horizon/data-scarce |

#### 6. Discussion

讨论部分不要重复实验结果，而要回答 Applied Energy 读者的问题：

- 为什么 component-token separation 对 VPP aggregate forecasting 有能源系统意义？
- 为什么 fixed priors 在这个 setting 中可能过拟合？
- 这个结论和 Applied Energy 中成功的 physics-informed forecasting 文献如何不冲突？
- 对 VPP dispatch、reserve、market bidding、future data-adaptive physics design 有什么启示？
- 单数据集、deterministic forecasting、private data 的局限是什么？

推荐讨论句式：

> Our results do not suggest that physics-informed forecasting is ineffective. Rather, they indicate that hand-specified fixed priors should be validated against portfolio heterogeneity and out-of-sample aggregate behavior.

### Phase 3：Argument Builder Agent

原始 ARS 目标：构建 claim-evidence chain、logical flow、counter-argument handling。

Applied Energy claim-evidence map：

| Manuscript claim | Evidence source | Allowed strength |
|---|---|---|
| VPP aggregate net-power forecasting is a signed multi-component problem | ARA problem facts, C08 | Strong |
| Aggregate accuracy depends on signed component-error covariance/cancellation | C08 | Strong |
| Component-token separation improves aggregate accuracy and cross-seed stability | C11 | Strong within current dataset/split |
| Fixed prior additions improve validation but degrade test MAE monotonically | C12 | Strong within A1-A5 design space |
| Fixed priors can overfit heterogeneous portfolio coupling | C12 + AE profile | Plausible interpretation, must be scoped |
| A1 improves operational value | Not yet fully supported before AE-Gate-1 | Do not claim yet |
| Results generalize to all VPPs | Not supported | Forbidden |

必须处理的 counter-arguments：

1. A1 是否只是更简单所以 regularization 更强，而不是 component-token separation 真正有效？
2. A2-A5 是否只是没有调好，而不是 fixed priors 本身有问题？
3. 私有单数据集是否足以支撑 Applied Energy？
4. 没有 probabilistic forecasting 是否削弱 VPP operational relevance？
5. 为什么不用更强 baseline？

ARS 在论证时必须把这些质疑前置处理，不能等到 reviewer 阶段才发现。

### Phase 4：Draft Writer Agent

原始 ARS 目标：section-by-section drafting。

Applied Energy 适配规则：

写作风格：

- 先写 energy-system problem，再写 model；
- 句子紧凑、证据驱动，避免宣传式 novelty；
- 少用 “novel”, “powerful”, “significant breakthrough”；
- 多用 “we investigate”, “we show”, “the results suggest”, “within this setting”；
- 所有强结论都要落到表格或 ARA claim。

强制禁区：

- 不得声称 physics-informed learning 一般无效；
- 不得声称 pure data-driven models 总是更好；
- 不得声称 A1 是全局最优；
- 不得声称已证明 market profit、dispatch cost、reserve cost 降低，除非 AE-Gate-1/2 补了对应 metric；
- 不得把 c23/旧 PhysFormer 作为最终正向主角。

允许在 AE-Gate-0 阶段写：

- Introduction skeleton；
- Related work map；
- Problem formulation；
- Method outline；
- Experiment plan；
- Discussion blueprint；
- provisional abstract with “to be completed after AE-Gate-1” 标签。

不允许在 AE-Gate-0 阶段写：

- 完整投稿 manuscript；
- 带最终结果语气的 abstract；
- cover letter；
- final conclusions。

### Phase 5：Citation Compliance + Abstract Agent

Citation 适配：

- AE01-AE15 的 DOI 和题名以 `applied_energy_literature_screen.md` 为准；
- 引文功能必须和段落功能对应，不能只堆 citation；
- 对 physics-informed 正反两侧都要引用，避免选择性引用；
- 对 Transformer 适用性要引用 AE09 这类“复杂模型不一定最好”的期刊内先例；
- 对 operational metrics 要引用 AE04/AE05/AE06/AE15。

Abstract 适配模板：

1. Energy-system motivation：VPP aggregate net-power forecasting supports dispatch/market/reserve decisions。
2. Structural problem：net power is a signed composition of heterogeneous components, so component-error coupling affects aggregate generalization。
3. Gap：fixed physics priors and shared encoders may not generalize under heterogeneous portfolio coupling。
4. Method：component-token separated inverted Transformer。
5. Evidence：multi-seed aggregate improvements and fixed-prior ablation。
6. Operational implication：after AE-Gate-1, add ramp/peak/deviation/reserve evidence；before gate, mark as pending。
7. Scoped conclusion：component separation is a robust inductive bias; fixed priors require data-adaptive validation。

Keywords 建议：

- virtual power plant；
- aggregate net-power forecasting；
- component-token separation；
- inverted Transformer；
- renewable energy forecasting；
- physics-informed learning；
- forecast generalization。

### Phase 6：Peer Reviewer Agent

原始 ARS 目标：模拟 peer review。

Applied Energy 适配：

Reviewer personas 应固定为：

| Reviewer | Focus |
|---|---|
| Handling editor | Applied Energy fit, desk-reject risk, energy-system contribution |
| Forecasting methodology reviewer | baseline strength, metrics, seeds, split, ablations |
| Power systems / VPP reviewer | operational relevance, dispatch/market/reserve/ramping connection |
| Physics-informed ML reviewer | C12 scope, fairness to physics-informed literature |
| Devil's advocate | single dataset, overclaiming, model-selection artifact risk |

Reviewer 不应只问“论文写得好不好”，而要强制检查：

- AE-Gate-1 是否满足；
- C12 是否被过度泛化；
- baseline 是否覆盖 Applied Energy 预期；
- operational metrics 是否足以支撑期刊 fit；
- limitation 是否诚实。

### Phase 7：Formatter Agent

暂不启动。等 AE-Gate-1 通过、完整稿完成后再做 Elsevier / Applied Energy 格式、reference style、cover letter。

## 目标论文标题候选

保守版：

> Component-token separation for robust virtual power plant aggregate net-power forecasting

Applied Energy 风格版：

> Robust aggregate net-power forecasting for virtual power plants via component-token separation under heterogeneous resource coupling

更有冲击力但需谨慎版：

> When fixed physics priors overfit: component-token separation for virtual power plant aggregate forecasting

当前推荐第二个。第三个可以作为 discussion angle，不建议一开始就作为标题，除非 AE-Gate-1/2 进一步支撑 fixed-prior overfitting 的外部有效性。

## ARS prompt templates

### Template 1：ars-plan

```text
ars-plan

Target journal: Applied Energy.
Task: Build a paper-writing plan, not a full manuscript.

Use these inputs:
- docs/analysis/applied_energy_ars_writing_adapter.md
- docs/analysis/applied_energy_journal_profile.md
- docs/analysis/applied_energy_paper_source_pack.md
- docs/analysis/applied_energy_pdf_extraction_matrix.md
- ara/logic/claims.md

Hard constraints:
- ARA is the fact layer; do not invent claims or numbers.
- Applied Energy AE01-AE15 are the journal-style/literature seed.
- Mainline: component-token separation improves VPP aggregate net-power forecasting generalization; fixed physics-prior additions can amplify validation-test overfitting under heterogeneous portfolio coupling.
- Do not claim physics-informed learning is generally ineffective.
- Because AE-Gate-1 is not yet complete, produce a writing plan, evidence gap list, and section-level blueprint only. Do not draft a full submission manuscript.

Output:
1. Paper Configuration Record.
2. Applied Energy-specific contribution statement.
3. Section-by-section plan.
4. Required evidence before full draft.
5. Risks and mitigation.
```

### Template 2：ars-outline

```text
ars-outline

Target journal: Applied Energy.
Mode: outline-only.

Use the Applied Energy ARS writing adapter and ARA source pack.
Generate a detailed outline for a research article on robust VPP aggregate net-power forecasting via component-token separation.

Requirements:
- Introduction must start from VPP operation, not Transformer.
- Related work must cover VPP uncertainty, aggregate net-load forecasting, Transformer forecasting, and physics-informed forecasting.
- Problem formulation must include net = Load - PV - Wind + Battery and signed error covariance.
- Experiments must include an AE-Gate-1 missing-evidence block for operational metrics and strong baselines.
- Discussion must include scoped interpretation of C12 and limitations.

Do not write full prose sections. Produce outline + claim-evidence map only.
```

### Template 3：ars-full

```text
ars-full

Target journal: Applied Energy.
Before drafting, verify AE-Gate-1.

AE-Gate-1 requires:
- operational metrics for A1, c23, and core baselines;
- strong baselines including persistence/naive, DLinear or NLinear, PatchTST or TFT, GRU/LSTM;
- mean ± std reporting;
- single-dataset defense.

If AE-Gate-1 is not satisfied, stop and produce only:
1. missing evidence table;
2. provisional manuscript skeleton;
3. exact experiments needed before full drafting.

If AE-Gate-1 is satisfied, draft a full Applied Energy research article using only facts from ARA and the source pack. Scope all claims conservatively.
```

### Template 4：ars-abstract

```text
ars-abstract

Target journal: Applied Energy.
Mode: abstract-only.

Write three abstract variants:
1. conservative version;
2. operation-focused version;
3. fixed-prior caution version.

Use this mainline:
Component-token separation improves VPP aggregate net-power forecasting generalization, while fixed physics-prior additions can amplify validation-test overfitting under heterogeneous portfolio coupling.

Constraints:
- Do not claim operational cost/reserve/market gains unless AE-Gate-1 metrics are available.
- Do not claim physics-informed learning is generally harmful.
- Mention multi-seed aggregate improvement and fixed-prior ablation only with ARA-supported numbers.
- Add a note after each abstract identifying which sentences are provisional if AE-Gate-1 is still missing.
```

## C12 的安全写作规则

可写：

- Fixed physics priors are not automatically beneficial in heterogeneous VPP aggregate forecasting.
- Stronger validation performance can hide poorer test generalization when priors encode portfolio-specific coupling.
- Component-token separation provided a more robust inductive bias than hand-specified physics tokens or graph biases in this setting.
- The finding complements physics-informed forecasting literature by identifying a boundary condition for fixed priors.

不要写：

- Physics-informed learning is ineffective.
- Physics priors hurt forecasting in general.
- Pure data-driven models are always better.
- The proposed model proves that physics should be removed from VPP forecasting.

原因：

- AE12/AE13 是 Applied Energy 内部 physics-informed / physics-constrained forecasting 的成功先例；
- 我们的证据只支持当前 VPP aggregate setting、当前 architecture family、当前 split family 下的 fixed-prior overfitting；
- 更强的广义结论需要额外数据集、portfolio adaptation 或外部验证。

## Plan A 试错结论

本轮 Scheme A 的正确结论是：

1. **Plan A 可以成立**：不需要先做复杂的 ARA-ARS formal bridge；用 `journal_profile + source_pack + writing_adapter` 就能让 ARS 开始按目标期刊写作。
2. **上一步 reviewer trial 不是终点**：它只暴露了痛点，真正产物应该是这个 writing adapter。
3. **当前最适合的 ARS 输出是 outline / argument blueprint**：不是完整稿。
4. **Applied Energy 最大痛点不是“会不会写”**：而是 AE-Gate-1 的 operational metrics 和 strong baseline closure。
5. **等 AE-Gate-1 补齐后，ARS 才能进入 full manuscript drafting**。

## 下一步建议

直接进入 AE-Gate-1 补实验包：

```text
Compare A1, c23, persistence, DLinear/NLinear, PatchTST/TFT, GRU/LSTM on:
- MAE / MSE / RMSE
- ramp-event MAE
- peak/valley MAE
- deviation penalty proxy
- reserve requirement proxy

Report mean ± std over seeds.
```

如果暂时不补实验，也可以先用本 adapter 让 ARS 生成：

- Applied Energy detailed outline；
- Introduction draft skeleton；
- claim-evidence map；
- experiment section TODO scaffold；
- abstract variants marked as provisional。
