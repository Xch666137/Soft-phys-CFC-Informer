# Peer Review Consensus Report

- File: `~/.claude/skills/train-review/SKILL.md`
- Reviewers: **claude** vs **codex**
- Matching rule: loose (v2: category equivalents + range overlap), tolerance=2

## Summary
| Bucket | Count |
|---|---:|
| Consensus (both flagged) | 4 |
| Only claude | 6 |
| Only codex | 10 |
| **Total unique** | **20** |

`claude` reported 10 raw issues; `codex` reported 14 raw issues.

## 1. Consensus Issues (4)
_Both reviewers independently flagged the same location + (compatible) category._

### L1 — `usability` (severity: medium/medium)
- **claude**: SKILL.md 约 200 行，加载时消耗 ~2500 tokens。对于 'review 一下这个 20 行的改动' 的场景，skill 本身比被审查的代码还长。没有提供 '快速模式' 或 '最小审查' 的入口。
  - Fix: `在文档开头增加模式选择：'## 快速审查（<50 行改动）→ 仅 Layer 1 关键项：1a, 1c, 1e'。完整审查保留当前三层流程。触发时 Claude 根据改动规模自动选择模式。`
- **codex**: Several checks require runtime gradient instrumentation but provide no executable probe, fallback, or threshold.
  - Fix: `Provide a minimal one-batch gradient audit recipe or bundled script that prints per-loss/per-module grad norms, trainable parameter lists, optimizer param-group membership, and missing/unexpected checkpoint keys.`

### L45 — `methodology` (severity: medium/high)
- **claude**: Layer 1（数据流完整性）和 Layer 3（设计幻觉）的边界模糊。'装饰性物理约束'（L3 幻觉类型）本质上是 gradient disconnect 的特例——这恰恰是 Layer 1 在检查的内容。类似地，Layer 1 的 '物理层梯度连通性'(1c) 和 Layer 3 的 '梯度寄生' 检测的是同一个东西。用户执行 review 时会在两个 layer 之间纠结归属，降低效率。
  - Fix: `明确 Layer 1 和 Layer 3 的判定标准：Layer 1 关注 '代码是否正确执行了计算'（syntactic correctness of data flow），Layer 3 关注 '计算是否实现了声称的语义'（semantic fidelity）。或者在 schema 中合并 data_flow_break 和 gradient_pathology 为单一类别。`
- **codex**: The three layers overlap heavily without a precedence rule, so the same defect can be classified differently by different reviewers.
  - Fix: `Add a classification rule: use Layer 1 for mechanical tensor/autograd breaks, Layer 2 for goal/metric/causal-direction mismatches after implementation exists, and Layer 3 for claim-vs-implementation gaps. Include examples of how to classify one issue that touches multiple layers.`

### L78 — `schema_design` (severity: medium/high)
- **claude**: 设计幻觉有 8 个子类型（装饰性物理约束、空转 ablation、假 curriculum 等），但 JSON schema 只有一个 design_hallucination category。这意味着：1) 聚合时无法区分 '发现装饰性物理约束' 和 '发现假 curriculum'——它们被当作同一类 issue；2) 长期统计无法追踪哪类幻觉最常见。
  - Fix: `在 JSON schema 中增加可选的 subcategory 字段：'subcategory: 'decorative_physics' | 'dead_ablation' | 'fake_curriculum' | 'metric_laundering' | 'normalization_confusion' | 'dimensional_illusion' | 'gradient_parasitism' | 'config_drift''。aggregate.py 在匹配时忽略 subcategory（仅用 category），但人工阅读时可利用 subcategory 快速定位问题类型。`
- **codex**: The skill assigns only three layer categories, while the schema/template allow eight overlapping categories; aggregate matching falls back to exact category equality.
  - Fix: `Choose one category system. Prefer either three primary layer categories plus a separate `subcategory` field, or add aggregate equivalence groups such as `data_flow_break` ↔ `gradient_pathology`, `design_hallucination` ↔ `metric_validity`/`training_dynamics` where appropriate.`

### L96 — `completeness` (severity: low/medium)
- **claude**: E 维度（配置一致性）仅覆盖了三阶段 config 之间的 struct 一致性，但缺少 config→代码的交叉验证。例如 YAML 中 `no_battery_branch: true` 被设置，但代码中 PhysFormer.__init__ 的默认值是 `False`，如果 YAML → argparse.Namespace → model init 的传递链有一处断裂，config 设置就静默失效。
  - Fix: `增加 E5 '配置→代码传递链验证'：对每个关键 ablation flag，追踪其在 YAML → config_to_args → finalize_args → model.__init__ → model.forward() 五步传递中的每一环，确认中间没有默认值覆盖。`
- **codex**: The schema/template allow `training_dynamics`, but the skill only adds visualization and metric optional sections, not optimizer/scheduler/early-stopping/checkpoint-state review.
  - Fix: `Add an optional T section for training dynamics: optimizer param groups, LR scheduler step timing, gradient clipping, AMP overflow handling, early stopping metric, checkpoint resume including optimizer/scaler/scheduler state, and eval/train mode transitions.`

## 2. Only from claude (6)

- **templates/train_review.txt: 文件路径硬编码** `[low/template_quality]` Codex 模板硬编码了 PhysFormer 项目的文件路径（physformer/models/physical_layer.py, physformer/utils/losses.py 等）。如果该 skill 被复用到其他项目或 PhysFormer 重构了目录结构，模板将指向不存在的文件。
  - Original: `Also read these supporting files to understand the full training pipeline:
- physformer/models/physformer.py
- physformer/models/physical_layer.py
...`
  - Fix: `将这些路径作为模板变量（{{SUPPORTING_FILES}}）而非硬编码。或者在 SKILL.md 中维护一个 '关键文件清单'，让 codex_review.sh 的调用者指定 --var SUPPORTING_FILES='...'。短期内可接受，因为这是项目专用 skill。`
  - Why: 模板硬编码降低了 skill 的可移植性。虽当前仅服务于 PhysFormer 项目，但值得在文档中注明此限制。

- **SKILL.md: Layer 2 因果链校验** `[high/completeness]` Layer 2 的因果链校验要求画出 '改动 → 机制 → 指标' 链条，但缺少一个关键检查：**多机制混杂**。在 PhysFormer 这种多 loss 联合训练的场景中，一个指标的变化可能来自多个机制的叠加效果。例如 'BVR 下降' 可能同时来自 ramp penalty（直接约束）和更好的 component 分解（间接约束）。如果只验证了 ramp penalty 的梯度路径，就声称 'BVR 下降归因于 ramp penalty'，这是归因幻觉。
  - Original: `2a. 因果链校验: 画出 '改动 → 机制 → 指标' 的因果链。链条上的每一环在代码中是否都存在？`
  - Fix: `在 2a 之后增加 2a-bis '归因排他性检查'：是否存在其他机制也能解释同一指标变化？如果存在，改动是否做了消融实验来隔离？至少要在 review 报告中标注 '可能存在混杂因素'。`
  - Why: 这是物理引导 ML 论文的核心方法论陷阱——声称 '物理约束有效' 但实际上是数据驱动的 encoder 在起作用。

- **SKILL.md: 与 dual-draft 集成** `[medium/integration]` Severity 值使用 P0/P1/P2/P3，与 dual-draft 标准 schema 的 high/medium/low 不一致。aggregate.py 不做 severity 验证所以不会崩溃，但同一仓库中两种 severity 体系并存会导致：1) 跨 review 类型对比困难（paper review 的 'high' 和 train review 的 'P1' 等价吗？）；2) 未来若 aggregate.py 增加 severity-based 分组，会出错。
  - Original: `severity: P0 (BLOCKING) | P1 (HIGH) | P2 (MEDIUM) | P3 (LOW)`
  - Fix: `两种方案：A) 统一为 high/medium/low 保持与 dual-draft 一致；B) 保留 P0-P3（因为训练代码需要更细粒度的分级），但在 schema 中明确映射关系：P0→high, P1→high, P2→medium, P3→low，并在 aggregate 阶段按此映射做匹配。推荐 B。`
  - Why: 两种 severity 体系并存会在 cross-review 场景中制造混乱，尤其是当同一 PR 同时包含 paper text 和 training code 的修改时。

- **SKILL.md: Layer 3 检测方法** `[medium/completeness]` Layer 3 的检查方法缺少一个关键步骤：**数值实验验证**。对于疑似装饰性物理约束的问题，仅仅 trace 代码路径是不够的——需要建议一个微型数值实验来确认。例如 '将 phys_layer 参数随机化 ±50%，看 loss 是否变化 >1%'。当前方法全是被动的代码阅读，无法区分 '正常但看起来可疑的代码' 和 '真正有问题的代码'。
  - Original: `1. 声明-代码对照：逐条列出 paper/methodology 中的声明...
2. ablations 反推法：每个 ablation flag 设为 True vs False...
3. checkpoint 加载审计...
4. 梯度流审计...`
  - Fix: `增加检测方法 5：'微型数值验证'。对每个疑似幻觉，设计一个 ≤10 行的验证脚本（如 register_hook 检查梯度、随机化参数观察 loss 变化），在 review 报告中建议用户执行。这不是让 reviewer 运行实验，而是给出可操作的验证步骤。`
  - Why: 代码静态分析有天花板。某些设计幻觉只有运行时才能确认。提供验证脚本让 review 从 '我发现了一个可疑点' 升级为 '我发现了可疑点，运行这个 3 行脚本即可确认'。

- **SKILL.md: 可视化代码审查 (V)** `[high/completeness]` 可视化审查 (V) 和指标审查 (M) 被降级为'额外审查维度'，用平铺 checklist 而非三层方法论审查。这意味着它们不受 Layer 2（目标对齐）和 Layer 3（设计幻觉）的覆盖。但实际上可视化代码中的设计幻觉非常常见——例如 '硬编码参数画出的物理曲线可能来自不同实验版本，曲线看起来合理但数值不对'。这恰恰是 Layer 3 的典型场景。
  - Original: `### V. 可视化代码审查

当改动涉及 visualization/ 时启用：
| # | 检查项 | 要点 |
|---|--------|------|
| V1 | 数据泄露 | ...`
  - Fix: `将 V 和 M 提升为 Layer 2/3 的子维度，增加对应的设计幻觉类型：'绘图幻觉'（可视化声称展示 X 实验但实际数据来自 Y checkpoint）、'指标漂移'（metrics.py 中的计算与论文中定义偏离）。在 schema 中增加 'visualization_issue' 和 'metric_drift' 两个 category 或至少一个 subcategory 字段。`
  - Why: 可视化是论文的最终呈现——可视化中的设计幻觉直接导致虚假的实验结论被发表。当前设计将其视为二等审查，是方法论的结构性盲区。

- **SKILL.md: 审查前的必要澄清问题** `[high/usability]` 5 个前置问题被埋在文档末尾，且没有强制执行机制。实际使用中 Claude 很可能跳过提问直接开始审查（尤其是用户说 'review 一下这个文件' 时），导致 Layer 2 和 Layer 3 的审查缺少关键上下文。
  - Original: `## 审查前的必要澄清问题

开始审查前，若用户未提供，必须向用户确认：
1. 这次改动的目标是什么？...`
  - Fix: `1) 将前置问题移到三层方法论之前，标注为 GATE（不回答则不进入审查）；2) 增加 '快速模式' 选项——若用户不提供假设，则仅执行 Layer 1 并明确标注 'Layer 2/3 因缺少目标假设而跳过'。`
  - Why: 方法论的价值取决于正确使用。如果最关键的前置步骤容易被跳过，Layer 2/3 的产出将是猜测而非分析。

## 3. Only from codex (10)

- **train_review.txt target-alignment prompt** `[high/template_quality]` The Codex template asks for Target Alignment but does not supply or request the stated goal/hypothesis that the SKILL.md requires.
  - Original: `For each change or mechanism in the code, ask: does this actually move toward the stated goal?`
  - Fix: `Add prompt variables for goal, mechanism, metric, stage, and ablation context. If absent, instruct Codex to separate inferred assumptions from confirmed facts and avoid target-misalignment issues that depend only on speculation.`
  - Why: Codex cannot reliably judge alignment to an unstated goal. This will cause noisy or hallucinated target-alignment findings.

- **train_review.txt hallucination list diverges from SKILL taxonomy** `[medium/template_quality]` The Codex template lists seven anti-patterns and omits two of the SKILL.md taxonomy items while adding a different one.
  - Original: `1. Decorative physics ... 7. Component consistency fiction`
  - Fix: `Align the template with the SKILL.md taxonomy or explicitly define template-only subtypes. Include `dimension illusion` and `configuration drift`, and decide whether `component consistency fiction` is a subtype of target alignment, physics fidelity, or design hallucination.`
  - Why: Claude and Codex are not being prompted to look for the same hallucination classes, reducing cross-verification quality.

- **Core review layers omit temporal/data-split checks** `[high/completeness]` The core data-flow layer traces tensors but does not require review of temporal alignment, horizon offsets, scaler fitting, or train/val/test leakage.
  - Original: `| **1a. 解包对齐** | 对比 `PhysFormerDataset.__getitem__` 的 return 顺序与 `_process_one_batch()` 的解包顺序。字段增减时是否同步？ |`
  - Fix: `Add checks for window/label horizon alignment, target leakage through future covariates, scaler/statistics fitted only on train split, chronological split integrity, and denorm consistency between training, validation, and test.`
  - Why: For forecasting pipelines, silent one-step horizon mistakes and leakage often produce believable but invalid metrics while all tensor paths and gradients still look correct.

- **Layer 1 / 1b forward path tracing** `[medium/methodology]` The autograd heuristic about tensor reassignment is technically misleading.
  - Original: `是否存在 `.detach()` 截断、tensor 重赋值（`=` 而非 `[:]`）断裂梯度？`
  - Fix: `Replace this with PyTorch-specific break patterns: `.detach()`, `.data`, `.item()`, `.numpy()`, rewrapping with `torch.tensor(existing_tensor)`, non-differentiable ops such as `argmax`, and unsafe in-place mutation of leaf tensors/views.`
  - Why: Plain assignment like `x = f(x)` does not break the computation graph; slice/in-place mutation is often riskier. The current wording can create false positives and teach reviewers the wrong failure mode.

- **Design hallucination taxonomy** `[medium/methodology]` The eight hallucination types are useful but not MECE, and the skill does not define priority among overlapping types.
  - Original: `装饰性物理约束 ... 梯度寄生 ... 指标换壳 ... 归一化黑洞 ... 配置漂移`
  - Fix: `Group the taxonomy by claim source: mechanism wiring, experiment protocol, metric definition, scale/unit semantics, configuration/runtime behavior. Add precedence examples for overlaps such as decorative physics vs gradient parasitism and metric laundering vs normalization black hole.`
  - Why: The taxonomy will be hard to apply consistently. Overlap weakens both human usability and dual-review consensus.

- **Design hallucination detection source of truth** `[medium/completeness]` The detection method assumes paper/methodology claims are available, but many code reviews only have a diff, PR description, comments, config names, or commit message.
  - Original: `逐条列出 paper/methodology 中的声明，在代码中找对应实现。无法对应的 = 幻觉。`
  - Fix: `Define a source hierarchy: user-stated goal, PR description, commit message, docs/paper, config/CLI names, comments/docstrings, test names. Require reviewers to quote the source claim before labeling a design hallucination.`
  - Why: Without an explicit claim source, reviewers may infer author intent and produce speculative hallucination findings.

- **Directory/current-branch review advertised but Codex driver is file-only** `[high/integration]` The skill advertises `<target_file_or_dir>` and `/train-review` for current-branch changes, but the actual `codex_review.sh` preflight requires `-f` and rejects directories.
  - Original: `bash ~/.claude/skills/dual-draft/codex_review.sh --template train_review <target_file_or_dir> ...
/train-review                    # review 当前分支所有改动`
  - Fix: `Either restrict the skill to single files or add a wrapper that expands branch diffs/directories into file-level Codex reviews and then aggregates or summarizes them.`
  - Why: The most valuable L2/L3 training reviews are often multi-file changes, but the documented invocation cannot run on that scope as written.

- **Dual-draft output format conflict** `[high/integration]` The skill says Claude should produce `claude.json` for aggregation, but the documented review output format is Markdown rather than the JSON schema required by `aggregate.py`.
  - Original: `# Step 2: Claude 独立 review（本 skill），产出 claude.json ...
## 审查输出格式
```markdown
## Train Review: <target>`
  - Fix: `Make JSON the required output for dual-draft mode, referencing `train_review_issues.md`. Keep the Markdown report only as a standalone human-facing format or as aggregate output.`
  - Why: If Claude follows the skill body literally, `aggregate.py` cannot consume the output. This breaks the advertised dual-draft workflow.

- **L3 Phase 2 critique integration** `[high/integration]` The skill requires Phase 2 critique for L3 tasks but does not define a train-review critique template or command.
  - Original: `**L3 任务**（重写 loss、架构变更、投稿前最终 check）→ 额外 Phase 2 critique。`
  - Fix: `Add `templates/critique_train_review.txt` and document the exact `codex_critique.sh --template critique_train_review ... --critique-by-codex ...` flow, or explicitly state that L3 currently falls back to human critique.`
  - Why: The existing dual-draft critique default is paper-section oriented. Without a training-specific critique prompt, L3 cross-verification is underspecified.

- **Prerequisite clarification gate** `[medium/usability]` The skill blocks Layer 2 and Layer 3 unless five questions are answered clearly, including ablation availability.
  - Original: `不回答清楚以上问题，审查无法进入 Layer 2 和 Layer 3。`
  - Fix: `Allow review to proceed with explicit assumptions when the user lacks answers. Require only goal and expected mechanism for Layer 2; allow Layer 3 to use PR text, comments, configs, docs, and code names as claim sources.`
  - Why: In real code review, ablations often do not exist yet. Blocking the review prevents the skill from catching exactly the design gaps it is meant to expose.
