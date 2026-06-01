# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Dual-Agent Cross-Verification (optional, L2/L3 tasks)

For decisions whose rework cost outweighs one Codex call, invoke the `dual-draft` skill so Codex independently drafts the same review, then aggregate into Consensus / Only-X buckets.

**Three-question trigger test (≥1 YES → L2; ≥2 YES + external stakes → L3):**
1. Would an error require edits across multiple files to fix?
2. Will this decision "freeze" after commit / submission?
3. Does it need cross-dimensional scrutiny (engineering + reviewer view, symbol self-consistency + cross-section integrity)?

**Invocation:**
```bash
bash ~/.Codex/skills/dual-draft/codex_review.sh <target> <codex.json>
python ~/.Codex/skills/dual-draft/aggregate.py <Codex.json> <codex.json> <report.md>
```

See `~/.Codex/skills/dual-draft/SKILL.md` for templates, fallback behaviour, and validation baselines.

**Do not overuse.** L0/L1 tasks execute directly — over-invocation violates "Simplicity First" and wastes Codex quota. Consensus rows are priority hints, not ground truth; Only-X rows are required reading (they're where each model's blind-spot coverage lives).

---

## 6. ARA Context Bootstrap

本项目维护了一个 [ARA (Agent-Native Research Artifact)](ara/) 知识包。**每个新 session 开始处理任务前，先读 ARA 快速加载上下文：**

```
1. ara/PAPER.md              → 研究概览 + 版本状态（200 tokens）
2. ara/trace/exploration_tree.yaml → 研究 DAG：已有的 dead_end、decision、当前状态
3. ara/logic/claims.md       → 当前声明的证伪状态（哪些已证实、哪些待实验）
4. ara/staging/observations.yaml → 待结晶的观察（若有）
```

**避免重复 dead_end 的检查流程：**
- 用户提出新改动 → 在 exploration_tree.yaml 中 grep 相关关键词的 dead_end 节点
- 若匹配到历史 dead_end → 引用具体节点 ID 和 failure_mode/lesson，告知用户
- 当前已知 dead_end：N06 (sigmoid gate)、N07 (component loss 过强)、N12 (Phase 3 无效)、N22 (V5.4 梯度消失 R8+B1)

**实验结束后：** 运行 `/research-manager` 将本回合的决策/死胡同/观察追加到 ara/。

## 7. Brainstorming Before Implementation

**L0/L1 直接执行。L2+ 改动必须先走 `/brainstorm` skill。**

分层标准：
- **L0**: typo、变量改名、单行修复、纯文档 → 直接执行
- **L1**: 单文件、方向明确、≤30s 回滚 → 一句话确认假设即可
- **L2**: 多文件、架构决策、实验设计、超参调优 → 完整 brainstorming 流程
- **L3**: L2 + 投稿/部署等外部 stakes → L2 流程 + dual-draft 交叉验证

**默认 L2，不确定时宁可多问一句。** Skill 接管后执行：ARA 上下文加载 → dead_end 检查 → 一问一答 → 2-3 方案 → 等用户选 → 成功标准确认（改动/验证/阈值）→ 代码。设计文档（≥300 行）保存到 `docs/plans/YYYY-MM-DD-<topic>.md`。

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.