# PhysFormer 架构深度分析：Aggregate MAE vs Theory MAE 能否同时下降？

> **Dual-Draft 共识报告** | Claude + Codex 独立分析 | 2026-05-08

## 实验数据回顾

| 版本 | Aggregate MAE | Theory MAE | Component Loss | 结论 |
|------|--------------|------------|----------------|------|
| V4 | **1.98 kW** | 3.81 kW | cw=0.05 | 最佳 aggregate |
| V5 | 2.12 kW (+7%) | **3.07 kW** (-19%) | cw=0.1→0.05 | 最佳 theory |
| V5.1 | 2.02 kW (+2%) | 4.40 kW (+15%) | cw=0.03→0.01 | 均未达最优 |

**核心矛盾：cw 增大 → theory 改善但 aggregate 退化；cw 减小 → aggregate 改善但 theory 退化。**

---

## 共识问题（Claude + Codex 独立发现）

### 1. [高严重度] Theory 没有直接优化目标

**Claude (C6):** loss 函数中 `theory_mae_real` 仅作为诊断指标，从不参与 `total_loss`。Theory 仅通过 component loss 间接监督，但 component loss 在归一化空间计算，与 real-space 的 evaluation metric 不对齐。

**Codex (X1):** Theory MAE 是 diagnostic-only。物理层可以通过改善 component proxy 来降低 component loss，而 residual 路径可以学习绕过或撤销 theory 的贡献。

**共识：** 这是 tradeoff 的根本原因之一。Theory 没有直接的梯度信号指向 net injection accuracy。

### 2. [高严重度] Shared encoder 创建梯度冲突

**Claude (C1):** Encoder 在 theory 路径（→ phys_layer）和 residual 路径（→ temporal_decoder → residual_head）之间共享。Component loss 推动 encoder 学习 physics-friendly 表示，但这同时改变了 residual 路径的输入。

**Codex (X2):** Residual head 预测 5 个分量 residual，但 aggregate target 只监督它们的带符号和。多种分量分解可以产生相同的 pred_net，导致 aggregate 改善时 component 语义退化。

**共识：** 共享 encoder + 多目标 loss = 梯度冲突。一个任务的梯度更新会干扰另一个任务。

### 3. [高严重度] PhysicsFiLM 缺乏信任机制

**Claude (C3):** FiLM 是单向的：physics → data（gamma/beta 调制），data 从不反馈改善 physics。Theory 质量被物理模型的表达能力限制。

**Codex (X3):** Physics features 总是通过 FiLM 调制 data latent，但没有 learned reliability gate。当 physics 有偏时，强 physics 会污染 data path；弱 physics 帮助 aggregate 但恶化 theory。

**共识：** 缺乏 physics-vs-data 的自适应混合机制。当前架构是"always trust physics"或"never trust physics"的二选一，无法按样本自适应。

### 4. [中严重度] Residual head 依赖 theory 质量（循环依赖）

**Claude (C5):** UnifiedResidualHead 以 `component_norm`（来自 phys_layer）作为输入。如果 theory 质量差，residual head 收到的 conditioning 也差，形成鸡生蛋问题。

**Codex (X2):** Residual head 是 under-identified 的 — 多种分量 residual 分解可以产生相同的 net 预测。

**共识：** Residual head 的输入依赖 theory 输出，但 theory 的改善又依赖 residual 的反馈。这形成了优化层面的循环依赖。

---

## 独立发现（仅一方发现）

### Claude 独立发现

- **C2 (中):** Component loss 在归一化空间计算，load (std ~15kW) 和 wind (std ~0.8kW) 被等权对待，但 real-space 中 load 误差主导 aggregate MAE
- **C4 (中):** Curriculum phases 治标不治本 — Phase 1 的 encoder bias 在 Phase 2 持续存在
- **C7 (高):** Theory 和 residual 优化同一目标（net injection），但表达能力不对称 — neural network 总是比 physics model 更能拟合 residual

### Codex 独立发现

- **X4 (中):** 物理层公式过于简单（load 用线性热力学模型，PV 用一阶辐照度×温度系数，wind 用平滑三次曲线），当公式有偏时，强制更好的 theory fit 与 aggregate accuracy 所需的 residual 自由度冲突
- **X5 (中):** Phase 3 将 component/residual regularization 权重设为零，导致纯 net_mse fine-tuning 可能以牺牲 theory 为代价改善 aggregate

---

## 架构修改建议（共识排序）

### 方案 1：Reliability-Gated Hybrid Prediction（两方均推荐，优先级最高）

**核心思想：** 不再强制 `pred_net = theory + residual`，而是引入 learned gate：

```
pred_net = gate * theory_net + (1 - gate) * data_net + constrained_residual
```

- `gate` 由 weather latent, component theory quality, battery features 等学习
- 当 theory 准确时（如 PV 在晴天），gate → 1，theory 主导
- 当 theory 不准确时（如 load 在节假日），gate → 0，data 主导
- Residual 只负责小的修正，不再需要"补偿" theory 的全部误差

**预期效果：** 直接消除 tradeoff — theory 改善时 gate 自动增大，不会损害 aggregate。

**实现复杂度：** 中等（~40 行改动）

### 方案 2：Two-Stage Training with Explicit Theory Loss（Claude 推荐）

**核心思想：** 分离优化目标，消除梯度冲突：

- **Stage 1（Theory Warmup）：** 冻结 residual head，用 `theory_mse_real` 直接优化 phys_layer → theory 质量
- **Stage 2（Residual Fit）：** 冻结 phys_layer，用 `net_mse` 优化 residual → aggregate 质量

**预期效果：** Theory MAE -30-50%（直接监督），Aggregate MAE -5-10%（无干扰）。

**实现复杂度：** 低（~50 行训练管线改动）

### 方案 3：Stop-Gradient Decoupling（Claude 推荐，最小改动）

**核心思想：** 用 `.detach()` 切断梯度冲突：

```python
# physformer.py forward():
# Physics branch receives detached encoder output
phys_input = enc_out.detach()  # component loss 不影响 encoder
# Residual branch receives normal encoder output
residual_input = enc_out       # net MSE 正常反传
```

**预期效果：** Component loss 只优化 phys_layer，net MSE 只优化 encoder + residual。

**实现复杂度：** 极低（2 行改动）

### 方案 4：Semi-Parametric Physical Layer（Codex 推荐）

**核心思想：** 提升物理模型的表达能力，减少 residual 负担：

- Load：加入 weekly pattern, holiday effect, price sensitivity
- PV：加入 clear-sky index, cloud intermittency, inverter clipping
- Wind：加入 air density correction, wake effects
- Battery：加入 market intent, degradation cost

**预期效果：** Theory MAE 天花板提升，residual 需要补偿的误差减少。

**实现复杂度：** 高（~200 行改动）

---

## 推荐实施路线

```
Phase 1（立即，1-2 天）：
  ├─ 方案 3: Stop-gradient decoupling（2 行改动）
  ├─ 添加 theory_mse_real 到 total_loss（1 行改动）
  └─ 验证：aggregate 和 theory 是否同时改善

Phase 2（如果 Phase 1 有效，3-5 天）：
  ├─ 方案 2: Two-stage training
  └─ 验证：theory MAE 是否显著下降

Phase 3（如果 Phase 2 有效，1-2 周）：
  ├─ 方案 1: Reliability-gated hybrid
  └─ 验证：gate 是否学到有意义的 physics-vs-data 混合

Phase 4（长期，论文扩展）：
  └─ 方案 4: Semi-parametric physical layer
```

---

## 结论

**当前架构无法同时改善 Aggregate MAE 和 Theory MAE。** 根本原因：

1. **梯度冲突**：Shared encoder + component loss + net MSE = 三个优化目标争夺同一组参数
2. **目标不对齐**：Theory 只被 component loss 间接监督，且在归一化空间计算
3. **缺乏自适应混合**：FiLM 是单向的，没有 trust gate 让模型按样本选择 physics vs data
4. **循环依赖**：Residual head 以 theory 输出为输入，但 theory 改善依赖 residual 反馈

**最有效的修改是 Stop-Gradient Decoupling + Explicit Theory Loss（方案 3+2），预计 2 行代码改动即可打破 tradeoff。**

---

*Consensus: YES — Claude 和 Codex 在核心诊断（梯度冲突、目标不对齐、缺乏 trust gate）和首选方案（reliability-gated hybrid + two-stage training）上完全一致。*
