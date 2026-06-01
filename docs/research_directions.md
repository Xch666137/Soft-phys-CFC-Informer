# PhysFormer 研究方向与实验路线图

> 最后更新：2026-05-06 | 当前阶段：V5 完成，调优中

## 已完成实验总结

| 版本 | 核心改动 | Test MAE | Theory MAE | Val MSE | 结论 |
|------|---------|----------|------------|---------|------|
| V3 | Baseline (CosineAnnealing) | 1.932 kW | 4.874 kW | 0.387 | — |
| V4 (Phase 1) | calendar + component loss + load proxy | 1.976 kW | 3.811 kW | 0.381 | **最佳综合**：Theory -21.8% while MAE only -2.3% |
| V4.1 | V4 + sigmoid gate + time_proj + 归一化 loss | 2.002 kW | 4.966 kW | 0.379 | **回归**：gate 削弱 residual → theory 退化 |
| V4.2 | V4 - gate - time_proj + 归一化 loss + schedule fix | 2.048 kW | 2.981 kW | 0.378 | **理论最优但最终退化**：component loss 过强 |
| V5 | Component-consistent residual + TemporalDecoder + Curriculum | 2.120 kW | 3.074 kW | 0.404 | **Component 最优但 aggregate 退化**：需调优 |

### V5 分量级指标（V5 独有）

| 分量 | V4 | V4.1 | V4.2 | V5 | 最优 |
|------|------|------|------|------|------|
| Load MAE | 14.707 kW | 2.044 kW | 2.093 kW | 2.069 kW | V4.1 |
| PV MAE | 3.998 kW | 3.795 kW | 2.449 kW | 1.892 kW | **V5** |
| Wind MAE | 0.825 kW | 0.459 kW | 0.355 kW | 0.313 kW | **V5** |
| Battery Power MAE | 21.345 kW | 1.532 kW | 1.686 kW | 1.340 kW | **V5** |
| Battery SOC MAE | 20.111 kW | 4.579 kW | 6.190 kW | 4.422 kW | V4.1 |

### 关键发现

1. **V4 的 MW 空间 component loss (weight=0.05) 是当前最优配置**——在物理质量和最终精度间取得最佳平衡
2. **Sigmoid gate 有害**——减少 residual 自由度迫使 theory_net 退化（共享编码器梯度耦合）
3. **Load 是瓶径**——theory MAE=14.7kW vs Wind=0.83kW (18x 差距)
4. **Component loss 存在 theory-vs-final tradeoff**——太强则 theory 改善但 final 退化
5. **V5 Component-consistent residual 有效**——PV/Wind/Battery 分量指标均达最优，但 aggregate 退化说明 component loss 权重过高
6. **Curriculum Phase 3 无效**——纯 net_mse fine-tuning 未带来验证集收益，模型在 Phase 2 已收敛

---

## V5 方向（已完成，调优中）

### A+C: Component-Consistent Residual + TemporalDecoder 时间条件化

将 residual 从 1 维 scalar 扩展为 5 维 per-component 修正：
- `pred_net = (load_theory + load_res) - (pv_theory + pv_res) - (wind_theory + wind_res) + (batt_theory + batt_res)`
- 每个分量独立修正，消除"Load 误差交叉污染 PV 预测"
- TemporalDecoder 添加 `time_proj(y_mark)` 时间条件化（从零训练）
- 改动范围：`conditioning.py` (~40 行), `physformer.py` (~15 行), `losses.py` (~30 行), `temporal_decoder.py` (~10 行)

### D: Curriculum Training（三阶段训练）

- Phase 1 (epoch 1-15): component_loss_weight=0.1, residual 受限 → theory_net 学习物理
- Phase 2 (epoch 16-40): component_loss_weight 线性衰减, residual 全自由度
- Phase 3 (epoch 41-70): 纯 net_mse fine-tune
- 改动范围：`losses.py` (~20 行), `exp_physformer.py` (~30 行), `config.py` (~10 行)

### V5 结果与问题

**成功点：**
- Component-consistent residual 有效：PV -22.7%, Wind -11.8%, Battery Power -12.5%
- Residual 统计量改善：mean 最接近零 (-0.000848 MW)
- SOC 约束完美满足

**问题：**
- Aggregate 退化：MAE +7.5%, MSE +11.9%（相对于 V4）
- Curriculum Phase 3 无效：纯 net_mse fine-tuning 未带来验证集收益
- Component loss 权重过高导致模型容量从 net accuracy 转移到 component consistency

### V5 调优方案

**优先级 1（立即执行）：**
1. 降低 Phase 1 component loss 权重：cw 从 0.1 降到 0.03
2. 缩短 Phase 1：从 15 epochs 减到 5-8 epochs
3. 移除 Phase 3：它没有带来收益
4. 增加 residual head 初始化：std 从 0.01 增加到 0.05

**优先级 2（中期实验）：**
1. Component loss 改为 MSE 或 Huber loss
2. 仅对 load 分量施加 component loss（因为 load 是瓶颈）
3. Composite metric 用于 early stopping
4. Component residual 门控机制

**预期效果：** 通过降低 component loss 权重和缩短 Phase 1，预期 aggregate MAE 可以回到 V4 水平（~0.00198），同时保留 component 改善（PV/Wind/Battery 仍然优于 V4）

---

## 后续方向（V5 完成后评估）

### 方向 B：Load 独立建模 / Dual-Stream（大改动，高收益）

**动机**：Load theory MAE=14.7kW 是 Wind 的 18 倍。Load 受人类行为驱动（日历、电价、生活习惯），不适用物理引导。

**方案**：
- Stream A (PV+Wind+Battery)：保留现有 FiLM + physics equations
- Stream B (Load)：iTransformer-style variable attention
  - Input：calendar embeddings + historical net injection + temperature
  - 每个变量作为 token，cross-attention 捕获变量级交互
- Fusion：lightweight gated merge

**预期收益**：Load error 可能下降 50%+
**风险**：架构复杂度显著增加，训练可能需要两阶段

**决策触发条件**：V5 的 component-consistent residual 是否显著降低了 Load 分量误差？

---

### 方向 E：改进 PV/Wind 物理模型（小改动，小收益）

**动机**：PV 物理过于简单（一阶辐照度×温度），Wind 物理是通用三次曲线

**方案**：
- PV：太阳高度角、日照时长 mask、clear-sky index、逆变器限幅
- Wind：轮毂高度修正、空气密度修正

**预期收益**：PV/Wind theory 在极端天气/季节场景改善
**风险**：需要额外气象数据，增益可能被 residual 自动补偿

**决策触发条件**：V5 的 per-component residual 显示 PV/Wind residual 仍然显著 > 0？

---

### 方向 F：Uncertainty Quantification（中等改动，新能力）

**动机**：VPP 调度需要预测区间做风险决策

**方案**：
- MC Dropout（改动最小）或 Gaussian NLL head
- 输出 `(mean, variance)` 而非 point forecast

**预期收益**：论文加 "Uncertainty-aware VPP forecasting" 二级贡献
**风险**：NLL 训练可能不稳定，需要新评估指标

**决策触发条件**：V5 完成后评估剩余实验时间和论文叙事是否需要此方向

---

### 远期方向（Phase 2 / Next Paper）

| 方向 | 描述 | 优先级 |
|------|------|--------|
| **Portfolio Generalization** | DeepSets/GNN over assets 替代 ID embedding table | 中 |
| **Three-Stage Training** | 实际实现 YAML 中声明的 curriculum 配置 | 中 |
| **RevIN Integration** | 添加自适应归一化处理非平稳性 | 低 |
| **Battery SOC Fix** | 先惩罚再 clamp，让模型学习可行电池调度 | 低 |

---

## 实验决策树

```
V5 (A+C+D) 完成
  │
  ├─ Aggregate 退化？
  │   └─ YES → 调优方案 V5.5：降低 component loss 权重 + 缩短 Phase 1
  │   └─ NO  → V5 作为 baseline，进入方向 E/F
  │
  ├─ V5.5 调优后 aggregate 回到 V4 水平？
  │   └─ YES → V5.5 作为最终版本，保留 component 改善
  │   └─ NO  → 考虑仅对 load 分量施加 component loss
  │
  ├─ Load residual 显著 > PV/Wind residual？
  │   └─ YES → 启动方向 B (Load Dual-Stream)
  │   └─ NO  → 当前架构已足够，进入方向 E/F
  │
  └─ Component loss 改为 MSE/Huber？
      └─ YES → 对大误差给予更强惩罚
      └─ NO  → 保持 MAE，关注 aggregate 指标
```
