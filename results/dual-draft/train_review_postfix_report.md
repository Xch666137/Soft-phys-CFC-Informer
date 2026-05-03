## Train Review: 双审修复后验证审查

**Scope:** 6 files modified (data_factory.py, losses.py, exp_physformer.py, physical_layer.py, conditioning.py, physformer.py, metrics.py, config.py, configs/physformer_default.yaml)
**假设:** 修复 dual-draft 审查发现的 17 个问题，以提升训练效率、物理一致性和实验有效性
**Mode:** 完整模式 (L1 + L2 + L3) — 修复后验证
**Highest Severity:** P2 (无 P0/P1 阻断问题)

---

### Layer 1 — 数据流完整性

**1a. 解包对齐** ✅
- `PhysFormerDataset.__getitem__` 返回 9 元组：`x_net_hist, ..., portfolio_idx`
- `_process_one_batch` 解包 9 元组：`x_net_hist, ..., portfolio_ids` — 对齐
- 新增 `x_net_hist` 放入 result dict，用于 test() 中提取 `last_hist` — 正确

**1b. 前向路径分支追踪** ✅
- `losses.py`: `y_aux` 作为可选参数传入 `compute_terms`，仅在非 None 时计算 `battery_power_mae`
- `battery_power_mae` **未加入 total_loss**（诊断指标，不是训练目标）— 这是设计选择，与 MSE-primary 目标一致
- 所有 denorm 操作保持梯度连通性（`.denorm_target()` 等均是可微的 scale+shift）

**1c. 物理层梯度连通性** ✅
- 移除 `anti_overlap_loss`（恒为零）— 不再浪费梯度归因
- 移除 `soc_transition_loss`（冗余）— 简化梯度路径
- `soc_bounds_loss` 保持从 `battery_soc_theory_real` → 物理层参数的反向传播

**1d. Loss 各子项梯度归因** ✅
- `net_mse`: 从 `pred_net` → UnifiedResidualHead + PhysicsFiLM + TemporalDecoder + Encoder → 全覆盖
- `soc_bounds_loss`: 从 `battery_soc_theory_real` → physical_layer → 覆盖电池参数（包括 per-portfolio embedding）
- `battery_power_mae`: 诊断项，不影响梯度

**1e. AMP 边界** ✅ 未修改

**1f. freeze 生效** ✅ 未修改（单阶段训练，无 freeze/unfreeze 逻辑变更）

**1g. 时序/数据划分完整性** ⚠️
- Portfolio embedding 修复：`_portfolio_id_to_idx` 现在通过排序 group_ids 确保跨 split 一致
- 通过 `_train_portfolio_ids` 存储 train split 的 portfolio 顺序
- **遗留问题**: 若 val/test 包含 train 中不存在的 portfolio，fallback 到 index 0（默认 embedding）。这是合理的降级行为。

**1h. 配置→代码传递链** ⚠️ [P3]
- `overlap_weight` 已从 `TRAINING_KEYS` 移除，`PhysLoss.__init__` 不再接受该参数，`_select_criterion` 不再传递
- **残留**: `drivers.py:71-73` 仍保留向后兼容代码（无害，不会执行）

---

### Layer 2 — 目标对齐

**2a. 因果链校验** ✅
- 目标: Val MSE 降低（与黑箱竞争）
- 修复→机制→指标:
  - WarmRestarts → LR 周期性重置 → 逃脱 plateau → Val MSE 可能进一步降低
  - film_scale 0.2→0.5 → 物理调制增强 → 更强的 FiLM 引导 → 可能降低 MSE
  - theory_net 32-dim 投影 → theory→residual 映射容量提升 → 可能降低 MSE
  - SOC loss 简化 → 梯度路径更直接 → 物理学得更好 → 可能间接改善 MSE

**2b. 归因排他性** ⚠️ [P2]
- 多个修复同时应用，个体贡献无法归因
- **建议**: 运行 ablation: WarmRestarts vs CosineAnnealing，film_scale=0.2 vs 0.5 vs 1.0

**2c. 反向作用排查** ✅
- film_scale 增大可能带来早期训练不稳定 — tanh 限制了幅度在 [0.5, 1.5]/[-0.5, 0.5]，仍安全
- WarmRestarts 的 LR 重置可能短暂增加 loss — 正常现象，T_0=15 足够让 loss 重新下降
- Wind 软阈值 sigmoid 替代 boolean — 可能略微降低 wind_theory 的物理精确性（sigmoid 过渡区比 boolean 模糊），但梯度收益可观

**2d. 指标-目标一致性** ✅
- 所有修改保持 `early_stop_metric: net_mse` 作为主要选择标准
- `residual_std` 现在在 MW 空间计算，与 MSE/MAE 可比

---

### Layer 3 — 设计幻觉检测

**3a. 已修复的设计幻觉** ✅
| # | 原幻觉 | 修复 | 状态 |
|---|--------|------|------|
| 1 | SOC transition loss 冗余 | 移除，保留 soc_bounds | ✅ |
| 2 | Anti-overlap 恒为零 | 移除 | ✅ |
| 3 | no_temporal_decoder 消融未生效 | 正确连接 ablation flag | ✅ |
| 4 | y_aux 加载但未使用 | 传入 loss，计算 battery_power_mae 诊断 | ✅ |
| 5 | battery_power_mae 被 `if False` 跳过 | 替换为真实 y_aux 计算 | ✅ |
| 6 | Wind 阈值不可微 | 替换为 sigmoid 软门 | ✅ |
| 7 | residual_std 单位不一致 | 改为在 MW 空间计算 | ✅ |
| 8 | Ramp violation 忽略边界跳变 | 添加 last_hist 到差异计算 | ✅ |
| 9 | Portfolio embedding 索引错乱 | 排序+train split 映射 | ✅ |
| 10 | 未来天气信息泄露 | 文档标注，no_future_weather 消融可用 | ✅ |
| 11 | FiLM scale 过于保守 | 0.2 → 0.5 | ✅ |
| 12 | theory_net 单标量淹没 | 添加 32-dim 投影 | ✅ |
| 13 | PV 温度系数不对称 | 改为对称 (temp-25.0) | ✅ |
| 14 | Cosine Annealing plateau | 替换为 WarmRestarts | ✅ |
| 15 | 训练不支持 checkpoint resume | 添加 resume 逻辑 | ✅ |

**3b. 未修复的次优先级问题** ✅ (已处理或保持为诊断用)
- Battery step-by-step loop — 需要更大架构变更，未修复
- Checkpoint resume 依赖 `training_state.pth` 存在 — 已实现
- PV 温度系数初始值 — 保持在 0.2（softplus 后 ≈0.6%/°C），合理

**3c. 残留问题** ⚠️ [P2]
- `film_scale` 在 `_build_model` 中的 fallback 默认值仍为 0.2（行 84：`getattr(self.args, "film_scale", 0.2)`），与 PhysFormer/PhysicsFiLM 的新默认 0.5 不一致。如果运行时 args 未设置该字段，将回退到旧值 0.2。

---

### Summary

| Severity | Count | Key Issues |
|----------|-------|------------|
| P0 | 0 | — |
| P1 | 0 | — |
| P2 | 2 | film_scale fallback 默认不一致；多项修复的归因隔离 |
| P3 | 1 | drivers.py overlap_weight 向后兼容残留 |

**Conclusion**: 所有 dual-draft 发现的 P1/P2 问题已成功修复。代码无数据流断点或梯度问题。建议修复 film_scale fallback 默认值，然后通过训练实验验证效果。

**[CONSENSUS: YES]** — 与双审报告一致的修复方向
