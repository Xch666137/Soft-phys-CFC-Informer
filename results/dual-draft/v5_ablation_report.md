# Peer Review Consensus Report

- File: `results/dual-draft/v5_ablation_analysis.md`
- Reviewers: **claude** vs **codex**
- Matching rule: loose (v2: category equivalents + range overlap), tolerance=2

## Summary
| Bucket | Count |
|---|---:|
| Consensus (both flagged) | 2 |
| Only claude | 7 |
| Only codex | 12 |
| **Total unique** | **21** |

`claude` reported 9 raw issues; `codex` reported 14 raw issues.

## 1. Consensus Issues (2)
_Both reviewers independently flagged the same location + (compatible) category._

### L27 — `training_dynamics` (severity: high/high)
- **claude**: V5.4 最佳 epoch=9 而 V5.5 最佳 epoch=33 的差异未被解释。R8 组在 epoch 8 触发 cosine 重启（与 Phase 1→2 切换同步），这可能导致 Val MSE 的 '假性改善'——重启时 LR 回升暂时跳出局部极小
  - Fix: `分析 restart_t0=8 的 cosine 重启是否与 Phase 1→2 切换产生了人为的 Val MSE 低谷。如果确认，则需要排除重启后前 2 个 epoch 的 Val 值作为 '虚假最优'`
- **codex**: “全部 epoch ~39 early stopping”与最佳 epoch 和 patience=20 不自洽。
  - Fix: `逐实验报告 actual stop epoch、best epoch、patience counter、是否达到 max epochs，并核对日志。`

### L60 — `physics_interpretation` (severity: low/medium)
- **claude**: Residual mean 的正负号提供了重要信息：V5.4 的 residual mean = +0.00234 (系统高估 net)，V5.5a = -0.00144 (系统低估 net)。V5.5b 的 |residual mean| = 0.00052 最小——接近零偏置
  - Fix: `同时关注 residual mean（偏置）和 std（噪声）。V5.5b 的残差分布 (mean=-0.0005, std=0.0026) 最接近零均值高斯——这是物理 + 数据驱动残差设计的理想行为`
- **codex**: 组件 MAE 未按组件尺度归一化，也未检查物理约束一致性。
  - Fix: `补充 per-unit/相对误差、分位数、误差分布、功率平衡残差、SOC 边界、充放电互斥和 ramp 事件分析。`

## 2. Only from claude (7)

- **缺少 Phase 3 数据** `[medium/missing_analysis]` 所有实验在 epoch 39 early stopped，但 phase_2_epochs=50，意味着没有进入 Phase 3（纯 net MSE）。Phase 3 是论文设计的最终评估阶段，丢失了关键的纯 net fine-tune 数据
  - Original: `phase_1_epochs=8, phase_2_epochs=50, patience=20`
  - Fix: `建议增加 patience 到 30 或降低 early_stop_start_epoch，确保至少进入 Phase 3 前几个 epoch。或直接设置 phase_2_epochs=30，让模型有 20 个 epoch 在 Phase 3 训练`
  - Why: 如果 Phase 3 从未执行，则消融只覆盖了成分监督阶段，论文的核心声明（'Phase 3 net-only fine-tune is sufficient'）未在消融中验证

- **Ramp Violation 差异** `[medium/overfitting]` V5.5 的 Ramp Violation (0.0060%) 是 V5.5b (0.0025%) 的 2.4 倍。R20+B5 组合不仅泛化最差，ramp 违规也最严重——两个'safety'机制叠加反而产生了不安全的预测
  - Original: `V5.5: Ramp Violation = 0.0060%, V5.5b: 0.0025%`
  - Fix: `强调 R20+B5 组合的消极交互：更长的重启 + 更强的电池监督 → 模型在 Phase 2 可能过度信任物理层 → 残差分支补偿不足 → 预测不平滑。建议查看 V5.5 的 pred.npy 时序是否存在高频振荡`
  - Why: 如果 '安全机制组合导致不安全输出' 成立，这对论文的 claim（physics guidance improves stability）构成潜在反例

- **Train/Test 背离** `[high/overfitting]` Train/Test 背离的证据强度不足：仅单次 seed=2024 运行，无法排除随机性。V5.5a 和 V5.5b 的 MAE 差距仅 0.00002 MW (1%)，在单次运行下可能不显著
  - Original: `V5.5a 训练冠军但测试 2/7 指标最优；V5.5b 训练末位但测试 6/7 指标最优`
  - Fix: `明确标注此为单 seed 结果，建议用 3+ seed 重复确认，或在结论中使用 'preliminary' 限定词。同时计算 V5.5a vs V5.5b 各指标的实际差距百分比`
  - Why: 单次运行的 train/test 背离可能是随机波动而非系统性过拟合。需要多次运行才能做出可靠判断

- **V5.4 梯度死寂** `[high/training_dynamics]` V5.4 的 GradNorm ~1e-9 且 GradCos 恒定为 0，这是梯度消失 (vanishing gradients) 而非 '收敛'。对比 V5.5a（同样 R8，但 B5）GradNorm 正常为 ~1e-3，说明 B1 组合在 R8 下确实训练失败
  - Original: `V5.4 的 GradCos 恒定 0.000，GradNorm ~1e-9 — 梯度完全停滞`
  - Fix: `将 '梯度停滞' 升级为 '梯度消失/训练失败'。这比单纯的过拟合更严重——V5.4 的物理层可能完全没有学到有效表征。解释为何 B5 能阻止梯度消失（更强的 battery 监督信号为物理层提供了持续的梯度源）`
  - Why: 区分 '梯度消失' 和 '收敛' 对理解 R8 的失败机制至关重要。前者是训练缺陷，后者是过拟合

- **组件 MAE 解读** `[medium/physics_interpretation]` V5.4 的 Load MAE (0.0148) 和 Battery MAE (0.0213) 比 V5.5a 高 7-13 倍。这不仅是泛化问题——说明 B1+R8 组合下物理层的 load 和 battery 分支根本没有正确学习
  - Original: `V5.4 在 load 和 battery 组件上的误差显著高于其他三组 (7-13×)`
  - Fix: `添加物理诊断：V5.4 的 load 和 battery 分支是否输出恒为零或接近常量？检查 component_theory_real 的输出分布。如果是，说明梯度消失导致物理层退化为 trival predictor`
  - Why: 组件级诊断可以区分 '泛化差' 和 '根本没学到'——两者对应不同的解决方案

- **Train/Test 背离的因果链** `[high/overfitting]` 分析归因于 'R8 频繁重启导致过拟合' 但缺少直接证据。R20 的隐式正则化假说可以通过以下方式验证：(1) 比较 R8 vs R20 组的 weight norm、(2) 比较 train loss 和 val loss 的 gap 随 epoch 的变化
  - Original: `R8 频繁重启可能导致过拟合——训练集上的优势未能泛化`
  - Fix: `分析 train.log 中每 epoch 的 Train Loss vs Val Loss gap。如果 V5.5a 的 gap 持续扩大而 V5.5b 保持稳定，则为过拟合提供了直接证据。否则应重新考虑 '泛化差异' 的归因`
  - Why: 没有 train/val gap 的数据，'过拟合' 只是推测。Train Loss 数据在 train.log 中完全可以提取

- **推荐配置的二选一困境** `[high/recommendation]` 文档将推荐配置设为二选一（训练最优 vs 测试最优），但可能存在第三个选项：restart_t0=8 + batt_w=5.0 的 V5.5a（训练最优）+ 降低 patience 以在 Phase 3 fine-tune = 取两者之长
  - Original: `推荐的 thesis 配置是 V5.5a（训练最优）还是 V5.5b（测试最优）？`
  - Fix: `提出第三种方案：V5.5a 的配置 + early stopping 在 Val MSE 最低点（epoch 12-15）而非等到 patience 耗尽。理由是 V5.5a 的最佳 Val MSE (0.3830) 远低于 V5.5b 的任何 epoch。在最佳 checkpoint 处停止可能同时获得训练时的物理对齐优势和泛化的最低误差`
  - Why: 当前的 early stopping 在 epoch 39 才触发，此时 V5.5a 已从 0.383 退化到 0.411。在最佳 checkpoint 处测试可能完全改变结论

## 3. Only from codex (12)

- **实验设计** `[high/methodology]` 2×2 消融只有端点配置，缺少重复种子、参数梯度和交互效应分析。
  - Original: `2×2 消融：`restart_t0` (8 vs 20) × `battery_component_weight` (1.0 vs 5.0)`
  - Fix: `补充多 seed、restart_t0 与 battery_component_weight 的中间值或响应面分析，并报告主效应与交互项。`
  - Why: 当前设计只能说明四个配置的单次观测结果，不能稳健归因到 R20、B5 或二者交互。

- **训练排名表** `[high/recommendation]` 训练排名混用了最佳 Val MSE 与最终 Val MSE。
  - Original: `| V5.5 (R20,B5) | 0.4258 | 33 | 0.4265 | 2nd |`
  - Fix: `明确排名依据。若按最佳 Val MSE，排序应为 V5.5a、V5.4、V5.5b、V5.5；若按最终 Val MSE，应重新命名列和结论。`
  - Why: 后续“V5.5b 训练末位”和推荐逻辑依赖该排名，当前表述会误导结论。

- **测试指标计数** `[high/recommendation]` 表格只有 6 个测试指标，但文本称 2/7 和 6/7。
  - Original: `V5.5a 训练冠军但测试 2/7 指标最优；V5.5b 训练末位但测试 6/7 指标最优。`
  - Fix: `补齐第 7 个指标或改为 V5.5a 1/6、V5.5b 5/6，并避免简单投票式推荐，需指定主指标。`
  - Why: 指标数量错误直接影响“测试最优”的证据强度。

- **统计显著性** `[high/statistical_validity]` 测试差异很小但未报告方差、置信区间或显著性检验。
  - Original: `| MAE (MW) | 0.00207 | 0.00226 | 0.00198 | 0.00200 | V5.5a |`
  - Fix: `对多 seed 或时间块 bootstrap 报告均值、标准差、95% CI、效应量和配对检验。`
  - Why: V5.5a 与 V5.5b 的 MAE 差异仅 0.00002 MW，可能完全落在随机波动内。

- **MSE 单位** `[low/methodology]` MSE 单位显示损坏，影响可复现性。
  - Original: `| MSE (×10??) | 7.999 | 8.955 | 7.628 | **7.281** | V5.5b |`
  - Fix: `修正为明确单位，例如 `MSE (×10^-6 MW^2)`，并确认与 RMSE 数值一致。`
  - Why: 功率系统论文中单位和量纲必须明确。

- **Train/Test 背离表述** `[medium/overfitting]` 文档实际比较的是验证集与测试集，不是训练集与测试集。
  - Original: `**Train/Test 背离**: V5.5a 训练冠军但测试 2/7 指标最优`
  - Fix: `改称 Val/Test 背离，另行报告 train loss、val loss、test loss 和泛化间隙。`
  - Why: 没有训练损失曲线时，不能把验证/测试排序差异解释为训练过拟合。

- **梯度动态表** `[high/training_dynamics]` GradNorm 接近零时 GradCos/GradAngle 可能退化，不能直接解读为 90° 或梯度停滞。
  - Original: `V5.4 的 GradCos 恒定 0.000，GradNorm ~1e-9 — 梯度完全停滞。`
  - Fix: `说明 cosine 的 epsilon 处理，报告原始梯度范数、loss 分项、LR、是否存在 detach/clip/AMP，并避免把退化 cosine 当作方向证据。`
  - Why: 当梯度范数接近零时，角度计算数值不可靠，可能是度量实现伪象。

- **梯度动态符号与统计** `[medium/training_dynamics]` 负号疑似损坏且只给极值范围，缺少时间窗口和统计摘要。
  - Original: `| V5.5a | ?0.985~+0.941 (震荡) | 20°~172° | ~1e-3 | ~1e-2 |`
  - Fix: `修复负号，报告 epoch/batch 级均值、中位数、IQR、曲线图和计算样本数。`
  - Why: 单纯用最大最小值描述“震荡”无法判断训练是否稳定或是否改善对齐。

- **过拟合因果解释** `[medium/overfitting]` 将 R8 频繁重启归因为过拟合证据不足。
  - Original: `R8 频繁重启可能导致**过拟合**——训练集上的优势未能泛化。`
  - Fix: `用 train/val 曲线、checkpoint 泛化曲线、多 seed 和不同 LR scheduler 对照验证；表述改为假设而非结论。`
  - Why: 频繁重启也可能产生正则化或优化扰动，当前数据不能支持因果判断。

- **R20 泛化优势结论** `[high/recommendation]` “R20 的泛化优势”忽略了 R20+B5 的测试结果变差，交互效应未被处理。
  - Original: `单独使用 restart_t0=20（V5.5b）虽然训练慢，但测试中全面碾压——更长的重启周期提供了隐式正则化。`
  - Fix: `改为“R20 在 B1 条件下表现最好”，并量化 R20×B5 交互；不要泛化为 R20 普遍优势。`
  - Why: V5.5(R20,B5) 多项测试指标劣于 R8/B5 和 R8/B1，说明 restart_t0 的效果依赖 battery weight。

- **遗漏分析** `[medium/missing_analysis]` 缺少场景分层和极端工况分析。
  - Original: `- 组件误差分布是否揭示了物理层的结构性缺陷？`
  - Fix: `按高负荷、高 PV、高风电、SOC 边界、快速爬坡、日夜/季节等场景分层报告误差。`
  - Why: 电力系统模型的工程价值取决于关键工况下的鲁棒性，聚合均值可能掩盖失效模式。

- **最终配置选择** `[high/recommendation]` 推荐配置没有决策准则，且存在用测试集选模型的风险。
  - Original: `推荐的 thesis 配置是 V5.5a（训练最优）还是 V5.5b（测试最优）？`
  - Fix: `先定义主目标，如 RMSE、ramp violation、物理一致性或电池组件精度；用验证集/交叉验证选择配置，最终测试集只做一次报告。`
  - Why: 论文结论应基于预先定义的指标层级，否则容易 test-set overfitting。
