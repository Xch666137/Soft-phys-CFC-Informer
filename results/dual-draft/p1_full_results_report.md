# Peer Review Consensus Report

- File: `results/dual-draft/p1_full_results.md`
- Reviewers: **claude** vs **codex**
- Matching rule: loose (v2: category equivalents + range overlap), tolerance=2

## Summary
| Bucket | Count |
|---|---:|
| Consensus (both flagged) | 0 |
| Only claude | 9 |
| Only codex | 17 |
| **Total unique** | **26** |

`claude` reported 9 raw issues; `codex` reported 17 raw issues.

## 1. Consensus Issues (0)
_Both reviewers independently flagged the same location + (compatible) category._

_(none)_

## 2. Only from claude (9)

- **L1** `[high/methodology]` 梯度缩放实验 (Batch 3) 缺少 α=0 对照组: Batch 3 使用 seed=2024 但未重跑 baseline (α=1) 和 detach (α=0)。α=0.3-0.5 的组件崩溃需与同 seed 下的 α=0 对比以确认部分梯度流是否确实比零梯度流更差。当前结论依赖跨 seed 对比，可能受 seed 效应混淆。
  - Fix: `在同 seed=2024 下补充 baseline 和 detach 对照组，或引用 V6.1 seed=2024 的完整测试指标 (不仅 Theory MAE) 做同 seed 对比。`
  - Why: α=0.3/0.5 组件崩溃与 no_temp_s2026 崩溃数值几乎一致 (Load ~0.014, Batt ~0.021)，暗示这可能是 seed=2024 下的系统性崩溃而非梯度缩放特有。同 seed 对照可区分两类原因。

- **L1** `[high/physics_interpretation]` 物理分支梯度死亡的统一根因未归纳: d512、α≤0.5、no_temp_s2026 三种触发条件均导致组件崩溃到同一数值 (Load ~0.014, Batt ~0.021)，表明存在一个共同的底层机制——物理-数据容量比失衡。当前分析分散在各批次结论中，缺少统一的机制性假说。
  - Fix: `提出「梯度竞争假说」: 当共享编码器中数据通路的梯度幅度远超物理通路时 (通过 d_model 增大、梯度缩放减弱、或输入维度减少)，Adam 的 adaptive LR 会系统性压制物理分支参数更新。统一解释所有梯度死亡实例。`
  - Why: 分散的现象描述不能升级为机制性理解。论文需要 causal mechanism 而非 correlation observation。

- **L1** `[high/statistical_validity]` Batch 3 和 Batch 4 均为单 seed (2024)，统计显著性未验证。α=0.5 的聚合优势 (MAE 0.001951 vs 0.001985) 差异仅 1.8%，在单次运行下可能在噪声范围内。e3 vs baseline 的 MAE 差异 (+2.3%) 同样未经多 seed 确认。
  - Fix: `对 α=0.5 和 e3 做额外 2 个 seed 的确认实验，计算 Cohen's d 或 bootstrap 置信区间。如果资源有限，优先验证 e3 (最可行的架构改进方向)。`
  - Why: Batch 1-2 多 seed 验证了 C07 的统计显著性，但 Batch 3-4 的新发现仍基于单 seed。不应重复 V5.5 消融的教训。

- **L1** `[medium/training_dynamics]` OneCycleLR 的 pct_start=0.3 与 Phase 1→2 转换 (epoch 8) 不同步。Phase 1 (epoch 1-8) 使用 component_loss，Phase 2 (epoch 9+) 切换为 detach+joint loss。但 OneCycleLR 在 epoch 15 (30%×50) 才达到 max_lr。LR 峰值与 loss 景观切换错配——Phase 1 在低 LR 区域，Phase 2 的初始 7 个 epoch 仍在爬升 LR。
  - Fix: `将 pct_start 对齐到 phase_1_epochs/train_epochs ≈ 0.16，使 LR 峰值与 Phase 2 开始同步。或改用 ConstantLR warmup + CosineAnnealing 保留原语义。`
  - Why: Phase 1→2 是优化景观的断崖 (loss 函数变化 + detach 模式切换 + optimizer state reset)，此时 LR 应已在峰值区而非仍在爬升。

- **L1** `[medium/overfitting]` d512 的 Val/Test 背离未区分过拟合 vs 数据泄露。Val MSE=0.378 (epoch 10, best) 但在 Test MAE 上退化 7.4%。这可能不是经典过拟合 (train loss 仍在下降)，而是验证集与测试集分布不一致 (验证集包含与训练集相同的 portfolio，测试集是 hold-out portfolio)。
  - Fix: `检查 train/val/test 的 portfolio 分布。如果 val 和 train 共享 portfolio，大模型可能学到 portfolio-specific 模式，需要通过 cross-portfolio validation 诊断。`
  - Why: VPP 数据的 portfolio 结构可能引入隐式的数据泄露。此问题影响所有使用 portfolio 分裂策略的实验。

- **L1** `[medium/recommendation]` 最终推荐逻辑链存在跳跃: 从「e3 是性价比最高的架构改进」直接跳到 e3 是推荐配置，未与 detach (α=0) 做端到端对比。当前 e3 (no detach) 和 detach (d256, α=0) 是独立实验，缺少 e3+detach 的联合配置。论文推荐的最优配置可能是 e3+detach 而非 e3 alone。
  - Fix: `增加 e3+detach 联合实验。如果资源有限，在当前数据基础上明确标注推荐的不确定性: 「e3 no-detach 和 d256 detach 是两个 Pareto-optimal 点，联合配置待验证」。`
  - Why: 两个已验证的 Pareto 改进 (detach 改善物理, e3 改善深度) 可能叠加。不探索联合配置可能导致论文推荐的最优配置落后于实际 Pareto 前沿。

- **L1** `[medium/missing_analysis]` 缺少 e3 vs detach 的 head-to-head 对比分析。两者都是可行的单维度改进 (深度 vs 梯度隔离)，但适用场景不同: detach 改善物理一致性 (Theory -22%), e3 改善建模深度 (组件+物理双改善但幅度较小)。论文没有讨论两者的 trade-off 维度差异和组合可行性。
  - Fix: `增加一张 Pareto 前沿图，横轴 Theory MAE，纵轴 Test MAE，标注 baseline、detach (s2025/s2026)、e3、d512_e3，可视化配置间的 trade-off 关系。`
  - Why: 多目标优化问题的结论不应是单一推荐，应是 Pareto 前沿 + 场景化建议 (如物理一致性优先选 detach, 均衡选 e3)。

- **L1** `[low/physics_interpretation]` 组件崩溃数值的一致性 (Load ~0.014, Batt ~0.021) 缺乏物理语义分析。这两个值可能对应物理分支的 'trivial predictor'——预测常数均值。需要确认崩溃后的组件预测是否为常数 (如 Load_pred = mean(Load_train))。
  - Fix: `提取崩溃实验 (α=0.5, d512, no_temp_s2026) 的 test 组件预测，检查是否为常数输出。如果是常数，记录其常数数值并与训练集均值对比。`
  - Why: 确认崩溃的物理语义有助于理解梯度死亡的临床表现，也增强论文的机制性解释。

- **L1** `[low/methodology]` Batch 1-2 使用不同种子 (2025/2026) 但 Batch 3-4 回到 seed=2024，seed 选择缺乏系统性和理由说明。seed=2024 可能恰好是一个'好'或'坏'的种子，影响 Batch 3-4 结论的泛化性。
  - Fix: `后续实验使用固定种子集合 (如 2024/2025/2026 三个) 而非单一种子。短期: 在论文中明确标注哪些实验使用哪个种子，并讨论 seed 效应的可能影响。`
  - Why: 增加实验报告的透明度，避免审稿人质疑种子选择偏差。

## 3. Only from codex (17)

- **L5** `[low/methodology]` 并行 3 实验/批次且 GPU 99% 限制，未说明并行运行是否影响确定性、吞吐、温度降频或 dataloader 随机性。
  - Fix: `记录每次运行的 wall time、GPU 利用率、显存、确定性设置和依赖版本；关键实验单独复跑确认。`
  - Why: 硬件拥塞通常不是主要因素，但在小效应量比较中可能影响复现性。

- **L7** `[medium/training_dynamics]` P0 引入 OneCycleLR、early_stop_start_epoch=5、patience=8，但分析未检查 LR schedule 与 phase 切换、detach/alpha 切换之间的交互。
  - Fix: `报告 LR 曲线、phase 切换 epoch、梯度范数与 loss 分项曲线，并尝试固定 LR 或延后/提前 phase 切换作为对照。`
  - Why: OneCycleLR 的高 LR 阶段可能触发物理分支饱和或残差头接管，尤其会影响 alpha 部分梯度流实验。

- **L10** `[high/methodology]` Batch 1-2 只覆盖 baseline/detach/no_temp 的两个种子，Batch 3 和 Batch 4 又切换到 seed=2024，导致配置差异与种子差异混杂。
  - Fix: `对核心配置 baseline、detach、no_temp、alpha=0/0.3/0.5/0.7、e3、d512、d512_e3 至少统一跑 3-5 个相同种子，并报告均值、标准差和置信区间。`
  - Why: 当前很多结论依赖跨 seed 或跨批次比较，但不同批次不是同一随机条件下的完整因子实验。

- **L18** `[medium/recommendation]` detach 被称为两个种子下 MAE 和 Theory MAE 最优，但 seed=2026 中 detach 的 MAE 0.002008 只比 baseline 0.002016 好 0.4%，效应量很小。
  - Fix: `把结论改为“初步稳定但效应小”，并用多种子均值差、标准差和显著性检验支持。`
  - Why: 统计波动可能超过 8e-6 的 MAE 差距，直接推荐 detach 有过度解释风险。

- **L18** `[low/missing_analysis]` 只报告平均 MAE/MSE，缺少峰值、低负荷、高 PV、高风、SOC 边界等物理关键工况下的分层误差。
  - Fix: `按工况分桶报告组件误差、聚合误差和残差偏置，并检查崩溃配置是否集中失败于特定工况。`
  - Why: 电力系统预测的物理一致性通常在边界工况和快速变化阶段更关键。

- **L18** `[low/missing_analysis]` 没有报告误差方向、校准和残差自相关，只给 ResMean 一个均值指标。
  - Fix: `补充 residual distribution、ACF、按 horizon 的误差、P90/P95 absolute error 和 bias。`
  - Why: 均值接近 0 不代表残差无结构，可能仍存在系统性时序偏差。

- **L26** `[medium/overfitting]` no_temp_s2026 出现组件崩溃但聚合 MSE 最优，提示验证/测试指标可能允许非物理解；未检查是否存在数据泄露或目标构造捷径。
  - Fix: `检查时间切分、归一化统计是否只用训练集、滑窗是否跨越 split、未来变量是否进入输入，并做按时间段/工况的分层测试。`
  - Why: 残差补偿型模型若在测试集上异常稳健，需排除泄露或分布过近导致的捷径。

- **L34** `[high/methodology]` 梯度缩放实验缺失 alpha=0 和 alpha=1 的同批次直接对照，且 alpha 网格过稀。
  - Fix: `在同一 seed 和同一训练设置下补齐 alpha=0、0.1、0.2、0.3、0.5、0.7、0.9、1.0，并重复多种子。`
  - Why: 声称临界阈值 alpha>=0.7 和非单调最优位于 alpha=0 或 alpha>=0.7，需要端点和更密集区间支撑。

- **L34** `[medium/training_dynamics]` GradNorm 只给出近似数量级，未说明统计位置、层级、时间点或是否为全局范数。
  - Fix: `记录每个 epoch 的 theory 分支、残差头、共享 encoder、组件 head 的梯度范数曲线，并标注 phase 切换和 LR。`
  - Why: 判断“梯度死亡”需要动态证据；单点 GradNorm 无法区分早期死亡、后期衰减、梯度裁剪或学习率过低。

- **L38** `[high/statistical_validity]` Batch 3 的 alpha 结论来自单次运行，且 alpha=0.5 与 alpha=0.3 的聚合 MAE 差距很小，缺少不确定性估计。
  - Fix: `对每个 alpha 运行多种子，报告 paired seed 差值、Cohen's d 或 bootstrap CI，并给出排序稳定性。`
  - Why: 0.001951 vs 0.001985 这类差异可能落在训练随机波动内，不能直接作为最优点。

- **L40** `[high/physics_interpretation]` alpha=0.3 和 alpha=0.5 的 Load/Batt 均固定在约 0.0145/0.02135，与 no_temp_s2026、d512 崩溃数值高度一致，可能不是普通性能退化，而是物理分支输出进入常数解、裁剪边界或归一化反变换上限。
  - Fix: `检查组件输出分布、反归一化范围、激活饱和、clamp/softplus 边界、组件损失梯度路径，并报告每个组件的均值、方差、边界命中率。`
  - Why: 相同崩溃数值跨配置重复出现，说明存在共同机制或实现边界，不能只用“容量比失衡”解释。

- **L40** `[high/physics_interpretation]` alpha=0.5 聚合最优但组件崩溃，说明残差头可能绕过物理分支；当前未量化残差头贡献比例和物理分支可辨识性。
  - Fix: `报告 theory 输出、residual 输出、final 输出的方差占比、相关性、能量分解和残差均值漂移；增加限制残差容量或残差正则的对照。`
  - Why: 如果残差头完全接管，聚合指标改善不能说明模型学到了物理结构。

- **L51** `[high/overfitting]` Batch 4 使用 N44，而 Batch 1-3 使用 C07，跨数据集或场景的比较被混在同一个推荐链中。
  - Fix: `明确 C07 与 N44 的数据分布、任务难度、划分方式和指标尺度；核心推荐 e3/detach/alpha 应在两个场景分别复现。`
  - Why: 跨批次可比性不足时，不能把 C07 的 detach 结论和 N44 的 e3 结论直接合成统一推荐。

- **L55** `[medium/statistical_validity]` Best Val MSE 数值约 0.38-0.40，而 Test MSE 约 8e-6，量级不一致，缺少指标定义和归一化说明。
  - Fix: `说明 Val MSE 与 Test MSE 是否处于同一尺度、是否经过反归一化、是否是不同目标；若不同，不应直接用“Val 突破但 Test 退化”表述。`
  - Why: 量级不一致会削弱 Val/Test 背离判断，也可能隐藏数据处理或日志口径问题。

- **L57** `[high/overfitting]` d512 的 Val MSE 更优但 Test MAE 退化，被解读为过拟合，但缺少训练集指标、验证曲线、早停 epoch 和多种子复现。
  - Fix: `补充 train/val/test 曲线、early stopping 触发轮次、最终 checkpoint 与 best-val checkpoint 指标，并在多 seed 下验证。`
  - Why: 单个 d512 结果也可能来自验证集偏差、checkpoint 选择、学习率调度差异或数据划分问题，不一定是过拟合。

- **L58** `[medium/recommendation]` e3 推荐隐含假设是“可接受 2.3% aggregate MAE 代价以换取 6.7% Theory MAE 改善”，但没有给出业务目标权重。
  - Fix: `定义模型选择准则，例如 aggregate MAE、Theory MAE、组件 MAE、ResMean 的加权目标或 Pareto 前沿。`
  - Why: 若最终应用重视聚合预测，e3 未必优于 baseline；若重视物理可解释性，则需要明确代价函数。

- **L59** `[medium/recommendation]` d512_e3 与 e3 很接近，但没有分析深度和宽度的交互项，也没有测试 d384/e4 等中间配置。
  - Fix: `补充 d_model={192,256,384,512} 与 e_layers={2,3,4} 的小型网格或响应面实验。`
  - Why: “宽度有害、深度有益”可能只在 d512 或 e3 这两个点成立，无法推出单调结构规律。
