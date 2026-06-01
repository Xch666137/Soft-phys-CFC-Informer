# Peer Review Consensus Report

- File: `results/dual-draft/training_analysis_target.md`
- Reviewers: **claude** vs **codex**
- Matching rule: loose (v2: category equivalents + range overlap), tolerance=2

## Summary
| Bucket | Count |
|---|---:|
| Consensus (both flagged) | 5 |
| Only claude | 2 |
| Only codex | 6 |
| **Total unique** | **13** |

`claude` reported 7 raw issues; `codex` reported 11 raw issues.

## 1. Consensus Issues (5)
_Both reviewers independently flagged the same location + (compatible) category._

### L26 — `convergence_analysis` (severity: high/medium)
- **claude**: DBC 版本 Phase 1 初始 loss (~980) 比非 DBC 版本 (~5000-6000) 低 5-6 倍。根因：DBC 残差 MLP 的恒等映射初始化让首个 forward pass 输出接近 0（标准化后的 target 均值），避免了浅层 MLP 随机权重的巨大方差。这与 ResNet 理论一致——残差学习将优化问题从 F(x)=y 转化为 F(x)+x=y，初始化时 F(x)≈0 天然接近解。佐证：V5.4b（浅层MLP+新cw/tw）首步6177反证并非 cw/tw 差异造成。
  - Fix: `在所有版本中强制使用 DBC 架构。浅层 MLP 的高初始 loss 意味着 Phase 1 的早期 epoch 主要花在修正初始偏差而非学习数据模式，浪费了 warmup 阶段的 LR。`
- **codex**: 所有版本 best 后 stuck 19-20 epochs，很可能反映 early-stopping patience=20 的配置，而不一定是训练机制导致的收敛停滞。当前缺少 best epoch、LR、train loss 和 validation loss 的后续轨迹。
  - Fix: `报告 best epoch、stop epoch、patience、LR 曲线、train/val loss gap；尝试 best 后降低 LR 或 warm restart，验证是否确实无法继续改善。`

### L32 — `phase_transition` (severity: high/medium)
- **claude**: Phase 2 reset 改善幅度分解：DBC 贡献约 64%，新 cw/tw 贡献约 42%（独立可加性验证：0.0154+0.0102≈0.0256 vs 实测 0.0242）。V5.4 需要两步 (0.4185→0.3949→0.3872) 而 V5.3/V5.4b 只需一步，说明 DBC 使 Phase 2 的 loss landscape 更宽缓——需要多一步收敛但能到达更低的极小值。
  - Fix: `Phase 2 首次改善后应给至少 3-5 个 epoch 的缓冲窗口再开始 early stop count。当前 early_stop_start_epoch=20 在 V5.4 场景下足够，但如果 Δ 步数随架构复杂度增加，需要扩展缓冲。`
- **codex**: Phase 2 Δ 被直接比较为 reset 收益，但各版本进入 Phase 2 的起点不同，且 reset 同时包含 selective detach、optimizer momentum 清零和 LR 复位，无法归因。
  - Fix: `做 2x2x2 消融：detach on/off、momentum reset on/off、LR reset on/off；同时报告相对改善率、进入 Phase 2 的起点和 reset 后第 1/2/3 个 epoch 的指标。`

### L37 — `methodology_flaw` (severity: medium/high)
- **claude**: 所有版本 Phase 2 best 后全部 stuck 19-20 epochs 触发 early stop。Optimizer reset 提供了一次性盆地逃逸，但 reset 后 LR 从峰值快速衰减 (cosine T0=8)，在找到新 best 后 LR 已衰减到过低水平，无法继续探索。这不是 plateau 问题而是 LR 调度问题。
  - Fix: `方案 A：Phase 2 使用 constant LR (不衰减) 或更长的 cosine 周期 (T0≥20)。方案 B：使用 SWA (Stochastic Weight Averaging) 收集 Phase 2 best 后的多个 checkpoints 做平均。方案 C：Phase 2 使用周期性 cosine 重启 (T_mult=2)，每次重启到峰值 LR 给模型多次 escape 机会。`
- **codex**: Batt Power MAE 和 Batt SOC MAE 在五个版本中完全相同到 5 位有效数字，和架构/训练策略变化不相称，疑似评估代码复用了固定基线、字段未更新、模块冻结或指标被强制裁剪。
  - Fix: `审查 evaluator 的指标来源、batch 聚合和日志写入；输出每个版本的 battery prediction 样本、均值方差、误差直方图，并确认是否使用了同一 checkpoint 或缓存。`

### L45 — `methodology_flaw` (severity: low/medium)
- **claude**: Test MAE 虽然 V5.4 最优 (0.001909)，但 Theory MAE 与 V5.4a (0.002468 vs 0.002414) 差异仅 2.2%，Residual std 差异 1.7%。考虑到 V5.4a 的 Theory MAE 和 Residual std 都是最优，V5.4 的 0.001909 可能部分归因于其对 net_mse 的过优化（Test MSE 差距 7.26e-06 vs 7.77e-06 = 6.6% 但 Theory 反而差了 2%）。这在物理引导模型的评价体系中是一个矛盾——更好的净注入预测是以轻微牺牲物理一致性为代价的。
  - Fix: `在 test 报告中增加一个综合指标：Theory-Net Tradeoff Ratio = ΔTheory_MAE / ΔNet_MAE。当该比率 > 1 时说明 Net 改善是以超比例 Theory 退化为代价，V5.4 的比率为 (0.002468-0.002414)/(0.001909-0.002110) = 0.054/0.201 = 0.27 —— 实际上 Net 改善远超 Theory 退化，tradeoff 是可接受的。但应持续监控此指标。`
- **codex**: 第 49 行称 V5.4a Val 最差，但第 46 行排名显示 V5.2 的 Val Net MSE 为 0.4089，差于 V5.4a 的 0.3961。该表述范围不清，容易导致错误结论。
  - Fix: `改为“在 V5.3/V5.4/V5.4a/V5.4b 四个 reset 版本中 Val 最差”，或若比较全部五版则改为“Val 第四，仅优于 V5.2”。`

### L53 — `generalization_gap` (severity: high/high)
- **claude**: Val/Test rank 倒挂：V5.4a Val 最差 (0.3961) 但 Test 排第三 (0.002110)，V5.4b Val 排第三 (0.3955) 但 Test 垫底 (0.002133)。V5.4a 的 DBC+旧cw/tw 组合产生最强正则化——较高的 val loss 是欠拟合（非过拟合），反而泛化更好。V5.4b 的浅层 MLP+新cw/tw 导致残余 std 最高 (0.002437)，表明 residual 分量的 test 方差大，是选择性过拟合的典型信号。
  - Fix: `1) 训练时增加 val set 的理论一致性约束（而非仅在 train 上计算 theory loss）。2) 对 V5.4b 类配置，添加 val 上的 res_reg 作为 early stop 辅助指标。3) 在 test 报告中去重 Val 排序——当前 Val Net MSE 与实际泛化能力的相关性为 0.6 (Spearman)，不可靠。`
- **codex**: Val Net MSE 排名和 Test MAE 排名不一致，且 Test MSE 排名又与 Test MAE 不完全一致，说明验证目标与最终目标可能不匹配，或 val/test 分布存在差异。
  - Fix: `在 validation 上同步报告 Test 表中的全部指标，使用同一反归一化尺度；增加分场景/分负荷段评估，并用最终选择指标重新定义 early stopping。`

## 2. Only from claude (2)

- **training_analysis_target.md:30-35** `[medium/phase_transition]` V5.2 无 optimizer reset 的 Phase 1 best (0.4089) 即最终 best，Phase 2 零改善。这反证了 optimizer reset 是 Phase 2 产生改善的必要条件——没有 momentum 清零，Phase 1 的动量惯性将模型锁定在 Phase 1 basin，即使用了 selective detach 和不同 loss 权重也无法 escape。
  - Original: `V5.2: ...→0.4089 (No Phase 2 reset) → Best=0.408893, stuck 20 ep`
  - Fix: `Optimizer reset 应作为 curriculum training 的标准组件写入 CLAUDE.md 设计原则。任何多阶段训练方案，如果 Phase 间 loss landscape 不同，必须 reset 动量。`
  - Why: V5.2 vs V5.3-5.4b 的对比构成了一个干净的 ablation: optimizer reset 是 Phase 2 有效的必要条件。

- **training_analysis_target.md:60-61** `[high/architectural_insight]` 电池指标 Batt Power MAE 和 Batt SOC MAE 在全五个版本中精确锁定 (0.02135/0.02011)，不受 DBC、cw/tw、optimizer reset 任何改动影响。这强烈暗示电池预测子网络已收敛到 architecture-level 的容量上限，而非 loss 或优化问题。Physics Layer 的 battery 分支使用固定物理参数 (P_max, E_max, roundtrip efficiency) 做 SOC 递推，这些参数来自 training set 全局统计且对所有版本相同——因此电池预测质量完全由物理参数精度决定。
  - Original: `Batt Power MAE = 0.02135 for ALL versions. Batt SOC MAE = 0.02011 for ALL versions.`
  - Fix: `要突破电池天花板，唯一路径是提高物理参数精度：(1) 使用 per-portfolio 而非全局 P_max/E_max；(2) 引入电池老化/温度补偿模型更新物理参数；(3) 在训练中联合优化物理参数（当前是固定的），例如将 R_roundtrip 作为可学习参数。`
  - Why: Batt Power MAE (0.02135) 是 Net MAE (0.001909) 的 11 倍，SOC MAE (0.02011) 是 10 倍。电池是整个模型最大的单一误差源。如果物理参数的 10% 精度提升能带来电池 MAE 的 10% 改善，整个 Net MAE 可下降 ~0.001 量级——这是当前最大的可优化方向。

## 3. Only from codex (6)

- **轨迹记录格式** `[medium/convergence_analysis]` 训练轨迹只列出若干数值，没有 epoch 编号、loss component、checkpoint 选择点和是否为同一评估尺度，难以判断“1 步/2 步改善”的真实含义。
  - Original: `格式: 版本 → Phase 1 last → Phase 2 改善 → best；V5.4: inf→980→11.4→0.503→...`
  - Fix: `改成结构化表格：version、epoch、phase、val_net_mse、train_net_mse、loss components、LR、reset flag、checkpoint flag。`
  - Why: 当前轨迹适合概览，但不足以支撑 phase transition 和收敛机制分析。

- **训练 Val Net MSE 轨迹** `[high/architectural_insight]` DBC 版本的早期 loss 明显更低，但这个优势在 reset 前只收敛到约 0.4185/0.4186，并没有单独转化为最优 Test MAE。当前叙述容易把 5-6 倍初始 loss 降低解释为架构表达能力提升，实际也可能来自输出尺度、残差捷径、初始化或参数量差异。
  - Original: `V5.4: inf→980→11.4→...→0.4185 |RESET| ... Best=0.387212；V5.4a: inf→980→13.9→...→0.4186 |RESET| ... Best=0.396112`
  - Fix: `补充参数量、初始化分布、输出均值/方差、各 loss component 的 epoch 级曲线；增加 DBC 等参数量对照或冻结/去残差捷径实验。`
  - Why: DBC 的关键发现是降低初始优化难度，但现有数据不足以证明原因是深层电池上下文建模本身。需要排除尺度和容量带来的伪收益。

- **V5.4 / V5.4a / V5.4b 消融对比** `[medium/architectural_insight]` DBC 与新 cw/tw 的收益呈非加性互动：DBC-only Δ=0.0225，cw/tw-only Δ=0.0173，合并后 Δ=0.0313，小于二者简单相加。说明两者可能共享同一改善通道或存在权重冲突。
  - Original: `V5.4 Δ=0.0313；V5.4a Δ=0.0225；V5.4b Δ=0.0173`
  - Fix: `按 loss component 拆分 Δ，报告 net loss、theory loss、residual loss、battery loss 的贡献；额外画 DBC × cw/tw 的交互表，而不是只给单一 Val Net MSE。`
  - Why: 这是隐藏的结构性发现：组合版本最好，但并非线性叠加，后续调参应围绕交互项而不是单独增强某个模块。

- **SOC Viol 指标** `[low/methodology_flaw]` SOC Viol 全部为 0%，该指标在当前设置下没有区分度，可能因后处理 clipping 或约束过宽而饱和。
  - Original: `SOC Viol | 0% | 0% | 0% | 0% | 0%`
  - Fix: `补充 SOC margin 的 min/p1/p5、clip 前 violation rate、最大越界幅度，以及是否存在后处理裁剪。`
  - Why: 0% violation 本身是好结果，但如果指标饱和，就不能支持版本间物理一致性比较。

- **单次实验结果** `[high/methodology_flaw]` 多个版本的 Test MAE 差异非常小，例如 V5.3=0.002102 与 V5.4a=0.002110 只差 8e-06，V5.4b 与 V5.2 完全持平。单 seed 结果不足以支持精细排名。
  - Original: `V5.3 Test MAE 0.002102；V5.4a Test MAE 0.002110；V5.4b Test MAE 0.002133；V5.2 Test MAE 0.002133`
  - Fix: `至少运行 3-5 个随机种子，报告 mean±std、paired bootstrap CI，并只把超过随机波动的差异称为改进。`
  - Why: 当前结论中的第二、第三、垫底等排序可能落在训练噪声内。

- **physics metrics 与预测指标的权衡** `[medium/generalization_gap]` V5.4a 拥有最优 Theory MAE 和最低 Residual std，但 Test MAE 不如 V5.4/V5.3；V5.4 拥有最优 Test MAE 但不是最优 physics residual。说明物理一致性和预测误差之间存在 Pareto 权衡。
  - Original: `V5.4 Test MAE 0.001909；V5.4a Theory MAE 0.002414，Residual std 0.002318`
  - Fix: `用 Pareto front 同时比较 Test MAE、Theory MAE、Residual std；明确主目标，并按业务代价选择模型，而不是用单一排名。`
  - Why: 该发现对后续设计很关键：继续压 residual 未必提升净注入预测，loss 权重需要按最终目标调优。
