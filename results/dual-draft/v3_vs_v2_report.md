# Peer Review Consensus Report

- File: `results/physformer_v3 — V3 training results vs V2 comparison`
- Reviewers: **claude** vs **codex**
- Matching rule: loose (v2: category equivalents + range overlap), tolerance=2

## Summary
| Bucket | Count |
|---|---:|
| Consensus (both flagged) | 0 |
| Only claude | 8 |
| Only codex | 7 |
| **Total unique** | **15** |

`claude` reported 8 raw issues; `codex` reported 7 raw issues.

## 1. Consensus Issues (0)
_Both reviewers independently flagged the same location + (compatible) category._

_(none)_

## 2. Only from claude (8)

- **V3 overall — 缺少消融实验无法归因改善** `[P1/target_misalignment]` V3 包含 7 项改动，但最大的改善——Theory MAE 从 17.01→4.87 kW（-71.3%）和 Val MSE 从 0.420→0.387（-7.9%）——无法归因到具体改动。可能的贡献者：(a) film_scale 0.2→0.5（物理特征调制增强），(b) theory_net 32-dim 投影（表达能力提升），(c) portfolio_manifest split 策略（之前 val/test 的 portfolio 索引错乱导致 theory_net 表现差），(d) soft wind gates（梯度传导改善），(e) PV 温度对称化，(f) Loss 简化去噪，(g) WarmRestarts 调度器。需要至少 3-4 个关键消融来拆分 71% theory_mae 改善的贡献。
  - Fix: `运行 priority-ordered 消融：(1) film_scale=0.2 → 0.5（其他 V3 配置固定），(2) theory_proj_dim=1 → 32，(3) boolean_wind → soft_sigmoid，(4) CosineAnnealing → WarmRestarts。每个消融跑 25 epoch，比较 Theory MAE 和 Val MSE。这是论文 ablation table 的核心内容。`
  - Why: 71% theory_mae 改善太显著，审稿人会问'哪个改动贡献了多少'。无法回答意味着论文核心贡献（物理引导效果）无法被严谨支撑。消融实验是拒绝 desk-reject 的最低要求。

- **V3 metrics.json — Theory MAE 4.87kW vs Residual std 6.55W 矛盾** `[P1/metric_validity]` Theory MAE=4.87kW, Residual std=6.545kW (0.006545MW, 不是 6.55W)。修正单位后两者处于同一量级 (ratio≈0.74)，说明残差分支在物理模块基础上做了实质性的数据驱动修正。需要进一步区分残差的系统性偏置 (residual_mean) vs 随机波动 (residual_std)。如果 |residual_mean| 远大于 residual_std，说明残差主要是系统性修正（物理模块存在固定偏差）；如果两者相当，说明残差是逐点校正。当前代码只报告 std 不报告 mean，无法区分这两种情况。
  - Fix: `(1) 在 test() 中增加 residual_mean_real_mw 指标（代码已修复）；(2) 重新运行测试获取 residual_mean，确认残差成分；(3) 在论文中报告 theory_mae、residual_mean、residual_std 三者，完整刻画物理+残差的协同机制。`
  - Why: 6.545kW 的 residual_std 与 4.87kW 的 theory_mae 同量级，说明物理引导 + 残差修正的分工合理：物理模块提供粗粒度的物理合理初值 (theory_mae=4.87kW)，残差网络进行精修 (std=6.545kW)。这比之前误读的 '物理模块几乎无用' 更有利于支撑论文的 physical guidance 核心宣称。

- **V3 metrics — Test MSE denorm 一致性** `[P2/metric_validity]` Best Val MSE=0.3867（归一化空间），Test MSE=7.46e-6 MW²。需要验证两者一致性：Test MSE 应约等于 Val MSE × target_std²。若 target_std ≈ sqrt(7.46e-6 / 0.3867) ≈ 0.00439 MW = 4.39 kW——这是一个合理的 target_std 值（net_injection 标准差的典型范围）。但需要调用者显式确认：实际 denorm 使用的 target_std 是多少？Val MSE × target_std² 是否与 Test MSE 匹配？若不匹配，可能是 (a) test/val split 分布不同（portfolio_manifest 策略）或 (b) val loss 使用 per-batch 平均 vs test 使用全局平均。
  - Fix: `从 scaler 对象中读取 target_std，计算 expected_test_mse = best_val_mse * target_std²，与实测 Test MSE=7.46e-6 比较。差异 >15% 需要在论文中解释 split 分布差异。`
  - Why: Val→Test 的 denorm 一致性是实验可信度的基础检查。

- **V3 metrics — Test MAE 1.93kW vs RMSE 2.73kW 比值** `[P3/metric_validity]` MAE/RMSE = 1.93/2.73 = 0.707。对于正态分布残差，理论比值 ≈ sqrt(2/π) = 0.798。实际比值 0.707 低于理论值，说明残差分布有比正态更轻的尾部（fewer large errors）。这是一个正面信号——模型对大误差控制较好。但同时也意味着残差分布非正态，使用 RMSE 作为主指标是合理的（对较大误差更敏感）。论文中可以简单提及 MAE/RMSE 比值作为误差分布特征的证据。
  - Fix: `在论文中可选项：报告 MAE/RMSE 比值，讨论残差分布特征。不强制——这是一个 minor 加分项。`
  - Why: MAE/RMSE 比值偏离正态预期说明误差分布有结构，这对于物理引导模型是合理且期待的。

- **V3 train.log — Val Loss 始终等于 Val MSE** `[P3/metric_validity]` 所有 epoch 中 Val Loss == Val MSE（精确到 6 位小数），说明 soc_bounds_loss（soc_weight=0.1）在归一化空间中对总 loss 的贡献 < 1e-6。物理约束 loss 对模型选择完全无影响。虽然是设计意图（MSE 优先），但这意味着物理约束仅是结构性先验（网络架构中的 battery branch），而非来自 loss 的软约束。论文中需说明 soc_weight 设置的依据及物理约束在训练中的实际作用方式。
  - Fix: `在论文中：(1) 报告 Val Loss 构成（MSE 占比 >99.9999%，SOC loss 占比可忽略）；(2) 说明物理约束主要通过结构性先验（battery branch, portfolio embedding, theory_net）而非 loss 项起作用；(3) 或运行 soc_weight=0 的对照实验，证明即使 loss 无约束，结构性先验仍能保证 SOC 合规。`
  - Why: 审稿人会问'soc_weight=0.1 是否起作用'，需要提前回应。SOC violation=0 可能来自结构性先验而非 loss 约束——这是加分项需要明确展示。

- **V3 train.log — Epoch 1 Train loss=720948 vs Val MSE=2.95** `[P2/training_dynamics]` Epoch 1 的 Train loss（720,948）与 Val MSE（2.95）之间存在 ~244,000 倍的差距。与其他 epoch 对比（epoch 5: Train=0.427 vs Val=0.539，差距仅 21%），epoch 1 的比例极端异常。可能原因：(a) Train loss 包含预归一化的大值异常样本（而在 Val 中不存在）；(b) warmup 从 0.2x 起步（LR=3.6e-5），模型在 epoch 1 的大多数 batch 上几乎是随机的，少量正常 batch 拉低了平均 loss；(c) 数据集中存在极端 portfolio（高波动或大容量），在训练集中出现而在验证集中缺失。
  - Fix: `(1) 计算 epoch 1 per-portfolio 的 Train loss 分布，确认是否存在 1-2 个异常 portfolio 贡献了大部分 loss；(2) 检查数据划分后 train/val 集的 portfolio 容量分布是否均衡；(3) 如果异常是预期行为（大 portfolio 初始难拟合），在论文中说明并展示 epoch 1→5 的快速收敛过程。`
  - Why: 244,000x 的 train/val gap 在 epoch 1 是极度异常的。如果不解释，可能暗示数据泄露（val 泄露到 train 导致 val 损失偏高或偏低等）、数据划分问题、或 loss 计算 bug。但观察 epoch 2 Train=1129 → epoch 3 Train=129，说明模型快速适应，更可能是某些 portfolio 的初始误差极大。

- **V3 — WarmRestarts 边际收益递减，第3次无效** `[P2/training_dynamics]` WarmRestarts 配置 T_0=15, T_mult=1 产生 3 次 LR 重启 → 4 个 cosine 周期。收益递减明显：Cycle #1 best=0.4052 (epoch 13)，Cycle #2 best=0.3918 (epoch 30, +3.3%)，Cycle #3 best=0.3867 (epoch 37, +1.3%)。Cycle #4 开始于 epoch 51（第3次 LR 重启后）但训练截断，未来得及产生新 best。这符合预期——每次重启探索的区域愈发平坦，收益衰减是自然现象。论文中需明确区分 'LR restart'（事件）和 'cosine cycle'（周期），避免混淆。
  - Fix: `论文中展示 Val MSE vs epoch 曲线，标注 3 次 LR 重启点和 4 个周期边界。报告各周期 best：(1) Cycle #1: 0.4052 (epoch 13), (2) Cycle #2: 0.3918 (epoch 30), (3) Cycle #3: 0.3867 (epoch 37), (4) Cycle #4: 仅 1 epoch 即截断。强调 WarmRestarts 贡献了 0.4052→0.3867 ≈ 4.6% 的 Val MSE 改善。`
  - Why: 不区分 restart 贡献 vs 架构修复贡献导致归因混淆。审稿人需要看到每个组件独立的贡献度。

- **V3 train.log — Training truncated at epoch 51/100** `[P1/training_dynamics]` 训练在 epoch 51 中断（early stopping counter=14/25 未触发），仅完成了 51/100 epoch。第3次 LR 重启发生在 epoch 50，Cycle #4 刚开始 1 个 epoch 即被截断。前三次周期的效果：Cycle #1（epochs 1-20）best=0.4052 (epoch 13)，Cycle #2（epochs 21-35）best=0.3918 (epoch 30, +3.3%)，Cycle #3（epochs 36-50）best=0.3867 (epoch 37, +1.3%)。如果 Cycle #4 有类似递减收益，Val MSE 可能降至 ~0.383-0.385。中断原因可能是容器超时或手动 kill（非 early stopping 触发）。
  - Fix: `(1) 在新容器中从 checkpoint epoch 37（best）恢复训练，运行剩余 49 epoch 以完成 100 epoch 计划；(2) 在论文中标注训练停止 epoch 及原因，如不补跑则在消融分析中注明限制。`
  - Why: 未完成的训练导致 V3 的最佳性能可能被低估，也使得第4个 cosine 周期的收益无法评估。

## 3. Only from codex (7)

- **v3_vs_v2_target.md:4** `[low/style]` 句子带有审阅指令口吻，且 baseline 前缺少冠词，不适合作为论文正文。
  - Original: `Review the PhysFormer V3 training results against V2 baseline. Identify what improved, what regressed, and what can be further optimized.`
  - Fix: `We compare the PhysFormer V3 training results with the V2 baseline and identify improvements, regressions, and remaining optimization opportunities.`
  - Why: 期刊正文应使用陈述式学术表达，避免命令式任务说明。

- **v3_vs_v2_target.md:7** `[medium/latex_command]` 多处代码式标识含未转义下划线，作为 LaTeX 正文会报错或排版异常。
  - Original: `film_scale; theory_net; portfolio_manifest; soc_transition_loss + anti_overlap_loss; T_0=15, T_mult=1; soc_weight=0.1`
  - Fix: `将代码/配置标识写成 \texttt{film\_scale}、\texttt{theory\_net}、\texttt{portfolio\_manifest}、\texttt{soc\_transition\_loss}、\texttt{anti\_overlap\_loss}、\texttt{soc\_weight}；超参数写成 $T_0=15$ 和 $T_{\mathrm{mult}}=1$。`
  - Why: LaTeX 文本模式下裸下划线不是合法普通字符，模型配置名也应以等宽字体区分。

- **v3_vs_v2_target.md:13** `[high/formula_consistency]` Warm-restart 的数量、周期命名和收益归因不自洽。
  - Original: `7. Scheduler: CosineAnnealing → CosineAnnealingWarmRestarts (3 restarts, T_0=15, T_mult=1)
### Restart #1 (epoch 1-20, cosine cycle)
### Restart #3 (epoch 36-50)
- Epoch 37: Val=0.3867 (GLOBAL BEST), LR=9.6e-5
### Restart #4 (epoch 51+)
- WarmRestarts: restart #1 benefit +3.3%, restart #2 benefit +1.3%, restart #3 no benefit yet
- Training truncated at 51/100 epochs, 3rd restart barely started`
  - Fix: `统一改为 cycle 表述：Cycle #1 (epochs 1--20) best=0.4052；Cycle #2 (21--35) best=0.3918, improvement=3.3%；Cycle #3 (36--50) best=0.3867, improvement=1.3%；Cycle #4 started at epoch 51 after the third LR restart and had no new best yet.`
  - Why: 当前写法把 cycle 与 restart 混用，并称 restart #3 尚无收益，但 epoch 37 已给出全局最优，会误导收益归因。

- **v3_vs_v2_target.md:16** `[medium/format]` 多处数学和单位表达采用代码/纯文本写法，不符合 EPSR/LaTeX 风格。
  - Original: `- Test MSE: 9.0e-6 MW²
- Val Loss == Val MSE for ALL epochs (soc_weight=0.1 contributes <1e-6)
- Epoch 1 Train/Val gap: 244,000x (720948 vs 2.95)
| Metric | V2 | V3 | Δ |`
  - Fix: `例如写成 $9.0\times10^{-6}\,\mathrm{MW}^2$、$\Delta$、$\mathrm{validation\ loss}=\mathrm{validation\ MSE}$、$244{,}000\times$；将 ALL 改为 all。`
  - Why: 科学计数法、单位、等号和倍数应进入数学模式；全大写强调和代码式 == 不适合期刊正文。

- **v3_vs_v2_target.md:16** `[medium/symbol_undefined]` MSE、MAE、RMSE、SOC、LR 等缩写首次出现时未定义。
  - Original: `- Test MSE: 9.0e-6 MW²
- MAE: 2.21 kW
- RMSE: 2.96 kW
- SOC violation: 0.0
- Epoch 1: Train=720948, Val=2.9517, LR=3.6e-5`
  - Fix: `首次出现时写成 mean squared error (MSE)、mean absolute error (MAE)、root mean squared error (RMSE)、state of charge (SOC)、learning rate (LR)。`
  - Why: EPSR 面向跨领域读者，关键指标和控制变量缩写应在首次使用时完整定义。

- **v3_vs_v2_target.md:28** `[high/formula_consistency]` Residual std 的单位换算错误，0.006545 MW 应为 6.545 kW，不是 6.545 W。
  - Original: `- Residual std (MW): 6.545 W`
  - Fix: `- Residual std: 0.006545 MW (6.545 kW)`
  - Why: 该错误会使残差标准差低估 1000 倍，直接影响读者对模型误差规模的理解。

- **v3_vs_v2_target.md:50** `[high/formula_consistency]` Early-stopping counter 的计数前后不一致。
  - Original: `- Epoch 38-50: Early stopping counter 1→14, no improvement
- Epoch 51: Val=0.3967, counter=15/25
- **Training stopped at epoch 51** (not early-stopping — counter 14 < patience 25)`
  - Fix: `- Epoch 38-50: Early stopping counter 1→13, no improvement
- Epoch 51: Val=0.3967, counter=14/25
- **Training stopped at epoch 51** (manual truncation; not early stopping, because counter 14 < patience 25)`
  - Why: Epoch 38--50 共 13 个未提升 epoch，epoch 51 后 counter 应为 14/25；现有写法会造成训练停止原因判断错误。
