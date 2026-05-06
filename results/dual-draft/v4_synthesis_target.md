# PhysFormer V3→V4.x 训练综合分析 — 优化方向与创新点

## 完整结果矩阵

| 指标 | V3 | V4 (Phase 1) | V4.1 (+gate) | V4.2 (归一化loss) | 最优 |
|------|----|-------------|-------------|-----------------|------|
| Test MSE | 7.46e-6 | **7.35e-6** | 7.68e-6 | 8.04e-6 | V4 |
| MAE (kW) | **1.932** | 1.976 | 2.002 | 2.048 | V3 |
| Theory MAE (kW) | 4.874 | 3.811 | 4.966 | **2.981** | V4.2 |
| Theory RMSE (kW) | — | 5.185 | 7.183 | **3.874** | V4.2 |
| Residual std (kW) | 6.545 | 4.312 | 5.076 | **2.695** | V4.2 |
| Residual mean (kW) | — | 2.259 | -3.336 | **-1.151** | V4.2 |
| Val MSE | 0.387 | 0.381 | 0.379 | **0.378** | V4.2 |
| Converged epoch | 37 | 11 | 8 | 36 | V4 |
| SOC violation | 0 | 0 | 0 | 0 | All |

### Per-Component Theory MAE (kW)
| 分量 | V4 | V4.1 | V4.2 |
|------|-----|------|------|
| Wind | 0.83 | 0.46 | 0.36 |
| PV | 4.00 | 3.79 | 2.45 |
| Load | 14.71 | 2.04* | 2.09* |
| Battery P | 21.35 | 1.53* | 1.69* |

*V4.1+ component MAEs are in normalized space (not directly comparable to V4 MW)

## V4 vs V4.2 的矛盾

V4.2 在物理质量（Theory MAE -38.8% vs V3）和收敛精度（Val MSE 0.378）上全面领先，但最终 MAE 反而退化（+6.0% vs V3）。核心矛盾：**物理先验越好 → 最终预测不一定越好**。

## 训练中获得的经验法则

1. **Component loss 在 MW 空间（V4 weight=0.05）效果优于归一化空间（V4.2 weight=0.02）**——MW 空间的 component supervision 在改善 theory 的同时保持了 residual 的补偿能力
2. **Sigmoid gate 有害（V4.1 regression）**——减少 residual 自由度迫使 theory_net 承担更多，但共享编码器的梯度耦合导致理论退化
3. **time_proj 可能有害**——预训练 decoder 的 query 分布被随机初始化破坏
4. **patience=20 + early_stop_start=20 有效**——V4.2 的两次 WarmRestart 都找到了新的最优
5. **Load 是瓶径**——Load theory MAE (14.7kW) 是 Wind (0.83kW) 的 18 倍

## 待回答的战略问题
1. 继续改进 theory_net（如更好的 PV/Wind physics、Load 独立建模）还是改进 residual（如 uncertainty-aware gating）？
2. 物理引导 vs 数据驱动的 tradeoff：当前 V4 是最佳平衡点，如何在不牺牲物理质量的前提下提升最终精度？
3. 创新点方向：Dual-Stream（PV/Wind physics + Load data-driven）还是 component-consistent residual？
4. 论文的新颖性定位：当前 PhysFormer 的核心贡献是什么？与 iTransformer/PatchTST 的差异化在哪？
