# PhysFormer V5 系列训练动态分析 Target

## 实验背景

PhysFormer 是物理引导 Transformer，用于 VPP 净注入预测。两阶段 curriculum：
- Phase 1 (epoch 1-8): 全梯度流，高 cw/tw
- Phase 2 (epoch 9+): selective detach + optimizer momentum 清零 + LR 复位

五个版本：
- V5.2: curriculum training 初版，无 optimizer reset
- V5.3: selective detach + optimizer reset + res_reg 调整
- V5.4: DeepBatteryContext (DBC, 残差 MLP) + 新 cw/tw + optimizer reset
- V5.4a: 仅 DBC，保留 V5.3 cw/tw
- V5.4b: 仅新 cw/tw，保留浅层 MLP

## 训练 Val Net MSE 轨迹 (逐改善步骤)

格式: 版本 → Phase 1 last → Phase 2 改善 → best

```
V5.2:  inf→5338→14.6→0.555→0.511→0.459→0.4089 (No Phase 2 reset) → Best=0.408893, stuck 20 ep
V5.3:  inf→6125→9.8→0.569→0.525→0.462→0.441→0.4014 |RESET| 0.4014→0.3943 → Best=0.394313, stuck 19 ep
V5.4:  inf→980→11.4→0.503→0.456→0.430→0.4185 |RESET| 0.4185→0.3949→0.3872 → Best=0.387212, stuck 19 ep
V5.4a: inf→980→13.9→0.513→0.449→0.444→0.4186 |RESET| 0.4186→0.4037→0.3961 → Best=0.396112, stuck 19 ep
V5.4b: inf→6177→8.2→0.517→0.493→0.442→0.4128 |RESET| 0.4128→0.3955 → Best=0.395548, stuck 20 ep
```

Phase 2 改善幅度 (Δ):
- V5.3: Δ=0.0071 (1步)
- V5.4: Δ=0.0313 (2步, 是V5.3的4.4倍)
- V5.4a: Δ=0.0225 (2步)
- V5.4b: Δ=0.0173 (1步)

## 最终 Test 指标

| 版本 | Test MAE | Test MSE | Theory MAE | Theory RMSE | Residual std | SOC Viol | Batt Power MAE | Batt SOC MAE |
|---|---|---|---|---|---|---|---|---|
| V5.2 | 0.002133 | 7.95e-06 | 0.002936 | 0.003899 | 0.002920 | 0% | 0.02135 | 0.02011 |
| V5.3 | 0.002102 | 8.21e-06 | 0.002482 | 0.003460 | 0.002403 | 0% | 0.02135 | 0.02011 |
| V5.4 | 0.001909 | 7.26e-06 | 0.002468 | 0.003438 | 0.002357 | 0% | 0.02135 | 0.02011 |
| V5.4a | 0.002110 | 7.77e-06 | 0.002414 | 0.003351 | 0.002318 | 0% | 0.02135 | 0.02011 |
| V5.4b | 0.002133 | 8.17e-06 | 0.002498 | 0.003492 | 0.002437 | 0% | 0.02135 | 0.02011 |

## Val vs Test Rank 不一致

- Val Net MSE rank: V5.4 (0.3872) < V5.3 (0.3943) < V5.4b (0.3955) < V5.4a (0.3961) < V5.2 (0.4089)
- Test MAE rank: V5.4 (0.001909) < V5.3 (0.002102) < V5.4a (0.002110) < V5.4b (0.002133) = V5.2 (0.002133)

V5.4a Val 最差 (0.3961) 但 Test 第三 (0.002110)，且 Theory MAE 最优 (0.002414), Residual std 最低 (0.002318)。

V5.4b Val 第三 (0.3955) 但 Test 垫底 (0.002133)，Residual std 最高 (0.002437)。

## 分析问题

1. Phase 1 初始 loss：DBC 版本(~980) vs 非DBC(~5000-6000) — 为什么残差 MLP 能降 5-6 倍初始 loss？
2. Phase 2 reset Δ 分解：DBC 和 cw/tw 各自贡献多少？为什么 DBC 版本需要 2 步才达最佳？
3. 所有版本 Phase 2 best 后全部 stuck 19-20 epochs → early stop — 是什么机制导致无法继续改善？
4. Val/Test rank 偏差：V5.4a 欠拟合泛化好？V5.4b 过拟合？
5. Batt Power/SOC MAE 全版本锁死 (0.02135/0.02011) — 物理层瓶颈还是度量尺度问题？
