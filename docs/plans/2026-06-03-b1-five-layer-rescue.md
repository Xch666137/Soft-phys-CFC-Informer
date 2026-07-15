# B1 五层根因修复方案

> 日期: 2026-06-03
> 状态: 待用户审批
> 关联: N106-N115, O53-O55, O58-O59, C08, C11, C12

---

## 1. 问题总览

B1（Masked Component Pretraining + Finetune）落后 A1（纯 net MSE from scratch）的根因不是单一 bug，而是 **5 层独立问题叠加**：

| 层级 | 问题 | 影响量级 | 是否已确认 |
|------|------|---------|-----------|
| L1 | MCP 的 λ_net=0.3 导致预训练梯度 77% 优化分量精度，抵消结构学习不足 | 根本性 | 理论推导，待 ablation 验证 |
| L2 | checkpoint 选 total-loss-best 而非 net-loss-best；best_val_net_checkpoint.pth 未保存 | 中等 | 源码确认 |
| L3 | Scaler 在 train+val 上 fit，与 finetune/test 的 train-only scaler 不一致 | 小但系统性 | 源码确认 |
| L4 | OneCycleLR lr=1e-4 在已收敛 basin 上破坏性 warmup | 大 | 日志确认（Epoch 1 退化 3×） |
| L5 | 无 mask token（用物理零替代）+ 预训练忽略 calendar features | 中等 | 源码确认 |

---

## 2. 各层根因分析与修复方案

### L1: λ_net 权重失衡——预训练梯度偏重分量精度

**问题本质**

A1 的核心优势来自 C08 误差抵消机制：`net = load - pv - wind + batt` 的符号结构让各分量误差正负抵消。A1 只优化 `net_mse`，100% 梯度用于学习抵消结构。

MCP 预训练目标（`loss.py:307-317`）：

```python
comp_loss = (per_component_mae * mask.float()).sum() / mask.float().sum()
net_loss = F.mse_loss(outputs["pred_net"], y_target)
total = comp_loss + self.lambda_net * net_loss  # comp 权重 1.0, net 权重 0.3
```

**梯度贡献分析**（假设 comp_loss ≈ 0.3, net_loss ≈ 0.3）：

| 项 | 权重 × 值 | 梯度贡献占比 |
|----|----------|------------|
| comp_loss | 1.0 × 0.3 = 0.30 | **77%** |
| net_loss | 0.3 × 0.3 = 0.09 | **23%** |

**77% 梯度把分量推向独立准确，只有 23% 在学习抵消结构。** 分量独立最优解 ≠ 聚合最优解（C08 的核心发现）。预训练结束后，模型表征是"分量比较准但抵消结构弱"，finetune 切到纯 net_mse 时需要大幅重组表征——但 L4 的高 LR 把这个重组变成了破坏。

**预训练直接测试 MAE = 0.001820** 已经接近 A1 的 0.001811，说明即使 λ=0.3 也能学到部分有用的聚合表征。但 finetune 的 L2/L3/L4 问题把这个优势毁掉了。

**修复方案：λ_net Ablation**

保留 MCP 的 comp_loss + λ_net × net_loss 框架（不删 comp_loss，保留分量预测的物理可解释性），但通过 ablation 找到最优的 λ_net 值，让预训练的梯度贡献更平衡：

```python
# PretrainLoss 保持不变，只调 λ_net
total = comp_loss + lambda_net * net_loss
```

**λ_net ablation 设计**：

| λ_net | comp 梯度占比 | net 梯度占比 | 预期行为 |
|-------|-------------|-------------|---------|
| 0.3（当前） | 77% | 23% | 分量准，抵消弱 |
| 1.0 | 50% | 50% | 分量和聚合平衡 |
| 3.0 | 25% | 75% | 聚合主导，分量正则化 |

**不需要改 loss.py 的代码结构**——`lambda_net` 已经是 PretrainLoss 的参数，只需在 config 中设置不同值。

**改动范围**

| 文件 | 改动 |
|------|------|
| `loss.py` | 无代码改动——lambda_net 已是参数 |
| `configs/physformer_igt_b1_pretrain.yaml` | pretrain_lambda_net: 1.0（默认值） |
| 新增 configs | `physformer_igt_b1_pretrain_lam03.yaml`、`_lam10.yaml`、`_lam30.yaml` |

---

### L2: Checkpoint 选择指标错误

**问题根因**

`pretrain_exp.py` 中存在两个 checkpoint 追踪：

1. **Early stopping**（`pretrain_exp.py:329`）：跟踪 `vali_stats["loss"]`（total loss = comp + λ×net）
2. **best_val_net**（`pretrain_exp.py:321-327`）：跟踪 `vali_stats["net_loss"]`

但日志中 **从未出现 "Val Net improved"** 消息，说明 `best_val_net_checkpoint.pth` 没有被保存到磁盘。最终保存的 `pretrained_checkpoint.pth` 来自 early stopping 的 total-loss-best checkpoint（`pretrain_exp.py:337-339`）。

```python
# 训练结束后加载的是 total-loss-best，不是 net-loss-best
self.model.load_state_dict(
    torch.load(self.checkpoint_path(), map_location=self.device)  # checkpoint_path = early stopping best
)
pretrained_path = self.run_dir / "pretrained_checkpoint.pth"
torch.save(self.model.state_dict(), pretrained_path)  # 保存的是 total-loss-best
```

**修复方案**

L1 保留 comp_loss 后，early stopping 用 total loss 选 checkpoint 是合理的——它选的是"分量和聚合都还行"的平衡点。但需要额外保存 net-loss-best 作为备选。

1. 保留 early stopping 跟踪 total loss（与 MCP 的多任务目标一致）
2. 修复 best_val_net_checkpoint.pth 的保存逻辑（诊断为什么没触发）
3. 额外保存 `best_val_net_checkpoint.pth` 作为 finetune 的备选 checkpoint
4. Finetune 时默认用 total-loss-best，如果效果不佳再试 net-loss-best

```python
# pretrain_exp.py train() 方法

# 保留 early stopping 跟踪 total loss
early_stopping(vali_stats["loss"], self.model, ...)

# 修复 best_val_net 保存：确保 net_loss 被正确追踪
if save_best_val_net and math.isfinite(vali_stats["net_loss"]) and vali_stats["net_loss"] < best_val_net:
    best_val_net = vali_stats["net_loss"]
    torch.save(self.model.state_dict(), best_val_net_path)
    self.logger.info("Val Net improved (%.6f --> %.6f). Saving net-best checkpoint ...",
                     prev_best, best_val_net)
```

```python
# 训练结束后：从 total-loss-best 加载并保存为 pretrained_checkpoint.pth
# （保持当前行为，因为 total-loss-best 对 MCP 预训练是合理的）
self.model.load_state_dict(torch.load(self.checkpoint_path(), map_location=self.device))
torch.save(self.model.state_dict(), pretrained_path)

# best_val_net_checkpoint.pth 已在训练过程中保存
# Finetune config 可通过 pretrained_path 选择用哪个 checkpoint
```

**改动范围**

| 文件 | 改动 |
|------|------|
| `pretrain_exp.py` | 修复 best_val_net 保存逻辑；增加诊断日志 |

---

### L3: Scaler 空间污染

**问题根因**

`data.py` 中 pretrain 和 finetune 的 scaler fit 逻辑不一致：

```python
# data.py:360-363 — pretrain 在 train+val 上 fit scaler
if self.pretraining_mode and self.set_type == 0:
    train_frame = df_raw.loc[df_raw[self.split_col] == "train"]
    val_frame = df_raw.loc[df_raw[self.split_col] == "val"]
    fit_frame = pd.concat([train_frame, val_frame])

# data.py:365 — finetune 只在 train 上 fit scaler
fit_frame = df_raw.loc[df_raw[self.split_col] == "train"]
```

PhysFormeriGT 的 `target_mean/target_std/aux_mean/aux_std` 作为 `register_buffer` 存在 state_dict 中。预训练权重在 train+val scaler 空间上学到，但 finetune 时输入数据是 train-only 归一化空间——系统性分布偏移。

同样，`data.py:291-293` 的 `_split_one_group` 也有 pretraining_mode 扩展 train 的逻辑：

```python
if self.pretraining_mode and self.set_type == 0:
    border2s[0] = num_train + num_val
```

**修复方案**

去掉 pretraining_mode 的 train+val 扩展，pretrain 也用 train-only：

```python
# data.py:360-363 — 删除 train+val 扩展
# 改前
if self.pretraining_mode and self.set_type == 0:
    train_frame = df_raw.loc[df_raw[self.split_col] == "train"]
    val_frame = df_raw.loc[df_raw[self.split_col] == "val"]
    fit_frame = pd.concat([train_frame, val_frame])
# 改后：删除整个 if 分支，走下面的 else

# data.py:291-293 — 删除 pretraining_mode 扩展
# 改前
if self.pretraining_mode and self.set_type == 0:
    border2s[0] = num_train + num_val
# 改后：删除
```

**影响**：预训练数据量从 train+val 减少到 train-only（约 -12.5%），但 scaler 和验证信号变得干净。

**改动范围**

| 文件 | 改动 |
|------|------|
| `data.py` | 删除 ~6 行 pretraining_mode 的 train+val 扩展逻辑 |

---

### L4: OneCycleLR 在已收敛 Basin 上的破坏性

**问题诊断**

B1 finetune 的 LR 曲线存在断崖：

| 时刻 | LR | 来源 |
|------|-----|------|
| 预训练 epoch 46（末尾） | ~1e-7 | OneCycleLR cosine decay |
| Finetune epoch 1（开局） | 4e-6 | OneCycleLR warmup (1e-4/25) |

**日志证据**：B1 finetune Epoch 1 Train Loss = 0.7012，预训练末尾 Train Loss = 0.3004。LR 跳升 40× 导致第一个 epoch 就把预训练权重推离最优 basin 2.3×。

**现代 LLM 的标准做法**

| 阶段 | GPT/LLaMA/BERT | PhysFormer B1 当前 |
|------|----------------|-------------------|
| Pretrain LR | 恒定 或 cosine decay | OneCycleLR cosine, pct_start=0.12 |
| Pretrain→Finetune 断点 | LR 跳到 finetune 的低 LR | LR 从 ~1e-7 跳到 4e-6（40× 跳升） |
| Finetune LR | pretrain peak 的 1/10 ~ 1/100 | 与 pretrain 同 max_lr=1e-4 |
| Finetune schedule | 短 warmup + cosine/linear decay 或 constant | OneCycleLR 重新完整 warmup+decay |

**修复方案：Finetune 用 Constant LR**

```yaml
# B1 finetune config
training:
  learning_rate: 1.0e-5      # pretrain peak 的 1/10
  train_epochs: 10
  patience: 3
  early_stop_start_epoch: 2
  schedule_type: constant     # 新增：不用 OneCycleLR
```

`physformer_exp.py` 中增加 `schedule_type: constant` 分支：

```python
# physformer_exp.py train() 方法
schedule_type = getattr(self.args, "schedule_type", "onecycle")

if schedule_type == "constant":
    scheduler = None  # 不创建 scheduler，LR 恒定
else:
    scheduler = OneCycleLR(...)
```

训练循环中相应跳过 `scheduler.step()`。

**改动范围**

| 文件 | 改动 |
|------|------|
| `physformer_exp.py` | 增加 schedule_type 分支（~15 行） |
| B1 finetune configs | learning_rate 改为 1e-5，增加 schedule_type: constant |

---

### L5: 无 Mask Token + 忽略 Calendar Features

**问题 A：无 mask token**

当前 `igt_model.py:275-276`：

```python
physical_zero_norm = -self.aux_mean[:4] / (self.aux_std[:4] + 1e-6)
```

被 mask 的分量历史被替换为"物理零的 z-score 值"。GRU 看到的是一个合法的零值序列，无法区分"该分量恰好为零"和"该分量被 mask"。

**修复：增加 Learnable Mask Token**

```python
# igt_model.py PhysFormeriGT.__init__
self.mask_token = nn.Parameter(torch.zeros(1, 1, 1))
nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
```

```python
# igt_model.py forward 中 mask 逻辑
# 改前
x_comp[:, :, :4] = torch.where(mask, physical_zero_norm.expand_as(x_first4), x_first4)

# 改后
x_comp[:, :, :4] = torch.where(mask, self.mask_token.expand_as(x_first4), x_first4)
```

`mask_token` 是一个标量参数，广播到所有被 mask 的分量和时间步。比每个分量一个 mask token 更简洁（只有 4 个分量，标量足够区分 mask vs 非 mask）。

**问题 B：忽略 calendar features**

`pretrain_exp.py:154-164` 中 forward 调用缺少 `y_mark`：

```python
outputs = self.model(
    ...,
    x_component_hist=x_component_hist,
    mask_indices=mask_indices,
    # 缺少 y_mark=y_mark
)
```

而 `igt_model.py` 的 forward 签名包含 `y_mark=None`，但当前代码中 y_mark 被接收后 **没有用于任何计算**。

**修复：Pretrain 时注入 Calendar Token**

折中方案：pretrain 时将 y_mark 作为额外 token 加入 attention（因为 mask 场景需要时间上下文），finetune 时不传（恢复 A1 的 8-token 结构，避免 C12 类过拟合）。

```python
# igt_model.py forward
if mask_indices is not None and y_mark is not None:
    # Pretrain: 加入 calendar token（第 9 个）
    cal_feat = y_mark[:, :self.pred_len, :]  # (B, pred_len, time_feat_dim)
    cal_token = self.calendar_proj(cal_feat.mean(dim=1, keepdim=True))  # (B, 1, d_model)
    tokens = torch.cat([comp_tokens, weather_tokens, cal_token], dim=1)  # (B, 9, d_model)
else:
    # Finetune: 保持 A1 的 8-token 结构
    tokens = torch.cat([comp_tokens, weather_tokens], dim=1)  # (B, 8, d_model)
```

需要在 `__init__` 中增加：

```python
self.calendar_proj = nn.Sequential(
    nn.Linear(time_feat_dim, d_model),
    nn.GELU(),
    nn.Linear(d_model, d_model),
)
```

**改动范围**

| 文件 | 改动 |
|------|------|
| `igt_model.py` | 增加 mask_token 参数、calendar_proj；forward 中条件拼接 calendar token |
| `pretrain_exp.py` | forward 调用传 y_mark |

---

## 3. 整合改动清单

### Phase 1：代码修复（本地验证，不需要 GPU）

| 文件 | 改动内容 | 关联层级 | 行数估计 |
|------|---------|---------|---------|
| `data.py` | 删除 pretraining_mode 的 train+val 扩展 | L3 | ~6 行删除 |
| `igt_model.py` | 增加 mask_token、calendar_proj；条件拼接 calendar token | L5 | ~25 行 |
| `pretrain_exp.py` | 修复 best_val_net 保存；传 y_mark | L2+L5 | ~15 行 |
| `physformer_exp.py` | 增加 schedule_type: constant 分支 | L4 | ~15 行 |
| 新增 3 个 pretrain configs | λ_net ∈ {0.3, 1.0, 3.0} 的预训练配置 | L1 | ~10 行/文件 |
| `configs/physformer_igt_b1_finetune.yaml` | learning_rate: 1e-5, schedule_type: constant | L4 | ~3 行 |
| `configs/physformer_igt_b1_r1_*.yaml` | 同步更新 LR 和 schedule | L4 | ~3 行/文件 |

**验证方式**：
- `py_compile` 全部修改文件
- `--print-config` 确认新 key 解析正确
- CPU smoke test：mask token 梯度流通、calendar token 拼接维度正确、constant LR 不创建 scheduler

### Phase 2：λ_net Ablation 预训练（需要 GPU）

用修复后的代码（L2+L3+L5 修复）并行跑 3 组预训练：

| 实验 | λ_net | 其他条件 | 输出 |
|------|-------|---------|------|
| B1-pre-lam03 | 0.3 | train-only, mask token, calendar token | pretrained_checkpoint.pth |
| B1-pre-lam10 | 1.0 | 同上 | pretrained_checkpoint.pth |
| B1-pre-lam30 | 3.0 | 同上 | pretrained_checkpoint.pth |

每组配置：

- **数据**：train-only（L3 修复）
- **目标**：comp_loss + λ_net × net_loss
- **Mask**：Learnable mask token（L5 修复）
- **Calendar**：pretrain 时传 y_mark 作为第 9 个 token（L5 修复）
- **Checkpoint**：total-loss-best 为主，额外保存 best_val_net（L2 修复）
- **其他**：batch_size=256, 50 epochs, patience=12

**预训练评估指标**：

每组预训练完成后，直接测试（不 finetune）：

| 指标 | 含义 |
|------|------|
| Test MAE | 聚合精度——能否接近/超越 A1 的 0.001811 |
| Component MAE (load, pv, wind, batt) | 分量精度——MCP 的物理可解释性是否保留 |
| Val Net Loss | 聚合泛化能力 |
| Val Total Loss | 多任务泛化能力 |

**λ_net 选择标准**：

```
优先选: Test MAE 最低的 λ
约束:  Component MAE 不能全部退化到 A1 水平以下（否则 MCP 的物理意义消失）
如果:   所有 λ 的 Test MAE 都 > 0.001830 → L1 假设错误，问题在 L2-L5
```

### Phase 3：Finetune 验证（需要 GPU）

用 Phase 2 选出的最优 λ_net 的预训练 checkpoint 做 finetune：

- **LR**：constant 1e-5（L4 修复）
- **Epochs**：10, patience=3
- **架构**：恢复 A1 的 8-token（不传 y_mark）
- **Loss**：纯 net MSE（与 A1 完全一致）
- **Seeds**：3 seeds (2025, 2026, 2027)
- **Checkpoint**：对比 total-loss-best vs net-loss-best 两个预训练 checkpoint

---

## 4. 成功标准与证伪协议

### Phase 2 成功标准（预训练直接测试）

```
改动: L2+L3+L5 修复 + λ_net ablation
验证: 预训练直接测试 MAE < A1 mean 0.001811（不 finetune）
阈值:
  - PASS:    ≥1 个 λ 的预训练直接 MAE < 0.001811
  - MARGINAL: 最优 λ 的预训练直接 MAE < 0.001830
  - FAIL:    所有 λ 的预训练直接 MAE > 0.001830
```

### Phase 3 成功标准（finetune 后测试）

```
改动: 最优 λ_net 的预训练 + constant LR finetune
验证: B1 Test MAE < A1 mean 0.001811，至少 2/3 seeds
阈值:
  - PASS:    ≥2/3 seeds MAE < 0.001811
  - MARGINAL: 所有 seeds MAE < 0.001830
  - FAIL:    最佳 seed MAE > 0.001830
```

### 证伪协议

```
Claim: 调整 λ_net 可以让 MCP 预训练学到更好的抵消结构，
       使 pretrain+finetune 超越 A1 from-scratch

证伪条件:
  Phase 2: 所有 λ ∈ {0.3, 1.0, 3.0} 的预训练直接 MAE > 0.001830
  Phase 3: 最优 λ 的 finetune 3 seeds 全部 MAE > 0.001811

通过条件:
  Phase 2: ≥1 个 λ 的预训练直接 MAE < 0.001811
  Phase 3: ≥2/3 seeds MAE < 0.001811

基线:
  A1 s2025: MAE = 0.001807
  A1 s2026: MAE = 0.001811
  A1 s2027: 待确认
  旧 B1 预训练直接测试: MAE = 0.001820 (λ=0.3, train+val, 无 mask token)

混杂因素:
  1. λ_net 和 L2-L5 修复同时生效 → 无法单独归因
     缓解: Phase 2 先看预训练直接 MAE，排除 finetune 混淆
  2. mask_token 初始化可能影响结果 → 多 seed 验证
  3. calendar token 可能引入 C12 类过拟合 → finetune 不用 calendar 可排除
  4. constant LR=1e-5 可能太小或太大 → 通过 Val MSE 曲线判断
  5. 预训练数据量减少 12.5%（train-only）→ 对比旧 pretrain 可量化
```

### 决策树

```
Phase 2: λ_net ablation 预训练直接测试
├── PASS (≥1 λ 的 MAE < 0.001811)
│   ├── 选最优 λ → Phase 3 finetune
│   └── 如果 3.0 最优: 预训练已接近纯 net，finetune 改善空间可能有限
├── MARGINAL (最优 λ MAE < 0.001830)
│   ├── 方向正确 → Phase 3 finetune 看能否突破
│   └── 如果 finetune 仍不过 gate → 尝试 λ 网格 {5.0, 10.0}
└── FAIL (所有 λ MAE > 0.001830)
    ├── L1 假设错误：λ_net 不是瓶颈
    ├── 排查: 是 L2-L5 修复不足，还是 pretrain+finetune 范式本身无效？
    └── 对比: 旧 pretrain (λ=0.3, train+val) vs 新 pretrain (λ=0.3, train-only)
         如果新 < 旧 → L3/L5 有效，继续调参
         如果新 ≈ 旧 → L3/L5 无影响，问题在 L4（finetune schedule）

Phase 3: 最优 λ finetune
├── PASS (≥2/3 seeds MAE < 0.001811)
│   ├── Phase B rescue 成功
│   ├── 记录最优 λ_net 值和 pretrain+finetune 范式
│   └── 进入 B2（进一步 ablation）
├── MARGINAL (所有 seeds MAE < 0.001830)
│   ├── 尝试: LR 网格 {5e-6, 2e-5}、finetune epochs {15, 20}
│   ├── 尝试: 用 net-loss-best checkpoint 替代 total-loss-best
│   └── 如果仍不过 → 转入 FAIL
└── FAIL (最佳 seed MAE > 0.001830)
    ├── pretrain+finetune 范式在此数据集上无效
    ├── 记录为 dead end
    └── 关闭 Phase B，论文围绕 A1 + 消极先验结果（C12）撰写
```

---

## 5. 与已知 Dead Ends 的关系

| Dead End | 与本方案的关系 |
|----------|--------------|
| N06 (sigmoid gate) | 无关——本方案不涉及残差连接 |
| N07 (component loss 过强) | **部分相关**——N07 证明过强的 component loss 导致崩溃。本方案的 λ_net ablation 是在找 component loss 的最优权重，而非去掉它 |
| N12 (Phase 3 无效) | 间接相关——本方案的 finetune 不是 Phase 3（不从 curriculum 预训练 checkpoint finetune），而是从 MCP 预训练 checkpoint finetune |
| N22 (V5.4 gradient vanishing) | 无关——本方案不涉及 R8+B1 的 curriculum 配置 |
| N113 (pure net_mse finetune degrades) | **核心参照**——N113 的失败是本方案的出发点。本方案通过 L1(λ调优)+L2(checkpoint修复)+L4(schedule修复) 联合解决 N113 的失败根因 |

---

## 6. 回滚计划

如果五层修复后 B1 仍然 FAIL：

1. **代码回滚**：所有改动在 `codex/thesis-mainline` 分支，可 revert 到 pre-fix commit
2. **结论记录**：pretrain+finetune 范式在此数据集上无效，记录为 dead end
3. **论文方向**：围绕 A1（component-token separation, C11）+ 消极先验结果（C12）撰写
4. **不回滚的部分**：L2（checkpoint 选择修复）和 L3（scaler 清理）是代码质量改进，即使 B1 FAIL 也应保留
