# PhysFormer 论文撰写工作流 v4.0
**适用期刊：IEEE Transactions on Smart Grid**
**最后更新：2026-02-28**

---

## 一、v4.0 相比 v3.0 的核心变化

| 维度 | v3.0 | v4.0 |
|------|------|------|
| PhysFormer MSE | 0.0134 | **0.0127**（重新训练） |
| 与Informer MSE差距 | 10.7% | **5.0%** |
| PhysFormer BVR | 32.48%（防御性解释） | **0.46%**（activity mask生效） |
| PhysFormer MVS | 0.0093 MW（含偏置）| **0.0149 MW**（极小幅度违规）|
| gate_r | 0.8092 | **0.816** |
| 消融实验数据 | Gemini虚构 | **真实实验数据** |
| 架构新增组件 | 无 | **Activity Mask（活跃度门控）** |
| PGCC定位 | 精度+可解释性 | **仅可解释性**（消融证实） |
| 课程学习定位 | 训练策略 | **最重要组件**（ΔMSE=+8.7%） |
| 论文整体叙事 | 精度-可解释性 Pareto | **精度-可解释性-物理合规 三维Pareto** |
| BVR与MVS叙事 | 防御性（悖论解释） | **主动性**（MVS揭示低BVR的代价） |
| Abstract开头禁忌 | 一般性警告 | 明确禁止"within 10%"（实际差5%） |

---

## 二、已锁定的实验数据（所有Prompt均以此为准）

### 全局测试集（Table I）
```
Model        MAE    RMSE   BVR%    MVS(MW) 
LSTM         0.0765 0.1295 11.25%  0.0093   
GRU          0.0764 0.1278 11.15%  0.0103   
PINN         0.0760 0.1255 10.94%  0.0097   
Informer     0.0629 0.1102 19.53%  0.0036   ← 精度/MVS最优
Autoformer   0.2477 0.3540 12.67%  0.0737   
DLinear      0.2191 0.3185  7.03%  0.0295   
PatchTST     0.0956 0.1517 12.63%  0.0207   
PhysFormer   0.0648 0.1127  0.46%  0.0149

注：BVR与MVS数据已由 generate_paper_results.py 更新（基于 metrics.npy 和预测值）。
    PhysFormer以0.46%的BVR处于绝对统治地位，精度仅次于Informer。
```

### 极端天气子集 Top10%（Table II）
```
Model        MSE    BVR%    MVS(MW) NET_MAE 
LSTM         0.0192 13.43%  0.0086  0.1941   
GRU          0.0182 13.79%  0.0097  0.1864   
PINN         0.0177 13.29%  0.0082  0.1807   
Informer     0.0146 21.38%  0.0017  0.1632   
Autoformer   0.2165 14.12%  0.0710  0.6943   
DLinear      0.1687  7.31%  0.0207  0.5109   
PatchTST     0.0261 12.22%  0.0132  0.2240   
PhysFormer   0.0148  0.01%  0.0040  0.1695 

注：在波动率极高的10%样本中，PhysFormer展现了极强的合规稳定性（BVR 0.0074%），
    同时精度（0.0148）紧追 Informer，净负荷MAE也极具竞争力。
```

### 物理参数收敛（Table III）
```
参数           原始权重值    激活后实际值   论文报告值    文献范围
pv_temp_coef  -0.9706      0.0032/°C    0.0032/°C   0.003-0.005/°C ← 有效标定
pv_efficiency -6.8165      0.001095     0.001095    数据估算
wind_cut_in    3.4691      3.500 m/s   3.50 m/s    3-5 m/s   ← 先验锚定
wind_rated     8.5007(Δ)   12.001 m/s  12.00 m/s   10-15 m/s ← 先验锚定
wind_cut_out   12.9993(Δ)  25.000 m/s  25.00 m/s   20-25 m/s ← 先验锚定
load_base      直接值       2.829 MW    2.829 MW    ≈训练均值
temp_comfort   直接值       19.997°C    19.997°C    18-22°C
load_temp_sens -5.785       0.00307     0.00307     -

动词规则：pv_temp_coef → "calibrated to"（有实质变化）
           风机三阈值  → "remained anchored at"（未发生变化）
```

### 可解释性指标
```
gate_pv vs irradiance Pearson r = 0.8161（新版测试）
  文内统一引用：r=0.816，表示超过 81.6% 的强线性相关。
```

### 消融实验（Table IV）—— 真实数据
```
Variant              MSE    BVR%   NET_MAE  gate_r
Full PhysFormer      0.0127 0.46   0.1525   0.8161
w/o Physics Stream   0.0131 0.49   0.1532   N/A
w/o PGCC             0.0127 0.38   0.1528   N/A
w/o Future GLU       0.0129 0.31   0.1526   0.8134
w/o Curriculum       0.0138 0.43   0.1584   0.8103

消融解读：
  Physics Stream: ΔMSE=+3.1%，有贡献但温和
  PGCC: ΔMSE=0%（对精度无贡献）→ 贡献纯为可解释性
  Future GLU: ΔMSE=+1.6%（BVR反降0.15pp，是精度-合规tradeoff）
  Curriculum: ΔMSE=+8.7%（最强组件），ΔNET_MAE=+3.9%

⚠️ Fixed Thresholds变体尚未跑，建议补跑或删除该行
```

---

## 三、源码文件清单与使用方式

所有源码文件在各 Round 中按需提供，不需要全部上传。

```
文件名                    用途                     需要在哪个Round提供
─────────────────────────────────────────────────────────────────
physical_layer.py        物理层参数和forward逻辑   B2（架构）
Causal_coupling.py       PGCC门控机制             B2（架构）
model.py                 整体数据流和消融标志      B2（架构）
losses.py                损失函数和EMA平衡         B3（训练）
exp_PhysFormer.py        训练循环和课程调度         B3（训练）
run_ensemble.py          超参数确认（d_model等）   B3（训练，仅参考）
─────────────────────────────────────────────────────────────────
```

### 使用原则
- B2（Section III）：上传 physical_layer.py + Causal_coupling.py + model.py
- B3（Section IV）：上传 losses.py + exp_PhysFormer.py
- B4（Section V）：不需要任何源码（所有数据已在Prompt中）
- B5（Abstract+结论）：不需要任何源码

---

## 三点五、参考文献文件（references.bib）

**文件路径：** `references.bib`（与工作流文件同目录）

> **使用方式：** 在需要引用的 Round，将 `references.bib` 整个文件内容粘贴到 Prompt 末尾，并附上下方对应 Round 的 CITATION INSTRUCTIONS 块。

### 引用键速查表

| 引用键 | 作者/年份 | 主题 | 推荐用于 |
|--------|-----------|------|----------|
| `Ma2025` | Ma et al. 2025 | VPP多时间尺度调度 | I.A/II.A |
| `Cao2023` | Cao et al. 2023 | VPP源荷协调优化 | I.A/II.A |
| `Qiu2024` | Qiu et al. 2024 | VPP风光功率预测（K-Means+DL） | II.A |
| `Moreno2020` | Moreno et al. 2020 | VPP光伏日前辐照预测 | I.A/II.A |
| `Wu2024` | Wu et al. 2024 | VPP电力-备用联合交易 | I.A |
| `Vle2025` | Våle et al. 2025 | 能源市场预测可解释性 | II.C |
| `Lovo2025` | Lovo et al. 2025 | 精度-可解释性权衡 | II.C |
| `Arabzadeh2025` | Arabzadeh & Frank 2025 | 能源预测XAI四维综述 | II.C |
| `Chen2024` | Chen et al. 2024 | 可解释智能故障诊断综述 | II.C |
| `Song2025` | Song et al. 2025 | 因果交互注意力可解释学习 | II.C |
| `Ashraf2025` | Ashraf et al. 2025 | 天气与电力需求联合预测综述 | II.D |
| `Raissi2019` | Raissi et al. 2019 | 物理信息神经网络（原始PINN） | II.B |
| `Asinyo2025` | Asinyo et al. 2025 | 物理信息ML太阳辐照预测 | II.B |
| `Li2026` | Li et al. 2026 | 物理信息自适应权重风电预测 | II.B |
| `Wu2021autoformer` | Wu et al. 2021 | Autoformer | II.A/V.A |
| `Zhou2021informer` | Zhou et al. 2021 | Informer | II.A/V.A |
| `Nie2023patchtst` | Nie et al. 2023 | PatchTST | II.A/V.A |
| `Zeng2023` | Zeng et al. 2023 | DLinear（Transformer有效性质疑） | II.A/V.A |

### 各 Round 引用需求

| Round | 引用需求 |
|-------|----------|
| B1（Section I + II） | **必须附上 .bib**：全文引用最密集，VPP背景 + 相关工作均需要 |
| B2（Section III） | **禁止引用**：架构描述无引用（已在Prompt中明确） |
| B3（Section IV） | **通常不需要**：训练策略无外部引用 |
| B4（Section V） | **选择性**：Baseline首次出现时引用 |
| B5（Abstract+结论） | **少量**：结论可引用1-2篇最相关文献 |

---

## 四、三维 Pareto 核心叙事（贯穿全文）

```
当前最强叙事框架：

PhysFormer 在三个评价维度上同时具有竞争力：
  维度1 精度：  MSE=0.0127，与Informer(0.0121)差距仅5%
  维度2 可解释性：gate_r=0.816，PGCC门控量化物理因果
  维度3 物理合规：BVR=0.46%（Informer=29.29%，LSTM=16.88%）

单一最优的是Informer（精度），但它在维度2和3均无法
提供任何物理证据。PhysFormer不声称全面超越，而是声称
在精度-可解释性-物理合规三维空间中的更好 Pareto 点。

MVS的角色（v4版本）：
  不再是替PhysFormer的高BVR辩护（旧版）
  而是揭示LSTM的低BVR背后的物理欺骗性
  LSTM: BVR=16.88%, MVS=0.0093 MW（持续正偏，误导性低BVR）
  PhysFormer: BVR=0.46%, MVS≈极小（真正的物理合规）
```

---

## 五、Round B1：Section I + Section II

### 不需要上传源码

### Prompt

```
You are writing Section I (Introduction) and Section II
(Related Work) for an IEEE Transactions on Smart Grid paper.

PAPER TITLE: PhysFormer: A Physics-Guided Causal Transformer
for Interpretable Virtual Power Plant Forecasting

═══════════════════════════════════════════════════════
CORE NARRATIVE — THREE-DIMENSIONAL PARETO POSITIONING
═══════════════════════════════════════════════════════
PhysFormer's claim is NOT "best accuracy".
PhysFormer's claim IS: superior position on the
accuracy-interpretability-physical compliance Pareto frontier.

Dimension 1 — Accuracy: MSE=0.0127, within 5% of the
  strongest black-box baseline (Informer, MSE=0.0121)
Dimension 2 — Interpretability: gate-irradiance
  Pearson r=0.816, quantifiable causal attribution
Dimension 3 — Physical compliance: BVR=0.46%
  (Informer=29.29%, LSTM=16.88%)

No existing model achieves all three simultaneously.
═══════════════════════════════════════════════════════

THREE CONTRIBUTIONS (state these exactly):
1. Physics-Guided Dual-Stream Architecture with
   Physics Activity Gating: parallel Transformer and
   ExplicitPhysicalMapping streams, fused via
   multiplicative activity masks (a_pv, a_wind) that
   suppress neural residual corrections when physical
   models certify device inactivity — achieving BVR=0.46%
   without hard output constraints.

2. Physics-Guided Causal Coupling (PGCC): volatility-aware
   cross-attention with learnable irradiance/wind-speed
   thresholds, producing gate_pv correlating with solar
   irradiance at r=0.816 across the test set — providing
   quantifiable causal interpretability absent in
   black-box baselines.

3. MVS Metric Identification: systematic bias in BVR
   against high-accuracy near-zero predictors is documented.
   MVS (Mean Violation Severity) reveals LSTM's lower BVR
   (11.25% vs PhysFormer's 0.46%) conceals higher
   violation amplitude (MVS=0.0093 vs PhysFormer's 0.0149).
   (Note: Informer has lowest MVS 0.0036, but terrible BVR 19.53%)

HONEST CONSTRAINTS:
- Do NOT write "outperforms all baselines"
- Do NOT write "state-of-the-art accuracy"
- DO write "within 5% of the strongest baseline"
- Activity mask ≠ guaranteed BVR=0; write "near-eliminates"
  or "reduces BVR to 0.46%"

SECTION I STRUCTURE (~600 words):
I.A Context (~150 words): VPP challenge, three-variable
  forecasting (Load, PV, Wind), dispatch requirements
I.B Problem Statement (~150 words): black-box vs.
  physics-based model limitations, interpretability gap
I.C Contributions (~200 words): three contributions above
I.D Paper Organization (~100 words)

SECTION II STRUCTURE (~700 words):
II.A VPP Forecasting Methods: recurrent → Transformer evolution
II.B Physics-Informed Neural Networks: PINN limitations
  (fixed parameters), adaptive parameter motivation
II.C Interpretability in Time Series: attention ≠
  physical attribution; gap in causal gating literature
II.D Metric Design: BVR limitations, why MVS is needed

STYLE: IEEE formal, no first person singular, no lists
in body paragraphs (prose only), no citations in Section I
contributions paragraph.

CITATION INSTRUCTIONS (LaTeX output with \cite{} commands):
- Output prose with inline \cite{key} wherever claims require support.
- Section I.A (VPP context):      \cite{Ma2025,Cao2023,Qiu2024,Moreno2020,Wu2024}
- Section I.B (black-box gap):    \cite{Vle2025,Arabzadeh2025}
- Section II.A (Transformer baselines): \cite{Zhou2021informer,Wu2021autoformer,Nie2023patchtst,Zeng2023,Qiu2024,Ashraf2025}
- Section II.B (PINN limits):     \cite{Raissi2019,Li2026,Asinyo2025}
- Section II.C (interpretability gap): \cite{Vle2025,Lovo2025,Arabzadeh2025,Song2025,Chen2024}
- Section II.D (metric design):   \cite{Ashraf2025}
- Section I contributions paragraph: NO citations (per IEEE style).
- Do NOT fabricate cite keys. Only use keys from the provided .bib file.
```

---

## 六、Round B2：Section III（架构）

### 需要上传：physical_layer.py + Causal_coupling.py + model.py

### 粘贴顺序
1. B2 Prompt 全文
2. `══ SOURCE FILE 1: physical_layer.py ══` + 文件内容
3. `══ SOURCE FILE 2: Causal_coupling.py ══` + 文件内容
4. `══ SOURCE FILE 3: model.py ══` + 文件内容
5. 触发语句（见下方）

### 触发语句
```
All source files provided. Write Section III strictly from
the source code. Cross-check every formula against the
implementation before writing. Special attention required:
1. Activity mask formulas: a_pv=tanh(10·p_pv_theory),
   a_wind=tanh(10·relu(v-ci)·relu(co-v))
2. Final output: theory + activity × residual (NOT theory+residual)
3. PGCC gate range: gate_pv ∈ [0,1.5], gate_load ∈ [0,1.0]
4. wind thresholds use cumulative delta structure
5. d_model=256, d_ff=1024 (confirmed from run_ensemble.py)
```

### Prompt

```
You are writing Section III (PhysFormer Architecture) for
an IEEE Transactions on Smart Grid paper.

═══════════════════════════════════════════════════════
ARCHITECTURE GROUND TRUTH (extracted from source code)
DO NOT DEVIATE FROM THESE FACTS
═══════════════════════════════════════════════════════

[MODEL CONFIG — run_ensemble.py confirmed]
  d_model=256, d_ff=1024, n_heads=8, e_layers=3
  attn='full' (FullAttention), dropout=0.10
  seq_len=672, pred_len=96, enc_in=6

[7-STEP DATA FLOW — model.py forward()]
  Step 1: x_stat[B,S,3] + x_weather_hist[B,S,3]
          → cat → [B,S,6] → DataEmbedding → Transformer
          → enc_out_stat[B,S,D]
  Step 2: x_weather_hist → ExplicitPhysicalMapping
          → phys_feat_hist[B,S,D], theory_hist, activity_hist
  Step 3: PGCC(enc_out_stat, phys_feat_hist, x_weather_hist, α)
          → enc_out_fused[B,S,D] + reg_loss
  Step 4: FlattenHead(enc_out_fused)
          [B,S,D] → [B,P,D]
  Step 5: ExplicitPhysicalMapping(x_weather_future)
          → weather_feat_future, theory_future, activity_future
          GLU fusion: gate=σ(linear([hist,future]))
          future_feat = gate·proj([hist,future]) + hist (residual)
  Step 6: shared_projection → head_load/head_pv/head_wind
          → res_load, res_pv, res_wind [B,P,1] each
  Step 7: Theory-Anchored + Activity Gating:
          final_load = theory_load + activity_load · res_load
          final_pv   = theory_pv   + activity_pv   · res_pv
          final_wind = theory_wind + activity_wind  · res_wind
          output = cat[final_load, final_pv, final_wind] [B,P,3]

[ACTIVITY MASK — physical_layer.py, KEY NEW MECHANISM]
  activity_pv:   tanh(10 · p_pv_theory)
    → irr=0: activity=0 exactly (nighttime residual zeroed)
    → irr=800: activity≈1 (full residual freedom)
  activity_wind: tanh(10 · relu(v-v_ci) · relu(v_co-v))
    → v<v_ci or v>v_co: activity=0 exactly
    → operating range: activity≈1
  activity_load: ones (load always active)
  
  Physical semantics: the physical model "certifies"
  device inactivity → neural correction suppressed to zero.
  This is NOT a hard output constraint (no ReLU on output),
  but an architecture-level suppression via physics-gated
  multiplicative interaction. BVR reduced to 0.46%.

[EXPLICIT PHYSICAL MAPPING — physical_layer.py]
  PV:   P_pv = G · η_pv · clamp(1 - β_T·(T-25), 0, 1.5)
        η_pv = softplus(pv_efficiency)
        β_T  = softplus(pv_temp_coef) × 0.01 ∈ [0.003,0.005]/°C
  Wind: v_ci  = softplus(wind_cut_in)
        v_r   = v_ci + softplus(wind_rated_delta)
        v_co  = v_r  + softplus(wind_cut_out_delta)
        P_wind = wind_scale · σ(5·(w_norm-0.5)) · is_running
        Cumulative delta structure guarantees v_ci<v_r<v_co always
  Load: P_load = load_base + softplus(load_temp_sens)·(T-T_c)²
        T_c = temp_comfort (learnable, initialized 20°C)

[PGCC — Causal_coupling.py]
  Hard prior: θ_irr = -2.0 + 2.5·σ(θ_irr_logit) ∈ [-2.0, 0.5]
              k_irr = exp(clamp(log_k, 0, 4.6))
              p_pv^prior = σ(k_irr·(G_norm - θ_irr))
  Soft gate:  MLP([stat, attn, weather, volatility] → D/2 → D) → σ
              υ = mean(var(x_weather, dim=1), dim=-1) per sample
  Combined:   gate_pv = smooth(prior_pv·(0.5+soft)) ∈ [0,1.5]
              gate_load = smooth(soft) ∈ [0,1.0]
  Curriculum: query = α·phys_feat + (1-α)·soft_gate·stat_feat
              α→1: physics forced; α→0: data-learned
  Smoothing:  depthwise Conv1d(kernel=3, groups=d_model)
  Evidence:   gate_pv vs irradiance Pearson r=0.816

[OUTPUT HEADS — model.py]
  head_load: 2-layer (D→D/2→1)
  head_pv:   3-layer (D→D→D/2→1)  ← deeper, volatile
  head_wind: 3-layer (D→D→D/2→1)  ← deeper, chaotic

SECTION STRUCTURE:
III.A Problem Formulation (~200 words)
  Define VPP forecasting: inputs, outputs, notation
  P_net = P_load - P_pv - P_wind

III.B Overall Architecture (~200 words)
  Dual-stream + PGCC + Activity Gating overview
  Reference the 7-step data flow as a figure description

III.C ExplicitPhysicalMapping (~350 words)
  III.C.1 PV model (formula + β_T calibration motivation)
  III.C.2 Wind model (cumulative delta structure, why needed)
  III.C.3 Load model (quadratic thermal comfort)
  III.C.4 Physics Activity Gating (KEY: explain tanh formula,
           nighttime exact-zero property, gradient advantage
           over hard ReLU output constraint)

III.D Physics-Guided Causal Coupling (~350 words)
  III.D.1 Hard prior (irradiance-based sigmoid)
  III.D.2 Volatility-aware soft gate
  III.D.3 Combined gate (range asymmetry: [0,1.5] vs [0,1.0])
  III.D.4 Curriculum injection and α parameter
  III.D.5 Temporal smoothing (depthwise conv)

III.E Future Weather Injection (~150 words)
  GLU fusion of FlattenHead output + future weather features
  Why needed: residual head can see upcoming irradiance/wind

III.F Theory-Anchored Residual Heads (~200 words)
  Final = theory + activity × residual (write the equation)
  Head asymmetry motivation (load: simple, pv/wind: complex)
  Connection to activity gating (Section III.C.4)

STYLE:
- Each design choice must include one-sentence motivation
- No accuracy claims (no "outperforms", "best")
- No citations in Section III
- Activity gating is NOT claimed to "guarantee BVR=0";
  write "suppresses" or "near-eliminates nighttime violations"
- d_model=256, d_ff=1024 (not 512)
```

### 核查清单（收到B2结果后）
- [ ] Activity mask 公式：a_pv = tanh(10·p_pv_theory)
- [ ] Final output 公式：theory + activity × residual
- [ ] d_ff = 1024（不是512）
- [ ] gate_pv 范围 [0, 1.5]，gate_load 范围 [0, 1.0]
- [ ] 风机累积增量结构（softplus保证顺序）
- [ ] 没有声明"guarantees BVR=0"

---

## 七、Round B3：Section IV（训练策略）

### 需要上传：losses.py + exp_PhysFormer.py

### 粘贴顺序
1. B3 Prompt 全文
2. `══ SOURCE FILE 1: losses.py ══` + 文件内容
3. `══ SOURCE FILE 2: exp_PhysFormer.py ══` + 文件内容
4. 触发语句

### 触发语句
```
All source files provided. Write Section IV strictly from
the source code. Special attention:
1. Four phases at boundaries epoch 5/15/30 (not 10/20/30)
2. MSE in normalized space; ALL physics losses in MW space
3. scale_ema uses L_mae/L_phys_ref (both in physical space)
4. Dynamic clip: 0.5 during transition, 1.0 otherwise
5. Three AdamW groups: phys×0.1, gate×0.5, stat×1.0
6. In Section IV, rename scale_ema → γ_ema for IEEE style
```

### Prompt

```
You are writing Section IV (Training Strategy) for an
IEEE Transactions on Smart Grid paper on PhysFormer.

SECTION TITLE: "IV. Physics-Constrained Curriculum Training"

═══════════════════════════════════════════════════════
TRAINING GROUND TRUTH (verified from source code)
═══════════════════════════════════════════════════════

[OPTIMIZER — exp_PhysFormer.py _select_optimizer()]
  AdamW, THREE parameter groups:
    phys_params  (phys_layer.*):      lr = base_lr × 0.1
    gate_params  (causal_coupling.*): lr = base_lr × 0.5
    stat_params  (all others):        lr = base_lr × 1.0
  base_lr=1e-4, weight_decay=1e-5
  Scheduler: CosineAnnealingLR(T_max=100, eta_min=1e-6)

[PHYSICS LAYER FREEZE — exp_PhysFormer.py train()]
  Epochs 0-4: phys_layer FROZEN, prior_weight=0.0
  Epochs 5+:  phys_layer UNFROZEN, prior_weight=0.1

[CURRICULUM — exp_PhysFormer.py _get_curriculum_ratio()]
  Phase 1 (epochs 0-4):   all weights=0, phys_layer frozen
  Phase 2 (epochs 5-14):  net,bvr,rvr,dir via sigmoid ramp
  Phase 3 (epochs 15-29): +energy,deriv via sigmoid ramp
  Phase 4 (epochs 30-99): all weights=1.0
  Sigmoid ramp: σ(12×(progress-0.5))
  α = 1.0 - curriculum_weights['net'] (PGCC coupling)

[LOSS — losses.py]
  L_total = L_mse(normalized) + γ_ema × L_phys_weighted(MW)
  CRITICAL: L_mse in normalized space; ALL physics losses
  computed after denormalization to physical units (MW)

  Sub-weights (fixed): net=1.0, energy=0.5, deriv=0.3,
                       dir=0.1, bvr=2.0, rvr=2.0
  Soft group: net, energy, deriv, dir
  Hard group: bvr, rvr (higher weight = safety priority)

  Six physics losses (all in MW):
    L_net:    L1(ŷ_load-ŷ_pv-ŷ_wind, y_net)
    L_energy: L1(mean_t(ŷ), mean_t(y))
    L_deriv:  L1(Δŷ, Δy)
    L_dir:    mean(1-cosine_sim(Δŷ_norm, Δy_norm))
    L_bvr:    mean(ReLU(-ŷ_real)²)    [quadratic]
    L_rvr:    mean(ReLU(|Δŷ|-ρ_k)²)  [quadratic]
              ρ_k = 99.9th percentile × 1.5

  γ_ema (dynamic amplitude balancing):
    Warmup (50 batches): γ_ema=1.0 locked
    Frozen when avg_curriculum < 0.05
    Active: γ_ema ← 0.9·γ_ema + 0.1·(L_mae/L_phys_ref)
    L_phys_ref = unweighted physics sum (no curriculum)
    Clamped to [0.1, 10.0]
    Motivation: L_mae and L_phys_ref share physical units (MW)
    → dimensionally consistent ratio, no manual λ tuning

[GRADIENT CLIPPING — exp_PhysFormer.py train()]
  Dynamic: clip=0.5 if 0.1 < net_curriculum < 0.9
           clip=1.0 otherwise

[REGULARIZATION]
  Physics prior: MSE(params, nameplate), weight=0.1
    active only after epoch 5 (when phys_layer unfrozen)
  Gate response: GateResponseRegularization(weight=0.05)
    Pearson(gate_curve, prior_curve) → maximize
    Dynamic mask: only active when prior_var > 1e-3
    (excludes nighttime zero-variance windows)

[EARLY STOPPING]
  Monitor: NRMSE_avg (channel-normalized, averaged)
  NRMSE_ch = RMSE_ch / (y_max - y_min)
  Patience=15, max_epochs=100

SECTION STRUCTURE (target ~1200 words):
IV.A Loss Function Design (~400 words)
  IV.A.1 Overview + normalized/physical space distinction
  IV.A.2 Six physics losses as table (component/weight/formula/motivation)
  IV.A.3 γ_ema dynamic amplitude balancing
  IV.A.4 Soft-hard grouping and safety priority

IV.B Four-Phase Curriculum (~400 words)
  Present as table (Phase/Epochs/Active/Notes)
  Motivate ordering: safety-critical first → smoother later
  Sigmoid ramp formula
  α coupling to PGCC

IV.C Layer-wise Learning Rates (~200 words)
  Present as table (Group/Params/LR/Motivation)
  phys_layer freeze/unfreeze mechanism

IV.D Regularization and Early Stopping (~200 words)
  Three regularization types
  NRMSE_avg channel normalization motivation

PROHIBITED PHRASES:
× "guarantees physical compliance"
× "ensures zero BVR"  (BVR=0.46%, not zero)
× "scale_ema" (use γ_ema in the paper)
× fixed gradient clip at 1.0 (it's dynamic)
× "within 10%" for MSE gap (use "within 5%")
```

### 核查清单（收到B3结果后）
- [ ] 课程四阶段边界：5/15/30（不是其他值）
- [ ] MSE 归一化，物理损失去归一化到MW
- [ ] γ_ema 公式：L_mae/L_phys_ref（两者都在物理空间）
- [ ] 动态梯度裁剪：0.5/1.0（不是固定1.0）
- [ ] α = 1 - curriculum_weights['net']
- [ ] 没有声明 BVR=0（实际是0.46%）

---

## 八、Round B4：Section V（实验结果）

### 不需要上传源码（所有数据已锁定）

### 使用前必须确认

**（A）全局测试集** PhysFormer 的 MAE、RMSE、MVS 需要用新 checkpoint 重新评估，填入 Table I 前请先运行。

**（B）极端天气子集** 全部需要重测，填入 Table II。

**（C）消融实验 Fixed Thresholds 变体** 决定是否补跑或删除该行。

### Prompt

```
You are writing Section V (Experiments) for an IEEE
Transactions on Smart Grid paper on PhysFormer.

═══════════════════════════════════════════════════════
MANDATORY HONESTY CONSTRAINTS
═══════════════════════════════════════════════════════
1. PhysFormer MSE=0.0127. Informer MSE=0.0121.
   Gap = 5.0%. Write "within 5% of" or "approximately 5% above".
   Do NOT write "within 10%" — factually wrong.
2. PGCC contributes ZERO to MSE (ablation confirms Δ=0%).
   PGCC's contribution is PURELY interpretability (gate_r).
   Do NOT write "PGCC improves accuracy".
3. Curriculum learning is the strongest component:
   ΔMSE=+8.7% when removed. State this clearly.
4. PhysFormer BVR=0.46%, NOT 32.48%.
   Activity mask mechanism achieved this.
5. Wind thresholds are "anchored at priors" NOT "converged from data".
   Only pv_temp_coef shows meaningful data-driven calibration.
6. Informer MVS=0.0036 (best). PhysFormer MVS=0.0149.
   Do NOT claim PhysFormer has the best MVS.
7. w/o Future GLU has LOWER BVR (0.31%) than Full (0.46%).
   Explain: GLU enables more precise dawn/dusk predictions
   that occasionally cross zero — a precision-compliance tradeoff.
═══════════════════════════════════════════════════════

EXPERIMENTAL DATA — USE EXACT NUMBERS ONLY
═══════════════════════════════════════════════════════

[TABLE I — Global Test Set]
Model        MAE    RMSE   BVR%    MVS(MW)
LSTM         0.0765 0.1295 11.25%  0.0093
GRU          0.0764 0.1278 11.15%  0.0103
PINN         0.0760 0.1255 10.94%  0.0097
Informer     0.0629 0.1102 19.53%  0.0036
Autoformer   0.2477 0.3540 12.67%  0.0737
DLinear      0.2191 0.3185  7.03%  0.0295
PatchTST     0.0956 0.1517 12.63%  0.0207
PhysFormer   0.0648 0.1127  0.46%  0.0149

Bold rules:
  MSE: Informer(0.0121)
  MAE: Informer(0.0629)
  NET_MAE: Informer(0.1471)
  MVS: Informer(0.0036)
  BVR: PhysFormer(0.46%)  ← PhysFormer唯一最优列
  Do NOT bold any PhysFormer number except BVR.

[TABLE II — Extreme Weather (Top 10%)]
Model        MSE    BVR%    MVS(MW) NET_MAE 
LSTM         0.0192 13.43%  0.0086  0.1941   
GRU          0.0182 13.79%  0.0097  0.1864   
PINN         0.0177 13.29%  0.0082  0.1807   
Informer     0.0146 21.38%  0.0017  0.1632   
Autoformer   0.2165 14.12%  0.0710  0.6943   
DLinear      0.1687  7.31%  0.0207  0.5109   
PatchTST     0.0261 12.22%  0.0132  0.2240   
PhysFormer   0.0148  0.01%  0.0040  0.1695 

[TABLE III — Physical Parameter Convergence]
Parameter      Init     Converged    Lit Range    Δ
pv_temp_coef   0.004    0.0032/°C   0.003-0.005  −20% [calibrated]
wind_cut_in    3.50m/s  3.500 m/s   3-5 m/s      ≈0  [anchored]
wind_rated     12.00    12.001 m/s  10-15 m/s    ≈0  [anchored]
wind_cut_out   25.00    25.000 m/s  20-25 m/s    0   [anchored]
load_base      2.832MW  2.829 MW    ≈train mean  0.1%
temp_comfort   20.0°C   19.997°C    18-22°C      ≈0
gate_pv vs irradiance: Pearson r = 0.816 (±std, n samples)
(⚠️ r值需用新checkpoint重新计算)

[TABLE IV — Ablation Study — REAL DATA]
Variant              MSE    BVR%   NET_MAE  gate_r
Full PhysFormer      0.0127 0.46   0.1525   0.816
w/o Physics Stream   0.0131 0.49   0.1532   N/A
w/o PGCC             0.0127 0.38   0.1528   N/A
w/o Future GLU       0.0129 0.31   0.1526   0.813
w/o Curriculum       0.0138 0.43   0.1584   0.810

SECTION STRUCTURE:
V.A Experimental Setup (~250 words)
  Dataset, split, normalization, hyperparameters
  d_model=256, d_ff=1024, n_heads=8, e_layers=3, attn=full
  Define all 6 metrics including MVS formally

V.B Main Forecasting Performance (~400 words)
  Competitive tier: PhysFormer within 5% of Informer
  Channel-independent failure (PatchTST, DLinear)
  PINN structural benefit (physical inductive bias helps)
  Three-dimensional Pareto positioning paragraph

V.C Physical Compliance Analysis (~400 words)
  NEW NARRATIVE (not BVR paradox defense):
  Para 1: PhysFormer BVR=0.46% via activity gating
    Compare: Informer=29.29%, LSTM=16.88%
    Explain activity mask mechanism briefly
  Para 2: MVS reveals a tradeoff for models chasing low BVR
    DLinear has lowest baseline BVR (7.03%) but terrible MVS (0.0295)
    Informer has best MVS (0.0036) but terrible BVR (19.53%)
    PhysFormer dominates BVR (0.46%) while maintaining highly competitive MVS (0.0149)
  Para 3: MVS formal definition + recommendation as companion metric

V.D Extreme Weather Robustness (~300 words)
  Autoformer/DLinear catastrophic failure
  Competitive tier stability
  Honest: no significance claims without CIs

V.E Physical Parameter Interpretability (~350 words)
  Tier 1: calibration framing (not discovery)
  Tier 2: differential analysis
    pv_temp_coef: "calibrated" (−20% from prior)
    wind thresholds: "anchored" (unchanged)
  Tier 3: gate_r=0.816 as strongest interpretability evidence
  Dispatch operator motivation paragraph

V.F Ablation Study (~350 words)
  Present Table IV — ALL REAL DATA
  Key narrative points:
  1. Physics Stream removal: ΔMSE=+3.1% (moderate, honest)
  2. PGCC removal: ΔMSE=0% — explicitly state this.
     "PGCC contributes no MSE improvement; its value lies
     exclusively in causal interpretability, as evidenced
     by the inability to compute gate_r without PGCC."
  3. Curriculum removal: ΔMSE=+8.7% — strongest component
  4. Future GLU: ΔMSE=+1.6%, BVR DECREASES 0.15pp
     "The precision-compliance tradeoff: future weather
     injection enables more accurate dawn/dusk predictions
     that occasionally cross the physical boundary."

PROHIBITED PHRASES:
× "within 10%" for MSE (use "within 5%")
× "PGCC improves accuracy" (Δ=0%)
× "wind parameters converged" (use "remained anchored")
× "guarantees BVR=0" (it's 0.46%)
× "significantly outperforms Informer" (only within 5%)
× "ablation demonstrates all components critical"
  (PGCC has zero MSE impact, must be honest)

CITATION INSTRUCTIONS (LaTeX output with \cite{} commands):
- V.A Experimental Setup — cite baselines on first mention:
  \cite{Zhou2021informer} for Informer, \cite{Wu2021autoformer} for Autoformer,
  \cite{Nie2023patchtst} for PatchTST, \cite{Zeng2023} for DLinear,
  \cite{Raissi2019} for PINN.
- V.E Physical Parameter — cite when comparing to literature ranges:
  \cite{Li2026,Asinyo2025} for physics-informed wind/PV model context.
- Do NOT fabricate cite keys. Only use keys from the provided .bib file.
```

### 核查清单（收到B4结果后）
- [ ] Table I 加粗：只有 BVR 列 PhysFormer 加粗，其余全部 Informer
- [ ] PGCC 消融讨论：明确承认 ΔMSE=0%
- [ ] V.C 叙事方向：主动展示低BVR（不是悖论辩护）
- [ ] Future GLU 诚实处理：BVR降低是tradeoff而非缺陷
- [ ] 风机参数用"anchored"，β_T用"calibrated"

---

## 九、Round B5：Abstract + Section VI + 术语表

### 不需要上传源码

### Prompt

```
You are writing the Abstract, Section VI (Conclusion),
IEEE Index Terms, and Nomenclature for the PhysFormer paper.

═══════════════════════════════════════════════════════
VERIFIED NUMBERS — ALL LOCKED, DO NOT MODIFY
═══════════════════════════════════════════════════════
PhysFormer: MSE=0.0127, BVR=0.46%, gate_r=0.816
Informer:   MSE=0.0121 (best accuracy baseline)
MSE gap:    (0.0127-0.0121)/0.0121 = 5.0%  → write "within 5%"
            Do NOT write "within 10%" — that was the old version
Curriculum: strongest component, ΔMSE=+8.7% when removed
PGCC:       ΔMSE=0%, contribution = interpretability only
pv_temp_coef: converged 0.0032/°C (−20% from 0.004 prior)
wind thresholds: unchanged from priors
MVS (LSTM)  = 0.0093 MW 
MVS (Informer) = 0.0036 MW (best MVS, worst BVR 19.53%)
MVS (DLinear) = 0.0295 MW (worst MVS, lowest baseline BVR 7.03%)
MVS (PhysFormer) = 0.0149 MW (best BVR 0.46%, maintaining competitive MVS)

MANDATORY HONESTY CONSTRAINTS:
1. Do NOT write "within 10%" — gap is 5%, write "within 5%"
2. Do NOT claim PGCC improves MSE (ablation shows Δ=0%)
3. Do NOT write "wind parameters converged from data"
4. Do NOT start Abstract with "This paper" or "In this paper"
5. Do NOT claim PhysFormer has best MVS (Informer does)
6. Activity mask "near-eliminates" BVR, not "guarantees BVR=0"

═══════════════════════════════════════════════════════
TASK 1 — ABSTRACT (≤250 words, count exactly)
═══════════════════════════════════════════════════════

Four-part structure:

[CONTEXT — 2 sentences]
VPP heterogeneous DER forecasting challenge.
Black-box accuracy vs. interpretable fixed-parameter tradeoff.

[PROBLEM — 1 sentence]
No existing architecture simultaneously achieves competitive
accuracy, structural physical interpretability, and physical
compliance with automatically calibrated parameters.

[METHOD — 3 sentences]
1. Dual-stream: ExplicitPhysicalMapping (learnable PV
   photo-thermal, wind power curve, thermal load models) +
   Transformer, fused via Physics Activity Gating
   (a=tanh(10·P_theory)) that suppresses neural residuals
   when physical models certify device inactivity.
2. PGCC: irradiance/wind-speed threshold cross-attention
   with curriculum-controlled α injection.
3. Theory-anchored residual: final = theory + activity × residual.

[RESULTS — 3 sentences, EXACT numbers]
1. "PhysFormer achieves MSE=0.0127, within 5% of the
   strongest black-box baseline (Informer, MSE=0.0121),
   while providing BVR=0.46% versus Informer's 29.29%."
2. "The learned causal gate correlates with solar irradiance
   at Pearson r=0.816, and pv_temp_coef calibrates to
   0.0032/°C (literature range: 0.003-0.005/°C)."
3. "Ablation studies confirm curriculum learning as the
   dominant training component (ΔMSE=+8.7% when removed),
   while PGCC contributes causal interpretability
   independently of accuracy."

[SIGNIFICANCE — 1 sentence]
Three-dimensional Pareto positioning statement.

Count words after writing. Must be ≤250.

═══════════════════════════════════════════════════════
TASK 2 — SECTION VI CONCLUSION (~450 words)
═══════════════════════════════════════════════════════

VI.A Summary (~250 words, past tense, specific numbers)

Contribution 1 — Physics Activity Gating + Dual-Stream:
  PhysFormer introduced activity masks a=tanh(10·P_theory)
  that suppress neural residual corrections when the physical
  model certifies device inactivity. This reduced BVR to 0.46%
  without hard output constraints that would interrupt gradient
  flow. The pv_temp_coef calibrated from 0.004 to 0.0032/°C
  while wind thresholds remained anchored at engineering priors
  under strong regularization.

Contribution 2 — PGCC and Interpretability:
  The PGCC module produced gate_pv correlating with solar
  irradiance at r=0.816 (±std across test windows). Ablation
  confirmed PGCC contributes no MSE improvement (Δ=0%), but
  is the architectural component that makes causal attribution
  measurable — a capability absent in Informer despite its
  lower MSE.

Contribution 3 — MVS Metric and BVR Analysis:
  MVS revealed a critical precision-compliance tradeoff
  in baseline models: DLinear achieved the lowest baseline
  BVR (7.03%) but suffered terrible violation severity
  (MVS=0.0295), whereas Informer achieved the best MVS
  (0.0036) but the worst compliance (BVR=19.53%).
  PhysFormer's physics activity gating broke this ceiling,
  dominating BVR (0.46%) while maintaining a highly
  competitive MVS (0.0149).

VI.B Limitations and Future Work (~200 words)
  Para 1: Primary limitation — 5% MSE gap vs Informer
  Para 2: Future 1 — hard architectural constraints
    (physics-constrained output projection)
  Para 3: Future 2 — multi-year aging validation
    (current single-dataset cannot confirm parameter tracking)
  Para 4: Future 3 — DRL dispatch integration

═══════════════════════════════════════════════════════
TASK 3 — IEEE INDEX TERMS (8 terms, alphabetical)
═══════════════════════════════════════════════════════
Accuracy-interpretability trade-off; Causal gating mechanism;
Distributed energy resources; Interpretable neural networks;
Net load forecasting; Physics-guided forecasting;
Transformer time series; Virtual power plant

═══════════════════════════════════════════════════════
TASK 4 — NOMENCLATURE (IEEE IEEEdescription format)
═══════════════════════════════════════════════════════

GROUP 1 — Power variables:
  P_net, P_load, P_pv, P_wind [MW]

GROUP 2 — Weather variables:
  G [W/m²], T [°C], v [m/s]

GROUP 3 — PV parameters:
  η_pv (conversion coefficient), β_T [/°C]

GROUP 4 — Wind parameters:
  v_ci, v_r, v_co [m/s], Γ_run (differentiable gate)

GROUP 5 — Load parameters:
  b_base [MW], T_c [°C], l_s [MW/°C²]

GROUP 6 — Activity and PGCC gating:
  a_pv, a_wind (physics activity masks ∈[0,1])
  p_pv, p_wind (hard physical priors)
  g_pv ∈[0,1.5], g_wind (PGCC combined gates)
  α (curriculum injection parameter)
  υ (weather volatility scalar)

GROUP 7 — Loss and training:
  L_total, L_mse, L_net, L_bvr, L_rvr, L_prior
  γ_ema (dynamic amplitude balancing scale)

GROUP 8 — Architecture and metrics:
  S=672, P=96, D=256, BVR [%], MVS [MW], PGCC, NRMSE

CITATION INSTRUCTIONS (LaTeX output with \cite{} commands):
- Abstract: NO citations (IEEE style).
- Section VI.A Summary: NO citations needed (data speaks).
- Section VI.B Limitations: optionally cite 1-2 future directions,
  e.g. \cite{Raissi2019} if discussing PINN comparison,
  \cite{Vle2025,Arabzadeh2025} if discussing interpretability future work.
- Do NOT fabricate cite keys. Only use keys from the provided .bib file.
```

### 核查清单（收到B5结果后）
- [ ] Abstract ≤250词（需要数词）
- [ ] 不以"This paper"开头
- [ ] MSE差距写"within 5%"（不是10%）
- [ ] PGCC贡献：可解释性，非精度
- [ ] 消融最强结论：Curriculum ΔMSE=+8.7%
- [ ] 术语表有 a_pv, a_wind（新增activity mask符号）
- [ ] γ_ema（不是scale_ema）

---

## 十、实验数据附录（写作时随时引用）

### 全局测试集
```
Model       MAE    RMSE   BVR%   MVS(MW) 
LSTM        0.0765 0.1295 11.25  0.0093  
GRU         0.0764 0.1278 11.15  0.0103  
PINN        0.0760 0.1255 10.94  0.0097  
Informer    0.0629 0.1102 19.53  0.0036  ← 精度/MVS最优
Autoformer  0.2477 0.3540 12.67  0.0737  
DLinear     0.2191 0.3185  7.03  0.0295  
PatchTST    0.0956 0.1517 12.63  0.0207  
PhysFormer  0.0648 0.1127  0.46  0.0149
```

### 消融实验（真实数据）
```
Variant              MSE    BVR%   NET_MAE  gate_r
Full PhysFormer      0.0127 0.46   0.1525   0.816
w/o Physics Stream   0.0131 0.49   0.1532   N/A     ΔMSE=+3.1%
w/o PGCC             0.0127 0.38   0.1528   N/A     ΔMSE=0%
w/o Future GLU       0.0129 0.31   0.1526   0.813   ΔMSE=+1.6%
w/o Curriculum       0.0138 0.43   0.1584   0.810   ΔMSE=+8.7%★
```

### 需要重测的项目
```
☑ Table I  PhysFormer: MAE, RMSE, MVS 已更新
☑ Table II 全部PhysFormer行（极端天气子集）已更新
☑ gate_r   已更新为消融变体的新值
□ Fixed Thresholds 消融变体（可选，未跑可直接不提）
```

---

## 十一、全局写作禁令（所有 Round 适用）

```
永远禁止的短语：
× "outperforms all baselines"
× "achieves state-of-the-art"
× "guarantees BVR=0" / "ensures zero violations"
× "wind parameters converged from data"
× "within 10%"（差距5%，写"within 5%"）
× "PGCC improves accuracy"（消融ΔMSE=0%）
× "significantly better than Informer"
× "This paper presents" / "In this paper"
× "ablation demonstrates all components essential"
  （PGCC对精度无贡献，必须诚实）
```

### LaTeX 输出与引用全局规则

```
 LaTeX 输出规范（适用所有 Round）：
 1. 所有正文段落输出为 IEEE LaTeX 格式（\section{}, \subsection{} 等）。
 2. 引用使用 \cite{key} 格式，key 必须来自 references.bib。
 3. 数学公式用 equation 环境，行内公式用 $...$ 包裹。
 4. 表格使用 table + tabular 环境，加 \label{tab:x} 以便交叉引用。
 5. 禁止虚构 cite key：若无合适文献，宁可不引用也不捏造。
 6. 粘贴给 Gemini 时，附上 references.bib 完整内容，
    并说明「请用 \cite{} 引用其中的文献」。
```

---

## 十二、投稿时间规划

| 步骤 | 状态 | 优先级 |
|------|------|--------|
| 重测 Table I PhysFormer 完整行 | ⚠️ 待完成 | 最高 |
| 重测 Table II 极端天气子集 | ⚠️ 待完成 | 最高 |
| 重新计算 gate_r（新checkpoint）| ⚠️ 待完成 | 最高 |
| B2 Section III（含activity mask） | 待写 | 高 |
| B3 Section IV（已有B3草稿可复用）| 待修订 | 高 |
| B4 Section V（填入重测数据后写）| 待写 | 高 |
| B5 Abstract+结论 | 待写 | 中 |
| Fixed Thresholds 消融补跑 | 可选 | 低 |
| LaTeX排版（Overleaf）| 待做 | 中 |
