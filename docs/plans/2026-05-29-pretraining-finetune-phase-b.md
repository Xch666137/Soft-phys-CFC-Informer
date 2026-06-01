# Phase B: Self-Supervised Pretraining for Cross-Portfolio Generalization

**Status**: design (pre-implementation)
**Date**: 2026-05-29
**Author**: Xingyu Liu (direction), Claude Opus 4.7 (design + documentation)
**Branch**: codex/thesis-mainline
**ARA binding**: N106 (exploration_tree.yaml)

---

## 1. Motivation

### 1.1 What we know from Phase A

The PhysFormer-iGT ablation chain (A1-A5) produced a clean, monotonic result:

```
A1 (8 tokens, simple FFN decoder):             MAE 0.001811  ← OPTIMAL
A2 (+1 physics token):                         MAE 0.001819  +0.4%
A3 (+5 twin + 3 constraint tokens, 16 total):  MAE 0.001843  +1.8%
A4 (+graph bias annealing):                    MAE 0.001863  +2.9%
A5 (+per-step CrossAttention decoder):         MAE 0.001947  +7.5%
```

**Core finding (C10 confirmed):** Component-token separation eliminates the shared-encoder cancellation channel. This is the **only** architectural innovation that improves generalization. All subsequent additions — physics tokens, graph bias, horizon decoder — are overfitting amplifiers: they improve Val MSE while degrading Test MAE.

However, A1 does NOT solve the cross-portfolio generalization problem — it merely avoids making it worse. The model learns portfolio-specific patterns from the training set, and there is no explicit mechanism forcing it to learn **universal** coupling laws that transfer across portfolios.

### 1.2 Why pretraining?

The core insight from A1-A5 is:

> Fixed priors (physics tokens, graph bias, weather conditioning) overfit because they assume the **same coupling form** across all portfolios. Self-supervised pretraining forces the model to learn **data-driven coupling** from diverse portfolio observations, without imposing a specific parametric form.

This is analogous to the NLP paradigm shift from feature engineering → pretraining:

| Era | NLP | VPP Forecasting |
|-----|-----|----------------|
| Hand-crafted | POS taggers, parse trees | Physics equations, graph topology |
| Pretraining | BERT (masked LM) | Masked Component Pretraining |
| Finetuning | Task-specific head tuning | Net MSE supervised training |

### 1.3 Why Masked Component Pretraining (MCP)?

In BERT, masking a word forces the model to use **surrounding context** to infer the missing word. This learns syntax and semantics that generalize across domains.

In our setting, masking a component's history (e.g., setting PV history to zeros) forces the model to use **other components + weather** to infer the masked component's future. This learns cross-component coupling laws that should generalize across portfolios:

- PV masked → model must use Irradiance weather token + Battery token (surplus → charging) to infer PV
- Load masked → model must use Temperature weather token + Calendar features to infer Load
- Battery masked → model must use PV token + Load token + SOC token to infer Battery behavior

Crucially, these learned relationships are **emergent from data**, not imposed by fixed equations or graph biases.

---

## 2. Architecture Overview

### 2.1 Model (A1, unchanged)

The A1 architecture is **not modified** for pretraining. The masking happens at the input level only.

```
Input:
  x_component_hist   (B, 672, 5)   [load, pv, wind, batt_p, batt_soc]
  x_weather_future   (B, 96, 3)    [temp, irrad, wind_speed]
  y_mark             (B, 96, C)    future calendar features

Architecture (unchanged from A1):
  Step 1: comp_tokens    = BatchedGRU(x_component_hist)        → (B, 5, d_model)
  Step 2: weather_tokens = BatchedMLP(x_weather_future)        → (B, 3, d_model)
  Step 3: tokens = cat([comp_tokens, weather_tokens])          → (B, 8, d_model)
  Step 4: tokens = InvertedEncoder(tokens)                     → (B, 8, d_model)
  Step 5: comp_preds = SharedFFN(tokens[:, :4, :])             → (B, 4, pred_len)
  Step 6: pred_net_real = load − pv − wind + batt (real MW)
  Step 7: pred_net = normalize(pred_net_real)
```

### 2.2 Masking Strategy

```
Input: x_component_hist (B, 672, 5)
         ch0: load    ────→  GRU  ────→  D_load    valid
         ch1: pv      ────→  GRU  ────→  D_pv      valid
         ch2: wind    ────→  GRU  ────→  D_wind  [MASKED]
         ch3: batt_p  ────→  GRU  ────→  D_batt_p  valid
         ch4: batt_soc ────→  GRU  ────→  D_soc     valid

Masking operation:
  For each masked component i:
    x_component_hist[:, :, i] = 0.0              # zero the entire 672-step history

  This causes:
    GRU(zero_input) → near-zero hidden state → near-zero token
    The token slot still exists in attention, but carries no component-specific info

  The model must reconstruct the masked component's future from:
    (a) Weather tokens (e.g., irradiance → PV)
    (b) Other component tokens (e.g., PV surplus → battery charging)
    (c) Calendar context (e.g., daytime → PV, peak hours → load)
```

**Masking parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Maskable components | {0,1,2,3} (load, pv, wind, batt_p) | SOC is a state variable, not a direct prediction target |
| Masks per sample | 1 (25%) or 2 (50%), uniform random | Analogous to BERT's 15% — enough to force learning, not too hard |
| Mask value | 0.0 (zero out history) | Physically meaningful: "component not observed" |
| Weather masking | NEVER | Weather is the context that enables cross-component inference |

**Why not a learnable [MASK] token?**

A learnable mask embedding (BERT-style) would need to encode "missing component" semantics, but components have different scales and distributions (load ~MW vs PV ~kW). A learnable vector at the GRU input level doesn't convey scale information. Zeros are cleaner and physically interpretable: "this component was not measured during this period."

**Why only 4 maskable components?**

SOC (channel 4) is a state variable without a direct prediction projector (component_projectors maps 4 → pred_len, not 5). Masking SOC would not have a direct prediction target, making the training signal weak. We keep SOC visible as **context** for other components (especially battery power prediction).

### 2.3 Pretraining Objective

```
For each sample with masked component set M ⊆ {0,1,2,3}:

  L_pretrain = (1/|M|) Σ_{i∈M} MAE(comp_preds[:, :, i], comp_true[:, :, i])
             + λ_net · MAE(pred_net, true_net)

  where:
    comp_preds[:, :, i] = component_projectors(tokens[:, i, :])  # normalized prediction
    comp_true[:, :, i]  = (x_component_future[:, :, i] - aux_mean[i]) / aux_std[i]
    λ_net = 0.3  (auxiliary net supervision weight)
```

**Why auxiliary net supervision?**

Pure component-level pretraining might learn representations that are good for component reconstruction but suboptimal for the downstream task (net prediction). A small net MSE anchor prevents the model from drifting into a representation space that's incompatible with the finetuning objective. λ_net = 0.3 is small enough that component MAE dominates, large enough to provide a meaningful anchor.

### 2.4 Finetuning Protocol

```
Phase 2 — Finetuning (identical to A1 training):

  1. Load pretrained weights (all layers)
  2. Train with net MSE only (no component loss, no pretraining loss)
  3. Same hyperparameters as A1 (lr=1e-4, OneCycleLR, batch_size=256, epochs=50)
  4. Early stopping: patience=12, start_epoch=12, monitor Val MSE(MW²)
  5. 3 seeds (2025, 2026, 2027)
```

No special finetuning tricks — the goal is to test whether pretrained representations transfer better than random initialization, without confounding from finetuning protocol differences.

---

## 3. Data Strategy

### 3.1 Pretraining Data

| Split | Used? | Rationale |
|-------|-------|-----------|
| Train | YES | Core pretraining data |
| Validation | YES | Additional unlabeled data for pretraining |
| Test | **NO** | Excluded for clean evaluation — ensures no data leakage |

Using train + val for pretraining is the conservative choice: all data the model sees during pretraining is from the same temporal distribution as finetuning data. The generalization challenge comes from **different portfolios** within this distribution, not from temporal distribution shift.

### 3.2 Data Loading

New flag in data provider: `pretraining_mode=True`

```
pretraining_mode=True:
  → Loads (train ∪ val) samples
  → No split stratification (samples from all portfolios mixed)
  → Shuffled at epoch boundary

pretraining_mode=False (normal):
  → Standard train/val/test split as before
```

### 3.3 Future Component Ground Truth

The current data pipeline provides `y_aux` = (B, 96, 5) with true future component values. For pretraining, we need these as prediction targets. The data pipeline already returns `y_aux` — it's just currently ignored in A1's net-MSE-only loss. No data pipeline changes needed; we simply route `y_aux` to the pretraining loss function.

---

## 4. Implementation Plan

### 4.1 File Changes

| File | Action | Lines (est.) |
|------|--------|-------------|
| `physformer/models/physformer/igt_model.py` | Add `forward(..., mask_indices=None)` masking logic + component prediction output | ~30 |
| `physformer/train/pretrain_exp.py` | **New**: Pretraining experiment class | ~120 |
| `physformer/loss.py` | Add `PretrainLoss` class (component MAE + net MSE anchor) | ~40 |
| `physformer/data/data_factory.py` | Support `pretraining_mode=True` flag | ~15 |
| `configs/physformer_igt_b1_pretrain.yaml` | **New**: Pretraining config | ~30 |
| `configs/physformer_igt_b1_finetune.yaml` | **New**: Finetuning config (3 seeds) | ~30 |
| `physformer/config.py` | Add `pretraining_mode` to data keys | ~5 |

### 4.2 Masking Logic (igt_model.py)

```python
def forward(self, ..., mask_indices=None):
    """
    mask_indices: list of component indices to mask, e.g. [1, 2] for PV+Wind.
    If None, normal A1 forward (no masking). Used during pretraining.
    """
    x_comp = x_component_hist.clone()  # (B, 672, 5)

    if mask_indices is not None:
        for idx in mask_indices:
            x_comp[:, :, idx] = 0.0

    comp_tokens = self.comp_embedding(x_comp)  # masked channels → near-zero tokens
    # ... rest of forward unchanged ...

    # Return component predictions for pretraining loss
    return {
        "pred_net": pred_net,
        "comp_preds_norm": comp_preds_norm,  # NEW: (B, 96, 4) for pretraining
        ...
    }
```

### 4.3 Pretraining Experiment (pretrain_exp.py)

```python
class PretrainExperiment(BaseExperiment):
    def __init__(self, args):
        # Reuse PhysFormerExperiment infrastructure
        # Key differences:
        # 1. PretrainLoss instead of PhysLoss
        # 2. pretraining_mode=True for data loading
        # 3. No curriculum, single phase
        # 4. Saves pretrained checkpoint for finetuning

    def _process_one_batch(self, batch_data):
        # 1. Sample mask indices (1-2 components randomly)
        n_mask = random.choice([1, 2])
        mask_indices = random.sample([0, 1, 2, 3], n_mask)

        # 2. Forward with masking
        outputs = self.model(..., mask_indices=mask_indices)

        # 3. Compute pretraining loss
        loss = self.criterion(outputs, y_aux, mask_indices)
        return loss
```

### 4.4 Pretraining Loss (loss.py)

```python
class PretrainLoss(nn.Module):
    def __init__(self, aux_mean, aux_std, target_mean, target_std, lambda_net=0.3):
        # Normalize component targets to z-score space
        self.aux_mean = aux_mean[:4]
        self.aux_std = aux_std[:4]
        self.lambda_net = lambda_net
        # ... net normalization buffers ...

    def forward(self, outputs, y_aux, mask_indices):
        # Component MAE on masked components only
        comp_preds = outputs["comp_preds_norm"]  # (B, 96, 4)
        comp_true = (y_aux[:, :, :4] - self.aux_mean) / self.aux_std  # normalize

        comp_loss = 0.0
        for idx in mask_indices:
            comp_loss += F.l1_loss(comp_preds[:, :, idx], comp_true[:, :, idx])
        comp_loss /= len(mask_indices)

        # Net MSE anchor
        net_loss = F.mse_loss(outputs["pred_net"], y_target_normalized)

        return comp_loss + self.lambda_net * net_loss
```

---

## 5. Experiment Plan

### 5.1 B1 Series

| ID | Configuration | Purpose | Seeds |
|----|--------------|---------|-------|
| **B1** | A1 + Masked Component Pretraining + Finetune | Test whether pretrained representations improve cross-portfolio generalization over from-scratch A1 | 3 (pretrain once, finetune ×3) |
| B1-aux | B1 with λ_net ∈ {0.1, 0.5} | Sensitivity to auxiliary net supervision weight | 1 (if B1 passes) |

**B1 workflow:**

```
Pretraining (1 run, no seed needed):
  Data: train ∪ val, batch_size=256
  Epochs: 50, early stopping on component MAE
  Output: pretrained_checkpoint.pth

Finetuning (3 seeds × 1 pretrained checkpoint):
  Data: train only (standard split)
  Config: identical to A1 (net MSE only, epochs=50, patience=12)
  Seeds: 2025, 2026, 2027
  Output: metrics.json per seed
```

### 5.2 Compute Budget

| Phase | Runs | Epochs/run | Time/epoch | Total |
|-------|------|------------|------------|-------|
| Pretrain | 1 | ~30-50 | ~200s | ~3h |
| Finetune | 3 | ~20-25 (early stop) | ~200s | ~4h (parallel) |
| **Total** | | | | **~7h GPU (vGPU-32GB)** |

### 5.3 Comparison Baselines

| Baseline | MAE (3-seed mean) | What it tests |
|----------|-------------------|---------------|
| A1 (from scratch) | 0.001811 ± 6e-6 | Lower bound: no pretraining |
| c23 (PhysFormer) | 0.002069 ± 1.34e-4 | Old architecture baseline |
| **B1 target** | **< 0.001811** | Pretraining hypothesis |

---

## 6. Falsification Protocol

### Claim: Masked Component Pretraining forces the model to learn portfolio-agnostic cross-component coupling, improving Test MAE over from-scratch A1

**Falsification condition:** B1 finetuning (3 seeds) Test MAE mean ≥ A1 mean (0.001811).

**Pass condition:** B1 Test MAE mean < A1 mean on ≥ 2 of 3 seeds, with σ overlap ≤ 1σ.

**Gate:**

| Outcome | Interpretation |
|---------|---------------|
| MAE < 0.001811 (≥2/3 seeds) | **PASS.** Pretraining learns transferable coupling, resolving the Phase A generalization ceiling. Proceed to B2 (ablation on masking strategies). |
| 0.001811 ≤ MAE ≤ 0.001820 | **MARGINAL.** Pretraining provides negligible benefit. Test B1-aux (λ_net sweep) before concluding. |
| MAE > 0.001820 | **FAIL.** Pretraining signal is either redundant (A1 already optimal) or harmful (another overfitting source). This still supports the A1 narrative: from-scratch training on component tokens is the ceiling. |

### Confound Control

1. **Pretraining epoch count:** If B1 fails, test whether the failure is due to under/over-pretraining. Sweep pretraining epochs ∈ {20, 50, 100}.
2. **Mask ratio:** If B1 fails, test masking 1 always vs. 2 always vs. 1-2 random.
3. **λ_net:** If B1 fails, sweep λ_net ∈ {0.1, 0.5, 1.0} to rule out auxiliary weight sensitivity.

---

## 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Pretrained weights diverge from finetuning objective | Low-Medium | Medium | λ_net anchor prevents full divergence; sweep if needed |
| Masking 1 component is too easy (model learns trivial identity) | Low | Low | Use 1-2 random masking; if too easy, increase to 2-3 |
| Pretraining on train+val overfits to training portfolios | Medium | Medium | Already the case for A1; pretraining should reduce this, not increase it |
| GRU processes zero input in unexpected ways | Low | Low | GRU(zero_seq) → near-zero hidden state is deterministic and well-behaved |
| Pretrained component MAE is low but finetuning doesn't transfer | Medium | High | This would mean pretraining learns a different task than what finetuning needs — a fundamental misalignment |

---

## 8. Key Design Decisions

| Decision | Rationale | Alternative Rejected |
|----------|-----------|---------------------|
| Zero-mask (not learnable [MASK] token) | Physically meaningful: "component not measured." Components have different scales; a learnable token can't encode per-component magnitude. | Learnable mask embedding (BERT-style) — rejected because component scales differ by orders of magnitude |
| Mask 1-2 of 4 components | 25-50% masking ratio, analogous to BERT's 15%. Enough to force learning, not so much that task becomes impossible. | Mask 1 always — too easy; Mask 3-4 — too hard, model may collapse |
| Component MAE in normalized (z-score) space | Consistent with A1 output format; avoids scale dominance by large-magnitude components (load >> wind) | Real MW MAE — would cause load loss to dominate |
| λ_net = 0.3 auxiliary anchor | Small enough that component MAE dominates, large enough to prevent representation drift | No anchor (λ=0) — risk of degenerate representations; λ=1.0 — pretraining ≈ finetuning, defeats the purpose |
| Full finetune (no frozen layers) | Simplest protocol; tests pure transfer quality without protocol confounds | Partial finetune (freeze GRU) — adds complexity; test only if full finetune passes |
| Pretrain on train ∪ val, exclude test | Conservative: no test data leakage; tests generalization from training-distribution pretraining | Include test — more data but confounds evaluation |

---

## 9. Relationship to Phase A

```
Phase A (PhysFormer-iGT Architecture):
  Claim:   Component-token separation (iTransformer) eliminates C10 cancellation channel.
  Result:  A1 MAE −12.5% vs c23. A2-A5 all degrade — architecture additions overfit.
  Ceiling: A1 MAE 0.001811. From-scratch training on component tokens is the optimum
           under the current supervised learning paradigm.

Phase B (Self-Supervised Pretraining):
  Claim:   Masked Component Pretraining learns portfolio-agnostic coupling laws
           that transfer better than from-scratch training.
  Key difference from A2-A5: No fixed priors. The coupling is learned from data,
           portfolio by portfolio, through the masking objective.
  Hypothesis: If A2-A5 failed because fixed priors don't transfer across portfolios,
           then data-driven priors (learned via masking) should transfer BETTER.
```

**If B1 passes:** The narrative shifts from "A1 is the ceiling" to "A1 + pretraining breaks the ceiling." The Phase A monotonic degradation chain becomes a story about **why fixed priors fail and learned priors succeed** — a stronger and more nuanced contribution.

**If B1 fails:** The narrative becomes "A1 is the true ceiling — even data-driven pretraining cannot improve upon from-scratch component-token separation." This is also a strong result: it says the problem is fundamentally limited by the information content in the data, not by the training paradigm. The iTransformer's component-token separation extracts all available signal; anything beyond that is overfitting, regardless of whether the priors are fixed or learned.

---

## 10. References

- C10: Shared encoder cancellation channel (`ara/logic/claims.md`)
- N100: A1 experiment — 8-token iTransformer, MAE 0.001811
- N102-N105: A2-A5 experiments — monotonic degradation chain
- N106: Phase B decision (`ara/trace/exploration_tree.yaml`)
- PatchFormer: Masked patch reconstruction for cross-domain TS transfer ([arXiv:2601.20845](https://arxiv.org/abs/2601.20845))
- TimeCAP: Channel-aware cross-domain pretraining for MTS forecasting (AAAI 2026)
- SEMPO: Mixture-of-Prompts for cross-domain TS adaptation (NeurIPS 2025)
- BERT: Masked language model pretraining (Devlin et al., 2019)
- iTransformer: Inverted Transformer for time series (Liu et al., 2023)
- Current A1 implementation: `physformer/models/physformer/igt_model.py`
- Phase A design doc: `docs/plans/2026-05-26-physformer-igt-architecture.md`
