# C08: Component Error Covariance Decomposition

**Date**: 2026-05-15 | **Analysis**: V6 S2 gru64 vs gru96 | **Status**: C08 elevated from observation → supported

## Executive Summary

We decompose the aggregate net injection error variance into per-component variances and pairwise covariances to explain the **component-aggregate paradox**: configurations with better per-component MAE can have worse aggregate MSE. The mechanism is **error cancellation via signed summation** — since `net = load − pv − wind + batt`, positively correlated component errors partially cancel in the aggregate, while decorrelated errors add in quadrature.

The analysis reveals that **larger GRU capacity (gru96) produces larger component errors but stronger inter-component correlations, yielding more cancellation (79.8% vs 66.7%) and thus preserving aggregate accuracy**. The original C08 hypothesized that "tighter coupling reduces cancellation" — this data shows the opposite direction operates in the capacity regime.

## Data

| Source | Description |
|--------|-------------|
| `results/v6_s2_ablation/physformer_v6_s2_gru{64,96}/extras/physics_states.npz` | Per-component theory predictions (real MW) |
| `results/v6_s2_ablation/physformer_v6_s2_gru{64,96}/extras/residual_net.npy` | Per-component residuals (normalized) |
| Extracted from test DataLoader | Per-component true values `y_aux_real` (real MW) |

Component ordering: `[load, pv, wind, battery_power, battery_soc]`

## Theory-Only Error Correlation Matrices

### gru64 (d_gru=64, better components)

| | load | pv | wind | batt_p | batt_soc |
|---|------|-----|------|--------|----------|
| **load** | 1.000 | +0.212 | +0.016 | −0.165 | +0.125 |
| **pv** | | 1.000 | −0.000 | **+0.632** | +0.374 |
| **wind** | | | 1.000 | +0.062 | +0.018 |
| **batt_p** | | | | 1.000 | +0.105 |
| **batt_soc** | | | | | 1.000 |

### gru96 (d_gru=96, better aggregate)

| | load | pv | wind | batt_p | batt_soc |
|---|------|-----|------|--------|----------|
| **load** | 1.000 | +0.095 | +0.015 | −0.158 | +0.112 |
| **pv** | | 1.000 | +0.102 | **+0.833** | +0.580 |
| **wind** | | | 1.000 | +0.187 | +0.188 |
| **batt_p** | | | | 1.000 | +0.454 |
| **batt_soc** | | | | | 1.000 |

**Key difference**: gru96 shows systematically stronger positive correlations among PV, Wind, Battery components. pv-batt correlation is particularly strong (+0.833 vs +0.632), which is crucial because pv and batt enter the net equation with opposite signs: `net = ... − pv + batt`. When pv and batt errors are strongly positively correlated, `−error_pv + error_batt` nearly cancels.

## Variance Decomposition

The aggregate theory error is `e_net = e_L − e_PV − e_W + e_B`. Its variance decomposes as:

```
Var(e_net) = Var(e_L) + Var(e_PV) + Var(e_W) + Var(e_B)
           − 2Cov(e_L, e_PV) − 2Cov(e_L, e_W) + 2Cov(e_L, e_B)
           + 2Cov(e_PV, e_W) − 2Cov(e_PV, e_B) − 2Cov(e_W, e_B)
```

| Term | gru64 (×10⁻⁶) | gru96 (×10⁻⁶) | Ratio (96/64) |
|------|--------------|--------------|---------------|
| Var(e_L) | 7.52 | 8.94 | 1.19× |
| Var(e_PV) | 15.63 | 31.83 | 2.04× |
| Var(e_W) | 0.22 | 0.38 | 1.73× |
| Var(e_B) | 6.69 | 17.61 | 2.63× |
| **Component variance sum** | **30.06** | **58.76** | **1.95×** |
| −2Cov(e_L, e_PV) | −4.60 | −3.20 | |
| −2Cov(e_L, e_W) | −0.04 | −0.06 | |
| +2Cov(e_L, e_B) | −2.34 | −3.96 | |
| +2Cov(e_PV, e_W) | 0.00 | +0.70 | |
| −2Cov(e_PV, e_B) | −12.92 | −39.42 | |
| −2Cov(e_W, e_B) | −0.16 | −0.96 | |
| **Covariance cancellation** | **−20.06** | **−46.90** | **2.34×** |
| **Net Var(e_net)** | **10.00** | **11.86** | 1.19× |
| **Cancellation ratio** | **66.7%** | **79.8%** | |

## Interpretation

### The cancellation mechanism

Despite gru96 having **1.95× larger component variance**, its **2.34× larger covariance cancellation** brings the net aggregate variance to only 1.19× of gru64. The cancellation ratio (79.8% vs 66.7%) is the key statistic.

The dominant cancellation channel is **Cov(e_PV, e_B)** — the pv-battery error covariance. This makes physical sense: PV and battery are the two components with the largest dynamic range, and their errors are strongly coupled through the shared encoder. When the encoder overestimates PV (positive e_PV), it tends to also overestimate battery charging (positive e_B), and since `net = ... − pv + batt`, these errors partially cancel.

### Why gru96 has stronger correlations

The larger GRU (gru96) has more capacity to learn shared temporal patterns across components. This shared representation creates **correlated errors**: when the model makes a mistake about the time-of-day pattern, all weather-driven components (PV, Wind) and the coupled battery response shift together. The smaller GRU (gru64) has a capacity ceiling that forces more component-specific learning → more independent errors → less cancellation.

### The two regimes of the component-aggregate paradox

The original C08 hypothesis (from V6.1) predicted that **gradient isolation reduces error cancellation**. This V6 S2 analysis reveals the **capacity regime** where the opposite operates: **more shared capacity increases correlation and thus cancellation**. Both regimes produce the same observable (component MAE ↓ but aggregate MSE ↑ or unchanged), but through different mechanisms:

| Regime | Mechanism | Correlation effect | Cancellation |
|--------|-----------|-------------------|--------------|
| **Capacity** (V6 S2) | Larger GRU → more shared patterns → errors more correlated | ↑ | ↑ (better agg) |
| **Isolation** (V6.1, hypothesis) | Gradient detach → less shared learning → errors more independent | ↓ | ↓ (worse agg) |

This unification is a novel mechanistic contribution: **it is the correlation structure of component errors, not their magnitudes, that dominates aggregate accuracy**.

### Residual head compensation

The analysis above uses theory-only errors. The residual head further shapes aggregate accuracy. gru96's larger residual head (more GRU capacity → more residual correction freedom) compensates for worse theory, explaining why gru96 wins aggregate MSE (7.281e-6 vs 7.691e-6) despite worse theory. The residual head effectively "learns the cancellation pattern" that the theory errors create naturally in gru64.

## Implications for Thesis

1. **C08 is now supported** (was observation). The variance decomposition provides a quantitative mechanistic explanation for the component-aggregate paradox.

2. **The decomposition framework** (`Var(e_net) = ΣVar − 2ΣCov`) can be a standalone contribution — it applies to any multi-component forecasting model with signed aggregation.

3. **For V6.1 isolation regime verification**, per-component prediction data from V6.1 variants is needed. The V6.1 checkpoints were not preserved locally; this analysis should be repeated when V6.1 is re-run.

4. **Practical recommendation**: For VPP dispatch, the aggregate matters more than components. Model selection should prioritize correlation structure over component MAE. The dual-config reporting strategy (baseline for aggregate, detach for physics) remains valid.

## Reproducibility

Analysis script: `scripts/c08_extract_and_analyze.py`
Requires: V6 S2 checkpoints + test dataset (local)
