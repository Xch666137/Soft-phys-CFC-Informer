# PhysFormer-iGT: Physics-Twin Graph iTransformer for DVPP Forecasting

**Status**: ~~design (pre-implementation)~~ **DEPRECATED (2026-05-29)** — superseded by [Phase B: Self-Supervised Pretraining + Finetune](2026-05-29-pretraining-finetune-phase-b.md)
**Date**: 2026-05-26 (design) / 2026-05-29 (deprecation)
**Author**: Xingyu Liu (architecture), Claude Opus 4.7 (documentation)
**Branch**: codex/thesis-mainline
**ARA binding**: N96 (exploration_tree.yaml)

> **DEPRECATION NOTICE (2026-05-29):**
>
> This document proposed a 4-axis PhysFormer-iGT architecture (twin tokens, graph bias, component residual, battery decoder) with an 8-step ablation chain (A0-A8). After empirical validation:
>
> - **A1 (8-token pure iTransformer)** was the ONLY success: MAE 0.001811, −12.5% vs c23.
> - **A2-A5 ALL FAILED** with monotonic degradation: A2 +0.4%, A3 +1.8%, A4 +2.9%, A5 +7.5%.
> - **A6-A8 were never executed** and are presumed DOA based on the monotonic trend.
>
> The valid content from this document is limited to:
> - **§1**: C10 root-cause diagnosis (correct — validated by A1)
> - **§1.3**: Known dead ends (historical guardrails, still valid)
> - **§3**: P0 real-unit power balance fix (correctness fix for old PhysFormer)
> - **P1-1**: A1 8-token iTransformer specification (the architecture we KEEP)
>
> **Everything else (§4-§9) is degraded thinking** — the "full iGT" target architecture (twin tokens, graph bias, horizon decoder, battery state-space, component losses, 68-token layout) was empirically falsified by A2-A5. The simplest possible interpretation of this document's core insight (component token separation) turned out to be optimal. All subsequent additions were overfitting amplifiers.
>
> **Replaced by:** `docs/plans/2026-05-29-pretraining-finetune-phase-b.md` — Self-Supervised Masked Component Pretraining + Finetune on the frozen A1 architecture. No architecture additions. Data-driven coupling discovery via masking.

---

## 1. Motivation

### 1.1 Root cause: shared encoder → cancellation channel (C10)

C10 established that the current PhysFormer architecture has a structural flaw:

```
Shared Encoder (d_model=256, e_layers=2)
  → encodes ALL 5 components + weather + battery into ONE representation space
  → FFN at each time step mixes d_model dimensions → implicit cross-component coupling
  → Residual head gradient flows back through encoder → induces correlated biases in theory branches
  → Biases cancel under net = load − pv − wind + batt (C08 capacity regime)
  → Components degrade, aggregate preserved (or improved)
```

`detach` (selective gradient cut) is a **gradient hack** that blocks the backflow channel. It works (C09), but it doesn't fix the architectural root cause: **one shared representation space serving two contradictory purposes** — learning useful cross-component interactions AND preventing correlated error induction.

### 1.2 Why iTransformer variant?

iTransformer inverts attention from the time dimension (O(seq_len²)) to the variate dimension (O(num_variates²)). For PhysFormer:

- Standard Transformer: attention over 672 time steps → O(451k) per layer
- iTransformer: attention over ~50 component×patch tokens → O(2.5k) per layer

But the deeper motivation is **architectural component separation**:
- Each component gets its own token(s) → no implicit feature mixing through FFN
- Cross-component interaction is explicit (attention weights), limited, and auditable
- Physics enters as peer tokens, not as post-hoc FiLM conditioning

**Caveat on attention interpretability**: Attention weights indicate which token pairs the model attends to, but they are *structured interaction candidates*, not causal explanations. Interpretability claims must be validated through edge masking (zero out PV→Battery attention weights and measure component MAE degradation), token intervention (clamp/perturb one token and observe prediction changes), and counterfactual weather perturbation (increase irradiance → verify PV_pred responds, battery charging tendency increases). Attention maps alone are insufficient for strong interpretability claims.

### 1.3 Known dead ends (DO NOT REPEAT)

*Source: ARA historical experiment log (`ara/trace/exploration_tree.yaml`). These are prior-journey constraints, not design decisions from the current architecture discussion. They are included here as implementation guardrails.*

| Node | Lesson |
|------|--------|
| N06 | Sigmoid gate on residual → theory degradation (identity shortcuts are superior) |
| N07 | Normalized component loss w/o gate → aggregate degraded (+3.6% MAE). **Relevant to component supervision design** |
| N12 | Phase 3 (pure net_mse fine-tune) → no validation benefit |
| N22 | V5.4 gradient vanishing (R8+B1) → physics branches need adequate gradient |
| N84 | O37 bias-clearance mechanism refuted across seeds → mechanism claims need 3-seed corroboration |
| N92 | M1 cov-cross direct reduction FALSIFIED → detach does NOT reduce |cov(theory, residual)| |
| N93 | M2 residual fraction FALSIFIED → residual fraction ~98.75% invariant across arms |

The N07 lesson is especially relevant: **strong component supervision has historically hurt aggregate accuracy in the shared-encoder architecture**. Whether architectural component separation (iTransformer tokens) eliminates this trade-off is an open empirical question and should be tested as an ablation axis (A4 vs A6).

---

## 2. Current Architecture Baseline (PhysFormer c23)

```
Input:
  x_net_hist       (B, 672, 1)    historical net power
  x_weather_hist   (B, 672, 3)    historical weather
  x_battery_hist   (B, 672, 2)    historical battery state
  x_weather_future (B, 96, 3)     future weather (known)
  x_mark_enc       (B, 672, C)    historical calendar features
  y_mark           (B, 96, C)     future calendar features

Data flow:
  Step 1: DataEmbedding(concat(hist))  → Encoder(seq-level self-attn, 2 layers)
  Step 2: TemporalDecoder(seq→pred, cross-attn with y_mark)
  Step 3: WeatherFusion(cross-attn with future weather)
  Step 4: PhysicalLayer(external, parallel)
            → component_theory_real (B, 96, 5) [load, pv, wind, batt_p, batt_soc]
            → theory_net_real = load - pv - wind + batt
            → battery_feats_real (B, 96, 4) [soc_norm, headroom, eta_c, eta_d]
  Step 5: component_norm = per-component z-score(component_theory_real)  ← BUG: see §3
  Step 6: PhysicsFiLM(data_latent, component_norm + battery_feats) → conditioned
  Step 7: UnifiedResidualHead(conditioned, physics_features) → 5-dim residual (normalized)
  Step 8: pred_net = (load_norm+res) − (pv_norm+res) − (wind_norm+res) + (batt_norm+res)  ← BUG
  Step 9: Loss = MSE(pred_net, norm_target) + component supervision on theory + res_reg

Parameters: d_model=256, e_layers=2, n_heads=8, d_ff=512
```

**Critical observation**: The Transformer Encoder (Steps 1-3) never sees physics. Physics enters only at Step 6 as FiLM conditioning — it's a "bolt-on" rather than an integral part of the architecture.

---

## 3. P0: Real-Unit Power Balance (blocking correctness fix)

### 3.1 Problem

`_build_component_net()` in `model.py:174-185` performs power balance in per-component z-score space:

```python
component_norm = self._norm_aux(component_real)  # each column: (x - mean_i) / std_i
# ...
load_th_res = component_norm[..., 0:1] + component_residual[..., 0:1]  # in load-zscore
pv_th_res   = component_norm[..., 1:2] + component_residual[..., 1:2]  # in pv-zscore
# ...
return load_th_res - pv_th_res - wind_th_res + batt_th_res  # mixed z-score spaces!
```

Since `aux_mean[i]` and `aux_std[i]` differ across components:
- `(load - load_mean)/load_std - (pv - pv_mean)/pv_std` ≠ `(load - pv - target_mean)/target_std`
- The power balance is distorted by differing per-component normalization scales
- PV with small `aux_std` gets amplified weight in the net computation

**Impact**: Absolute metrics are distorted because the bug affects the forward/loss path (pred_net, loss landscape, gradient scale, residual head component weights, physics-vs-residual coupling). Rankings from pre-fix runs may be confounded — all baselines and ablations must be re-run after P0.

### 3.2 Fix

```python
# Step A: Denormalize component residual to real units
# Residual is zero-mean delta → scale by aux_std only, NO aux_mean offset.
component_residual_real = torch.stack([
    component_residual[..., i] * self.aux_std[i]
    for i in range(5)
], dim=-1)

# Equivalent (when component_residual is in per-component z-score space):
#   component_residual_real = component_residual * self.aux_std.view(1, 1, -1)

# Step B: Add to theory in real units
component_pred_real = component_theory_real + component_residual_real

# Step C: Power balance in real MW
pred_net_real = (
    component_pred_real[..., 0:1]   # load
    - component_pred_real[..., 1:2] # pv
    - component_pred_real[..., 2:3] # wind
    + component_pred_real[..., 3:4] # batt_p
)

# Step D: Normalize to target space for MSE loss
pred_net = self._norm_target(pred_net_real)
```

**Why NO aux_mean on residual**: The residual is a zero-centered delta learned by `UnifiedResidualHead` (weight init near zero, bias init zero). Adding `aux_mean` would shift the residual by the component's mean magnitude, introducing systematic bias. Only the full prediction (`component_norm + residual`) should receive the mean when denormalizing to real units.

**Code locations**: `model.py:174-185` (`_build_component_net`), `model.py:276-278` (normalization), `model.py:298-302` (forward path).

**Verification**: After fix, `pred_net_real` should equal `theory_net_real + residual_net_real` in MW units. The test metrics (MAE in MW) should shift — direction and magnitude unknown, but likely non-trivial if `aux_std` values differ substantially across components.

---

## 4. Target Architecture: PhysFormer-iGT

### 4.1 Naming

**PhysFormer-iGT**: Physics-Twin Graph iTransformer for DVPP Forecasting

Core innovations (4-axis):
1. **Component-Physics Twin Tokenization** — each physical component has a data token AND a physics token
2. **Graph-constrained Cross-variate Attention** — DVPP physical topology as attention bias prior
3. **Identifiable Component Residual Learning** — component-level supervision with real-unit power balance
4. **State-space Battery Policy Decoder** — SOC recurrence, action residual, physical constraints

### 4.2 Architecture Diagram

```
Inputs:
  x_component_hist   (B, 672, 5)   [load, pv, wind, batt_p, batt_soc] history
  x_weather_hist     (B, 672, 3)   [temp, irrad, wind_speed] history
  x_weather_future   (B, 96, 3)    [temp, irrad, wind_speed] future
  x_battery_hist     (B, 672, 2)   [batt_power, batt_soc] history (for physical layer)
  x_mark_enc         (B, 672, C)   historical calendar features
  y_mark             (B, 96, C)    future calendar features
  portfolio_ids      (B,)          optional portfolio identifiers
  asset_fingerprint  (B, D)        optional, derived from history statistics

═══════════════════════════════════════════════════════════════
STEP 1: Physical Layer (unchanged from current, load branch fix)
═══════════════════════════════════════════════════════════════

  PhysicalLayer(x_weather_hist, x_weather_future, y_mark,
                x_load_hist, x_battery_hist, portfolio_ids)
    → component_theory_real  (B, 96, 5)   [load, pv, wind, batt_p, batt_soc]
    → battery_feats_real     (B, 96, 4)   [soc_norm, headroom, eta_c, eta_d]
    → theory_net_real        (B, 96, 1)   load - pv - wind + batt
    → constraint_states

  Load branch fix: use x_load_hist instead of x_net_hist for
  autoreg and GRU temporal correction. Net history used as
  optional global context only.

═══════════════════════════════════════════════════════════════
STEP 2: Tokenization
═══════════════════════════════════════════════════════════════

  A. Component Daily Patch Tokens (5 components × 7 days = 35 tokens)
     For each component c ∈ {load, pv, wind, batt_p, soc}:
       For each day d ∈ {1..7} (each day = 96 time steps):
         D_{c,d} = PatchEmbed(x_component_hist[c, (d-1)*96 : d*96])
     PatchEmbed options: Linear, small TCN, or GRU

  B. Physics Twin Tokens (5 tokens)
     Each component gets its own physics twin token. Net-level balance
     information is carried by C_balance (below), NOT by individual Φ_c.
     This keeps component physics "pure" — no global net coupling in Φ_c.

       Φ_Load  = PhysicsEmbed(load_theory, temp_future, calendar_future)
       Φ_PV    = PhysicsEmbed(pv_theory, irradiance_future, temp_future, solar_time)
       Φ_Wind  = PhysicsEmbed(wind_theory, wind_speed_future)
       Φ_BattP = PhysicsEmbed(batt_power_theory, soc_theory, headroom, eta)
       Φ_SOC   = PhysicsEmbed(soc_theory, capacity, headroom)

  C. Constraint Tokens (3 tokens)
     C_balance    = BalanceEmbed(theory_net_real)   # net-level power balance
     C_soc        = SOCEmbed(battery_feats[:, 0:1], battery_soc_theory)
     C_capacity   = CapacityEmbed(battery_feats[:, 1:2])  # headroom

     Design rationale: Φ_c carries component-level physics (e.g., PV theory
     conditioned on irradiance). C_balance carries the AGGREGATE power-balance
     constraint. This prevents each Φ_c from "seeing" theory_net_real (which
     mixes all components) and re-introducing net-level coupling into
     individual component tokens.

  D. Future Weather Tokens
     **Default path: W2 weather patch tokens + per-step raw weather in horizon decoder.**

     Weather tokens participate in graph attention (capturing global weather
     shape over the prediction horizon). Additionally, the horizon decoder
     receives the raw future weather at each step t (capturing precise temporal
     alignment). This dual-path design preserves both coarse weather context
     and fine-grained step-level conditioning.

     Options (documented for ablation):
       W1 (24 hourly): 3 variables × 24 hourly tokens = 72 tokens
       W2 (8 patch):   3 variables × 8 3-hourly patches = 24 tokens  ← default
       W3 (decoder-only): weather NOT tokenized; injected per-step in decoder only

  E. Asset Fingerprint Token (1 token, optional)
     Derived from history statistics:
       load_scale, pv_capacity_est, wind_capacity_est,
       batt_p_max_est, batt_e_max_est, soc_current,
       pv_daytime_ratio, batt_charge_discharge_ratio
     → MLP → C_asset token

  F. Time Context Tokens (optional, 2-4 tokens)
     Day-type token, season token, etc.

     **Critical: Canberra local time / solar time**
     The CSV `date` column is UTC, but the VPP portfolio is in Canberra
     (Australia/Sydney, UTC+10 or UTC+11 with DST). PV and irradiance
     diurnal cycles must use local time, not UTC hour.
     Requirements:
       - Convert UTC timestamps to Australia/Sydney local time.
       - Use local_hour_sin/cos, local_day_of_year_sin/cos as calendar features.
       - PV, Irradiance, and Battery tokens must receive solar-time features.
       - Do NOT use raw UTC hour as the only diurnal feature.
       - Optional: include solar elevation angle as an additional PV token feature.
     This affects `x_mark_enc`, `y_mark`, and the PhysicalLayer's PV branch
     (currently `pv_temp_factor` uses `temp` which is weather, not time — but
     the irradiance→PV mapping's diurnal alignment depends on correct local time).

  Total token count (W2 option): 35 + 5 + 3 + 24 + 1 = ~68 tokens

═══════════════════════════════════════════════════════════════
STEP 3: Graph-biased Cross-variate Attention
═══════════════════════════════════════════════════════════════

  All tokens Z = [D_tokens, Φ_tokens, C_tokens, W_tokens, C_asset]

  For each attention layer:
    score_ij = (Q_i @ K_j) / sqrt(d) + B_phys[i, j] + B_learned[i, j]

  Physical graph bias B_phys (initial values):

    Strong positive bias (+1.0):
      D_PV      ← W_Irrad
      D_Wind    ← W_WindSpeed
      D_Load    ← W_Temp
      D_BattP   ← Φ_SOC
      D_BattP   ← Φ_PV          # PV surplus → battery charging
      D_BattP   ← D_SOC
      C_Balance ← D_Load / D_PV / D_Wind / D_BattP

    Moderate bias (+0.5):
      D_Load  ↔ Φ_Load
      D_PV    ↔ Φ_PV
      D_Wind  ↔ Φ_Wind
      D_BattP ↔ Φ_BattP
      D_SOC   ↔ Φ_SOC

    Weak/no bias (0.0):
      D_Load  ↔ D_PV            # cross-component: let data learn
      D_Load  ↔ D_Wind

    Negative bias (−1.0, strong prior against):
      D_PV    ← D_BattP         # battery doesn't cause PV output
      D_Wind  ← D_BattP         # battery doesn't cause wind

  B_learned: free parameter matrix, initialized to zero
  Training schedule: B_phys annealed from 1.0 → 0.1 over training

  **Expansion rules for patch tokens**: When data tokens are daily patches
  (D_{c,d}) and weather tokens are variable×patch (W_{v,p}), B_phys must
  expand from conceptual edges to the full token×token matrix. Rules:

    D_PV_d* ← W_Irrad_p* : +1.0  (all PV days attend to all irradiance patches)
    D_Wind_d* ← W_WindSpeed_p* : +1.0
    D_Load_d* ← W_Temp_p* : +1.0
    D_BattP_d* ← D_SOC_d* : +1.0  (same-day SOC→BattP coupling)
    C_balance ← D_Load_d* / D_PV_d* / D_Wind_d* / D_BattP_d* : +1.0

    Temporal proximity bonus (optional):
      D_{c, recent_days} ↔ W_{v, near_horizon_patches} : +0.3
      D_{c, older_days}  ↔ W_{v, far_horizon_patches}  : 0.0

    Self-attention within same component across days (optional):
      D_{c, d_i} ↔ D_{c, d_j} : +0.2  (encourages smooth day-to-day patterns)

═══════════════════════════════════════════════════════════════
STEP 4: Component-wise Horizon Decoder
═══════════════════════════════════════════════════════════════

  For each component c ∈ {load, pv, wind, batt_p}:
    For each horizon step t ∈ {0..95}:

      q_{c,t} = HorizonQuery(c, t, y_mark[:, t, :])
        # query encodes: which component, which horizon, calendar time

      z_{c,t} = CrossAttention(q_{c,t}, Z, Z)
        # attends to all tokens, weighted by relevance

      residual_c[t] = OutputHead(z_{c,t}, Φ_c, weather_future[:, t, :])
        # small MLP: concat(attended, physics_twin, weather_at_t) → scalar

  For battery (special):
    Δpolicy[t] = BatteryDecoder(q_batt,t, Z, Φ_BattP, Φ_SOC, C_soc)
    batt_power[t] = batt_theory[t] + Δpolicy[t]
    soc[t+1] = soc[t] + (κ_charge·charge − discharge/κ_discharge) · Δt
    Clip soc to [0, E_max]

    κ_charge / κ_discharge are **effective energy conversion coefficients**,
    not strict physical efficiencies. The dataset may encode battery power/SOC
    at a specific metering boundary (AC vs DC side, timestamp alignment, synthetic
    data rules). Observed SOC-vs-power empirical slopes may differ from physical
    η ∈ [0.80, 0.99]. Bounds should be calibrated from data before enforcing
    strict physical ranges. Current code uses η ∈ [0.80, 0.99]; the new
    architecture should start with wider bounds (e.g. κ ∈ [0.5, 1.5]) and
    optionally tighten after data calibration.

═══════════════════════════════════════════════════════════════
STEP 5: Real-Unit Constrained Reconstruction
═══════════════════════════════════════════════════════════════

  Load_pred  = Load_theory  + residual_load           # additive
  PV_pred    = PV_theory    * exp(residual_pv)        # multiplicative
  Wind_pred  = Wind_theory  * exp(residual_wind)      # multiplicative
  Batt_pred  = Batt_theory  + Δpolicy                 # action residual
  SOC_pred   = SOC recurrence from Batt_pred          # state-space

  pred_net_real = Load_pred − PV_pred − Wind_pred + Batt_pred

  pred_net_norm = (pred_net_real − target_mean) / target_std

  Multiplicative correction for PV/Wind prevents generating
  non-zero PV at night or Wind in calm conditions.
  Load and Battery use additive — Load is behavioral (additive noise),
  Battery is control (action adjustment on top of baseline).

═══════════════════════════════════════════════════════════════
STEP 6: Loss
═══════════════════════════════════════════════════════════════

  L = L_net_MSE(pred_net_norm, y_target)
    + λ_load   · MAE(Load_pred,  load_true)
    + λ_pv     · MAE(PV_pred,    pv_true)
    + λ_wind   · MAE(Wind_pred,  wind_true)
    + λ_batt   · MAE(Batt_pred,  batt_true)
    + λ_soc    · MAE(SOC_pred,   soc_true)
    + λ_phys   · L_physical_violation  # SOC bounds, PV night, etc.
    + λ_res    · L_residual_regularization

  All component MAE computed in REAL MW / MWh units.
  Component weights (λ_*) follow curriculum schedule.
  The component MAE uses PREDICTIONS (theory + residual), not theory alone.
  This makes per-component residuals identifiable:
    δload is pinned by load_true
    δpv is pinned by pv_true
    → no free ε to shift between components while preserving net

  N07 mitigation: component weights start low, architectural
  separation (component tokens) may reduce the component-vs-aggregate
  trade-off compared to the shared-encoder baseline. This is
  explicitly tested as ablation axis A4 vs A6.
```

### 4.3 Key Design Decisions

| Decision | Rationale | Alternative Rejected |
|----------|-----------|---------------------|
| PV/Wind multiplicative residual | Prevents generating PV at night, Wind in calm | Additive + nonnegative clamp (less physical) |
| Load additive residual | Behavioral noise is additive in nature | Multiplicative (would distort base load shape) |
| Battery action residual (Δpolicy) | Battery is a control variable, not a physical response | Direct power prediction (loses SOC consistency) |
| Graph bias annealing | Early physics guidance, late data flexibility | Fixed bias (too rigid) or pure free attention (loses prior) |
| Component loss on PREDICTIONS not theory | Makes residual identifiable; theory is just a prior | Theory-only supervision (leaves residual free to cancel) |
| Horizon-aware per-step decoder | Preserves temporal alignment with weather | Per-token FFN (loses step-level weather alignment) |

---

## 5. Phased Implementation Plan

### Phase 0: Correctness Fixes (existing PhysFormer)

**P0-1: Real-unit power balance** (~20 lines, `model.py`)
- Fix `_build_component_net` to operate in real units
- Add `_denorm_component_residual` helper
- Re-run 1 seed baseline to measure MAE shift
- **Verification**: pred_net_real ≈ theory_net_real + residual_net_real in MW

**P0-2: Load branch uses load history** (~15 lines, `physical_layer.py`)
- Change `_load_branch` GRU/autoreg input from `x_net_hist_real` to `x_load_hist_real`
- **Verification**: Load theory should no longer be contaminated by PV/Wind/Battery fluctuations in net history

**P0-3: Data pipeline adds component history** (~30 lines, `data.py`)
- Add `x_component_hist` to model forward inputs
- Columns: `[p_load_mw, p_pv_mw, p_wind_mw, p_battery_mw, e_battery_soc_mwh]`
- Backward-compatible: old configs continue to work with `use_component_hist=False`
- **Verification**: training runs without error on current config

### Phase 1: Minimal iTransformer Proof-of-Concept

**P1-1: 8-token inverted encoder (maps to ablation A1)** (~200 lines, new file `physformer/models/physformer/igt_model.py`)
- 5 component data tokens (each: GRU(672) → d_model)
- 3 weather tokens (future weather summary, one per variable)
- **No physics token** — pure data-driven inverted Transformer
- Pure self-attention (no graph bias, no twin tokens)
- Simple per-token FFN decoder (d_model → pred_len)
- Real-unit power balance aggregation
- **No component loss** — net MSE only (fair comparison with baseline)
- **Verification**: 1 seed, MAE should be within 2× of baseline (~2e-3 to ~4e-3 MW). If >> 4e-3, debug before proceeding.

**P1-2: +1 physics token (maps to ablation A2)** (~20 lines)
- Add `theory_net_real` as a single physics token → 9 tokens total
- Tests whether physics as a peer token (not FiLM) adds value in the inverted architecture

### Phase 2: Physics Integration

**P2-1: Physics-Twin tokens** (~50 lines)
- Replace single physics token with 5 Φ_c tokens + 3 constraint tokens
- Twin tokens initialized from PhysicalLayer outputs

**P2-2: Graph-biased attention** (~30 lines)
- Add `B_phys` bias matrix
- Annealing schedule

**P2-3: Component-wise horizon decoder** (~100 lines)
- Replace per-token FFN with horizon query decoder
- Integrate future weather at each horizon step

### Phase 3: Battery + Component Supervision

**P3-1: State-space battery decoder** (~80 lines)
- Action residual (Δpolicy) instead of direct power prediction
- SOC recurrence with physical constraints

**P3-2: Component-level supervision** (~40 lines in loss)
- Add component MAE terms on predictions (theory + residual)
- Curriculum schedule for component weights

---

## 6. Ablation Experiment Chain

| ID | Configuration | Tokens | Physics | Graph Bias | Comp Loss | Battery | Purpose |
|----|--------------|--------|---------|------------|-----------|---------|---------|
| A0 | Current PhysFormer c23 (post-P0 fix) | — | FiLM | — | theory only (current) | signed-power recurrence | Baseline |
| A1 | Pure iTransformer, **8 token**, no physics | 8: 5 component + 3 weather | none | none | none | FFN→96 | Lower bound: inverted attn value alone |
| A2 | A1 + 1 physics token | 9: A1 + 1 theory_net | single token | none | none | FFN→96 | Physics-as-token vs FiLM |
| A3 | A2 + Physics-Twin tokens (flat components) | 5+5+3+24: 5 comp + 5 twin + 3 constraint + weather | twin tokens | none | none | FFN→96 | Twin token value with flat component tokens |
| A4 | A3 + Graph-biased attention | same | twin tokens | B_phys | none | FFN→96 | Graph prior value |
| A5 | A4 + Component-wise horizon decoder | same | twin tokens | B_phys | none | horizon decoder | Decoder value: per-step weather alignment |
| A6 | A5 + Component losses | same | twin tokens | B_phys | λ_i > 0 | horizon decoder | **Key ablation**: does architectural separation resolve N07 trade-off? |
| A7 | A6 + State-space battery decoder | same | twin tokens | B_phys | λ_i > 0 | SOC recurrence + Δpolicy | Battery physics value |
| A8 | A7 + Daily patch tokens | **35**+5+3+24: 5 comp × 7 days + twin + constraint + weather | twin tokens | B_phys | λ_i > 0 | SOC recurrence + Δpolicy | Full iGT: daily patch temporal structure |

**Testing protocol**: Each configuration ≥ 3 seeds (2025, 2026, 2027). Compare on:
- Aggregate: MAE, MSE, RMSE, net_ramp_violation (real MW)
- Component: MAE per component (real MW)
- Physics: theory_mae, SOC bounds violation, PV night violation
- Diagnostics: variance decomposition (C08), attention map edge masking

**Additional data split — blocked temporal** (required for generalization claims):
The current split (different portfolios, same time period) is closer to
cross-portfolio than temporal generalization. Add a blocked temporal split:
  - train: Jan–Sep
  - val:   Oct
  - test:  Nov–Dec
Or the stricter variant:
  - train portfolios on Jan–Sep
  - test unseen portfolio on Oct–Dec
This better supports the claim that physics guidance improves extrapolation
to unseen time periods, not just unseen portfolio IDs within the same year.

**Compute budget estimate** (vGPU-32GB, 3-parallel, ~6h/run):
- P0 baseline re-run: 1 seed = 1 run
- A1→A8 full ablation: 8 configs × 3 seeds = 24 runs = ~8 batches = ~48h GPU
- Strategic subset (A0, A2, A4, A6, A8): 5 configs × 3 seeds = 15 runs = ~30h GPU

---

## 7. Falsification Protocol

### Claim: Architectural component separation (iTransformer tokens) eliminates the N07 component-vs-aggregate trade-off

**Falsification condition**: A6 (twin tokens + graph bias + component losses) shows:
- Component MAE improvement < 10% vs A4 (no component loss) on ≥ 3 of 5 components, OR
- Aggregate MAE degradation > 3% vs A4

**Pass condition**: A6 improves ≥ 3 of 5 component MAEs by ≥ 10%, with aggregate MAE degradation ≤ 1.5% vs A4.

**Baseline**: A4 (same architecture, no component loss)

**Confound control**: If A6 fails, test whether the failure is weight-dependent (sweep λ_i ∈ {0.01, 0.03, 0.05, 0.10}). If NO weight passes, the trade-off is fundamental (not architecture-dependent). If some weight passes, the trade-off is mitigated but not eliminated by architecture.

### Claim: Graph-biased attention improves component-level accuracy vs free attention

**Falsification condition**: A4 (graph bias) shows no component MAE improvement over A3 (free attention) on ≥ 3 of 5 components.

**Pass condition**: A4 improves ≥ 3 of 5 component MAEs by ≥ 5%, with no aggregate degradation.

### Claim: Multiplicative PV/Wind residual eliminates night-time PV and calm-condition Wind artifacts

**Falsification condition**: A8 night-time PV > 1% of daytime peak PV in ≥ 2 of 3 seeds.

---

## 8. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| P0 normalization fix shifts MAE unfavorably | Medium | Low | Accept; correctness > absolute numbers. Re-baseline post-fix. |
| iTransformer PoC MAE >> baseline | Low-Medium | High | Abort iGT direction; explore dual-encoder (direction C from brainstorm) |
| Component loss still hurts aggregate (N07 persists) | Medium | High | Drop component loss; fall back to theory-only supervision + architecture-driven disentanglement |
| 68-token attention has unexpected computational bottleneck | Low | Low | Reduce daily patches (7→4 days), weather patches (24→8), or twin tokens (5→1) |
| Graph bias too constraining → model can't learn real interactions | Low | Medium | Anneal B_phys → 0 by mid-training; let B_learned take over |
| Battery Δpolicy learns to ignore physics constraints | Medium | Medium | Add physics violation penalty to loss; enforce SOC bounds strictly |
| New architecture needs hyperparameter re-tuning | High | Medium | Start from current c23 hp (d_model=256, etc.); grid search only if PoC fails |

---

## 9. File Change Summary

| Phase | File | Action | Lines (est.) |
|-------|------|--------|-------------|
| P0 | `physformer/models/physformer/model.py` | Fix `_build_component_net`, add denorm helper | ~30 |
| P0 | `physformer/models/physformer/physical_layer.py` | Load branch uses load history | ~15 |
| P0 | `physformer/data.py` | Add component history columns | ~30 |
| P1 | `physformer/models/physformer/igt_model.py` | **New**: PhysFormer-iGT model | ~300 |
| P1 | `physformer/models/physformer/igt_tokenizer.py` | **New**: Tokenization modules | ~150 |
| P2 | `physformer/models/physformer/igt_attention.py` | **New**: Graph-biased attention | ~80 |
| P2 | `physformer/models/physformer/igt_decoder.py` | **New**: Horizon decoder | ~150 |
| P3 | `physformer/models/physformer/igt_battery.py` | **New**: State-space battery decoder | ~100 |
| P3 | `physformer/loss.py` | Add component loss terms on predictions | ~50 |
| — | `configs/physformer_igt_*.yaml` | Config files for A1→A8 | ~8 files |

---

## 10. References

- C08: Component error covariance cancellation framework (`ara/logic/claims.md`)
- C09: Selective detach aggregate dominance (outcome claim)
- C10: Detach disables encoder-depth→cancellation channel (mechanism)
- N07: Component loss too strong degrades aggregate (dead end)
- N84: O37 bias-clearance mechanism refuted (single-seed outlier)
- N92/N93: M1/M2 mechanism candidates falsified
- iTransformer: Liu et al., 2023 (https://arxiv.org/abs/2310.06625)
- Current PhysFormer architecture: `physformer/models/physformer/model.py`
- P2 mechanism search design: `docs/plans/2026-05-25-detach-mechanism-search.md`
