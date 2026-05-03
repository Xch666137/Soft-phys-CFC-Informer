# PhysFormer V4 Phase 1.1 Fix Plan — Peer Review Target

## Background
V4 Phase 1 introduced calendar embeddings, load proxy, and component loss supervision. Results: Theory MAE -21.8%, 3.4x convergence speedup, but final MAE regressed +2.3%. Dual-draft of V4 results identified three root causes + one config improvement.

## Proposed Fixes

### Fix 1: Normalize Component Losses to Dimensionless Space
**File:** `physformer/utils/losses.py`, `compute_terms()` method
- Current: component MAEs computed in real MW units after `denorm_aux(y_aux)`
- Proposed: compute `theory_comp_norm = (component_theory_real - aux_mean) / aux_std`, then compute MAE against raw `y_aux` (already normalized)
- Effect: all loss terms now in same normalized space; `component_loss_weight` decreases from 0.05 → 0.02

### Fix 2: TemporalDecoder Time Conditioning
**File:** `physformer/models/temporal_decoder.py`
- Current: `forward()` accepts `y_mark` but ignores it; uses static learned `query_pos`
- Proposed: add `self.time_proj = nn.Linear(time_enc_in, d_model)` in `__init__`; modify query to `query = query_pos + time_proj(y_mark)`
- Also: pass `time_feat_dim` from PhysFormer to TemporalDecoder constructor

### Fix 3: Training Schedule Adjustment
**File:** `configs/physformer_v3.yaml`
- `early_stop_start_epoch`: 10 → 20 (allow WarmRestarts to explore before counting)
- `patience`: 25 → 35 (match 100-epoch budget)
- `component_loss_weight`: 0.05 → 0.02 (weaker component supervision)

### Fix 4: Configurable WarmRestart Parameters
**File:** `physformer/exp/exp_physformer.py`, `physformer/runner/config.py`
- Make `T_0` and `T_mult` configurable via args (defaults: 15, 1 — preserves current behavior)
- Register keys in config.py: `restart_t0`, `restart_t_mult`

## Files Modified
- `physformer/utils/losses.py`
- `physformer/models/temporal_decoder.py`
- `physformer/models/physformer.py`
- `physformer/exp/exp_physformer.py`
- `physformer/runner/config.py`
- `configs/physformer_v3.yaml`

## Expected Outcomes
1. Component MAE values in ~0.1-2 range (comparable to net_mse ~0.4)
2. TemporalDecoder queries become calendar-aware → better for weekend/holiday patterns
3. Training runs deeper into WarmRestart cycles (50-60 epochs instead of 36)
4. Final MAE should no longer regress; Theory MAE improvement should translate to net improvement
