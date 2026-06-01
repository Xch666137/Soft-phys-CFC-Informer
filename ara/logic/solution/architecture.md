# Architecture

PhysFormer is a physics-conditioned Transformer for VPP aggregated net power forecasting.
It decomposes the forecasting problem into theory-driven and residual components across
four asset types (Load, PV, Wind, Battery).

## Component Graph

### Input Layer
- **Historical net injection**: (B, L_in, 1) — past aggregated net power
- **Physics features**: (B, L_in, F_phys) — irradiance, temperature, wind speed, calendar
- **Time marks**: (B, L_in + L_out, F_time) — hour, weekday, month, holiday flags
- **Portfolio ID**: (B,) — which VPP portfolio (for ID embedding)

### Encoder (Shared Transformer)
- **Inputs**: Historical net injection + physics features + time marks, projected to d_model
- **Operation**: N_enc layers of self-attention + FiLM conditioning
  - FiLM: γ, β = MLP(physics_features); h_out = γ ⊙ h + β
  - Applies per-channel affine transformation conditioned on physics
- **Outputs**: (B, L_in, d_model) encoded representations

### Theory Branches (Per-Component, Physics-Driven)
Four independent theory heads, each computing a physics-based estimate:

- **PV Theory**: P_pv ≈ η * G * (1 + α(T - T_ref)) * N_panels
  - Where G = irradiance, T = temperature, η = efficiency, α = temp coefficient
- **Wind Theory**: P_wind ≈ 0.5 * ρ * A * v³ * Cp
  - Where v = wind speed, simplified as learnable cubic fit
- **Battery Theory**: P_batt = f(SOC_t, SOC_{t-1}, efficiency, capacity)
  - SOC accumulation constraint: SOC_t = SOC_{t-1} + η_ch * P_ch * Δt - (1/η_dis) * P_dis * Δt
- **Load Theory**: P_load = f(calendar_features) — essentially a learned temporal model

Each theory branch outputs: (B, L_out, 1) scalar per-timestep component prediction.

### Temporal Decoder (Time-Conditioned)
- **Inputs**: Encoded representations + time_proj(y_mark)
- **Operation**: Time-aware decoding of encoder outputs to prediction horizon
  - time_proj: Linear projection of future time marks → decoder conditioning
- **Outputs**: (B, L_out, d_dec) decoded representations

### Residual Heads (Per-Component, Data-Driven)
Five independent residual heads (one per component + Battery SOC):
- **Operation**: MLP(decoded_representation) → scalar residual per timestep
- **Initialization**: Small std (0.01–0.05) so residuals start near zero
- **Outputs**: (B, L_out, 1) per-component residual correction

### Aggregation (Power Balance)
```
pred_net = (load_theory + load_res)
         - (pv_theory + pv_res)
         - (wind_theory + wind_res)
         + (batt_theory + batt_res)
```
This preserves the VPP power balance identity: Net = Load - Generation + Battery.

### Output
- **pred_net**: (B, L_out, 1) — aggregated net injection forecast
- **pred_components**: (B, L_out, 5) — per-component forecasts (for loss computation)
- **theory_components**: (B, L_out, 5) — theory-only estimates (for monitoring)

## Model Dimensionality

| Parameter | V4 | V5 |
|-----------|-----|-----|
| d_model | 512 | 512 |
| n_encoder_layers | 3 | 3 |
| n_decoder_layers | 1 | 2 |
| n_heads | 8 | 8 |
| d_ff | 2048 | 2048 |
| input_len | 96 | 96 |
| output_len | 96 | 96 |
| residual_dim | 1 (scalar) | 5 (per-component) |
| ~params | ~3M | ~3.5M |

## Key Design Choices

- **Shared encoder + separate heads**: Shared representation captures cross-component interactions; separate theory/residual heads maintain independent gradient paths per component.
- **FiLM over concatenation**: FiLM provides multiplicative + additive conditioning, more expressive than feature concatenation at the input.
- **Identity shortcut for residual**: No gate — gradient flows unimpeded from component loss to theory branches.
- **MAE for component loss**: Linear penalty in kW space prevents Load from dominating component gradient (cf. H02).
- **Time conditioning on decoder**: Calendar features projected into decoder so residuals are time-aware.
