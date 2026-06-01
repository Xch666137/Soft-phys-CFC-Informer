# Model Architecture Configuration

## PhysFormer (Shared Across V4/V5)

| Parameter | Value | Notes |
|-----------|-------|-------|
| d_model | 512 | Transformer hidden dimension |
| n_encoder_layers | 3 | Self-attention layers |
| n_decoder_layers | 1 (V4) / 2 (V5) | Cross-attention decoder layers |
| n_heads | 8 | Multi-head attention |
| d_ff | 2048 | Feed-forward dimension |
| dropout | 0.1 | Regularization |
| activation | GELU | Transformer FFN activation |
| input_len | 96 | 24 hours at 15-min |
| output_len | 96 | 24 hours ahead at 15-min |

## Theory Branches

| Component | Architecture | Input Features |
|-----------|-------------|----------------|
| PV | Irradiance × Temp (1st order) + small MLP | GHI, temperature |
| Wind | Learnable cubic-like MLP (3-layer, 32-dim) | Wind speed at hub height |
| Battery | SOC recurrence + MLP | SOC(t-1), battery params |
| Load | Calendar embedding (learned, 16-dim) + MLP | hour, weekday, month |

## FiLM Conditioning

| Parameter | Value |
|-----------|-------|
| Physics feature dim | ~10 (irradiance, temp, wind speed, calendar) |
| FiLM MLP | 2-layer, hidden=128 |
| Application point | After each encoder self-attention sublayer |

## Residual Heads (V5)

| Parameter | Value |
|-----------|-------|
| Number of heads | 5 (Load, PV, Wind, Batt Power, Batt SOC) |
| Head architecture | Linear → GELU → Linear (per component) |
| Initialization | N(0, 0.01) — V5; N(0, 0.05) — V5.5 planned |
| Output | Scalar per timestep per component |

## Temporal Decoder (V5 only)

| Parameter | Value |
|-----------|-------|
| time_proj | Linear(time_mark_dim → d_decoder) |
| time_mark_dim | ~5 (hour, weekday, month, holiday, etc.) |

## Model Size Comparison

| Variant | ~Parameters | Notes |
|----------|------------|-------|
| V3/V4 (scalar residual) | ~3.0M | 1-dim residual |
| V5 (per-component residual) | ~3.5M | +0.5M from 5 residual heads + temporal decoder |
