# V4.1 Regression Analysis: Why Did Theory MAE Degrade?

## The Puzzle
V4.1 added a sigmoid gate on residual (`pred = theory + sigmoid(alpha) * residual`). The gate only affects the residual path. Yet:

| Metric | V4 | V4.1 | Δ |
|--------|----|----|-----|
| Theory MAE | **3.811 kW** | 4.966 kW | **+30%** |
| Residual std | **4.312 kW** | 5.076 kW | +18% |
| Final MAE | **1.976 kW** | 2.002 kW | +1.3% |
| Val MSE best | 0.381 | **0.379** | -0.5% |

Theory MAE regressed 30% — but the gate shouldn't affect theory_net computation. Theory_net is computed independently upstream (physical_layer.py:331): `theory_net_real = load_theory - pv_theory - wind_theory + battery_power`.

## Candidate Mechanisms

### 1. Gradient Coupling Through Shared Encoder
The encoder produces `weather_latent` used by BOTH `UnifiedResidualHead` (gated) AND `WeatherFusion` (upstream of physical layer). When gate halves residual gradient, encoder weights receive weaker signal → affects theory_net indirectly through the shared representation.

### 2. Loss Landscape Shift
V4.1 has: (a) normalized component losses (aux_std space), (b) reduced component_loss_weight (0.02 vs 0.05 MW), (c) alpha parameter. Training resumed from V4 checkpoint with new loss → optimizer moved theory parameters toward a different optimum.

### 3. TemporalDecoder time_proj Disruption
V4.1 added `self.time_proj` to TemporalDecoder. This layer was randomly initialized (Xavier gain=0.1). Its output modifies the query that goes into cross-attention, which produces `coarse_future` → `weather_latent` → physical layer input. A poorly initialized time projection could have degraded the decoder's output, indirectly harming theory_net.

### 4. Val MSE ≠ Test MSE
Val MSE improved (0.379 vs 0.381) but Test MSE regressed (7.68e-6 vs 7.35e-6). This suggests overfitting to the validation distribution — the model potentially learned validation-specific patterns at the expense of generalization.

## V4 vs V4.1 Changes List
1. Loss: MW→normalized component MAE, weight 0.05→0.02
2. TemporalDecoder: +time_proj
3. Residual: +sigmoid(alpha) gate
4. Config: patience=25→20, early_stop_start=10→20, restart T_0/T_mult configurable

## Key Question
Which of these 4 changes caused Theory MAE to regress? And which contributed to Val→Test divergence?
