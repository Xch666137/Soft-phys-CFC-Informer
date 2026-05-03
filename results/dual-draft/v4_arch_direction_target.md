# PhysFormer V4 Architecture Direction: Load-Specific Modeling with Confidence Gating

## Current State (V3)

PhysFormer uses a unified `theory_net + residual_net` architecture for VPP net injection (PV + Wind - Load). Key results:

- Theory_net contributes 62.1% of prediction magnitude, residual 37.9%
- Residual degrades 26.2% of timesteps; 93.7% degradation when theory error < 0.32 kW
- High net-injection scenarios (PV-dominated, physics works well): degradation up to 48.9%
- Low net-injection / nighttime (load-dominated, no physics signal): theory struggles more

## The Core Problem

**Load is fundamentally different from PV/Wind:**

| Dimension | PV + Wind | Load |
|-----------|-----------|------|
| Primary driver | Environment (irradiance, wind, temperature) | Human behavior (schedules, holidays, seasons) |
| Physics model? | Yes — PV conversion equation, wind power curve | No — no equivalent first-principles model |
| Key missing variable | Cloud cover | Electricity price (not available in our data) |
| Temporal patterns | Diurnal (sun angle), weather fronts | Diurnal (commute), weekly (weekend), annual (holidays), seasonal |
| Predictability source | NWP weather forecasts | Calendar + autoregressive patterns |
| Suitability for physics-guided architecture | Excellent | Weak — temperature alone poorly explains load |

**Consequence:** The unified `theory_net` treats load with the same physics-based feature modulation (FiLM) as PV/Wind, but load's dominant drivers (calendar, human behavior) are not physics features. This misalignment is likely a key contributor to the 26.2% residual degradation rate.

## Proposed Direction: Dual-Stream Architecture

### Stream A — Physics-Guided (PV + Wind)
- Keep `theory_net` with FiLM modulation by weather features
- PV conversion physics, wind power curve, temperature effects
- Battery physics (SOC, power limits)
- **This stream retains the current architecture's strengths**

### Stream B — Variable-Attention (Load)
- **iTransformer-style** inverted attention: each variable (temperature, irradiance, wind_speed, historical load, calendar features) as a token
- Cross-variable attention captures: "how does temperature interact with weekday to determine load?"
- Temporal FFN captures per-variable seasonal/diurnal patterns
- Calendar embeddings: weekday/weekend, holiday, month, hour-of-day
- **Key insight from literature:** iTransformer achieves SOTA on Electricity dataset (MSE -18.75% vs standard Transformer) precisely because electrical load has strong variable-level correlations

### Gating Mechanism: Uncertainty-Confidence Fusion
- **MoGU** (Shavit & Goldberger, 2025): Each expert outputs Gaussian (mean + variance); gating weight ∝ 1/variance
- **Conf-SMoE** (2025): Confidence-guided gating with sigmoid-based scores to prevent expert collapse
- **Adaptation for PhysFormer:** Stream A outputs predictions + physics-based confidence (e.g., ∝ 1/|weather_gradient|), Stream B outputs predictions + data-driven confidence. Gate = learnable fusion weighted by confidence.

## Key Questions for Analysis

1. **Does separating PV/Wind from Load in the architecture make physical sense?** Or is the unified `theory_net` actually learning to automatically decompose them?

2. **Is iTransformer the right choice for the Load stream?** What are its limitations (O(N²) complexity in variables, performance on small-variable-count settings)?

3. **Can confidence gating replace our current post-hoc threshold approach?** MoGU-style uncertainty gating would be end-to-end learnable rather than heuristic.

4. **What evidence exists that Load ≠ PV/Wind in our own data?** Based on V3 results: degradation rate vs |true| magnitude shows a U-shape — theory_net excels at high net-injection (PV-dominated) but residual degrades more there, suggesting the residual is compensating for Load-related errors.

5. **Implementation complexity vs paper impact:** Does a dual-stream architecture add too much complexity for the contribution it provides? Or is the "physics + data-driven dual stream" narrative exactly what the paper needs?

## Relevant Literature (from web search)

- **iTransformer** (Liu et al., ICLR 2024 Spotlight): Inverted attention — variates as tokens, temporal FFN. SOTA on Electricity (321 variates) and Traffic. [openreview.net/forum?id=iiVdo6JFfk](https://openreview.net/forum?id=iiVdo6JFfk)
- **MoGU** (Shavit & Goldberger, Oct 2025): MoE with uncertainty-based gating for time series. Each expert predicts Gaussian, gate ∝ 1/variance. [arxiv.org/abs/2510.07459](https://arxiv.org/abs/2510.07459)
- **FMLP-iTransformer** (Energy Reports, Dec 2024): CNN + factorized MLP enhancement of iTransformer for load forecasting. Outperforms LSTM, GRU, Transformer on load data.
- **ASSA-iTransformer** (Energy and Buildings, Dec 2024): Adaptive signal decomposition + iTransformer for household load. R² near 1.0 on 5 real datasets.
- **Ziel et al.** (arXiv, 2024): GAM-based decomposition of load into weather-driven, calendar-driven, and socio-economic components. [arxiv.org/abs/2408.00507](https://arxiv.org/abs/2408.00507)
- **Chen et al.** (2025): PV disaggregation from net load using multi-scale temporal feature extraction + weather fusion. [arxiv.org/abs/2505.18747](https://arxiv.org/abs/2505.18747)
- **Conf-SMoE** (2025): Confidence-guided gating preventing expert collapse in sparse MoE. [arxiv.org/abs/2505.19525](https://arxiv.org/abs/2505.19525)
