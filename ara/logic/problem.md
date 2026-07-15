# Problem Specification

## Observations

### O1: Pure deep-learning VPP forecasts are physically inconsistent
- **Statement**: Black-box Transformer models (Informer, iTransformer, LSTM baselines) for VPP aggregated net power forecasting produce predictions that violate physical constraints — power balance between generation and load doesn't hold, battery SOC accumulates errors, and component-level attributions are physically implausible.
- **Evidence**: Baseline benchmark runs (V3 and earlier), where Theory MAE = 4.874 kW indicates substantial deviation from physics-constrained predictions.
- **Implication**: A purely data-driven model has no mechanism to respect known physical laws, so it must learn them from data — requiring excessive capacity and still failing on rare/edge cases.

### O2: VPP net power is a composite of heterogeneous components with different dynamics
- **Statement**: Aggregated net power = Load - PV - Wind + Battery. Load is human-behavior-driven (calendar, habits, pricing); PV is weather-driven (irradiance, temperature); Wind is weather-driven (wind speed, air density); Battery is control-driven (SOC state, dispatch policy). These fundamentally different dynamics are collapsed into a single scalar prediction target.
- **Evidence**: V4 component-level analysis: Load MAE = 14.707 kW vs Wind MAE = 0.825 kW (18x difference). Different components have different predictability and different optimal modeling strategies.
- **Implication**: A single homogeneous model architecture cannot optimally capture all four component dynamics. The architecture must be heterogeneous in how it handles each component.

### O3: Physics equations exist but are incomplete
- **Statement**: PV output ≈ f(irradiance, temperature), Wind output ≈ f(wind speed³), Battery power = f(SOC, efficiency). These equations provide useful priors but are simplifications — they ignore inverter clipping, wake effects, temperature coefficients, aging, etc.
- **Evidence**: V4 Theory MAE = 3.811 kW, which is substantial but 21.8% better than V3 — physics provides a meaningful but incomplete signal.
- **Implication**: Physics should be used as a learned prior (inductive bias), not as a hard constraint. There must be room for data-driven correction of the physics.

### O4: Component supervision creates a theory-vs-accuracy Pareto trade-off
- **Statement**: Adding per-component loss improves physical consistency but can degrade aggregate accuracy when the weight is too high. There exists a Pareto frontier where increasing component supervision beyond a certain point trades aggregate accuracy for physical consistency.
- **Evidence**: V4 (weight 0.05) → best balance. V4.2 (stronger component loss) → Theory MAE improved to 2.981 kW but Test MAE degraded to 2.048 kW. V5 (weight 0.1) → component metrics at best but aggregate MAE +7.5%.
- **Implication**: The component loss weight is a critical hyperparameter that controls position on the physics-accuracy Pareto frontier. It must be tuned, not set arbitrarily.

### O5: Load is the dominant error bottleneck and resists physics-based modeling
- **Statement**: Load forecasting error (MAE = 14.7 kW) is 18x worse than Wind (0.83 kW) and 3.7x worse than PV (4.0 kW). Load is driven by human behavior patterns that physics equations cannot capture.
- **Evidence**: V4 component-level error breakdown. Load dominates both the magnitude and variance of forecast error.
- **Implication**: The physics-guided approach is fundamentally asymmetric — it helps PV/Wind/Battery more than Load. Load may need a fundamentally different modeling approach (behavioral/statistical rather than physics-based).

## Gaps

### G1: Shared Transformer encoders create a cross-component error covariance cancellation channel
- **Statement**: Under standard Transformer architectures with a shared encoder, the residual head sends gradient through the encoder to induce cross-component error correlations that cancel under net = load − pv − wind + batt (C08 capacity regime). This masks individual component inaccuracy in aggregate metrics — worse component forecasts can produce better aggregate MAE through signed error cancellation. Neither per-component residual corrections (V5, symptom treatment) nor gradient-side detaching (C10 detach, gradient hack) addresses the architectural root cause: the shared representation space enables the cancellation channel.
- **Caused by**: O2, O4
- **Existing attempts**: V3/V4 scalar residual (allows cross-contamination), V5 per-component residual (symptom treatment — reduces but doesn't eliminate), c23 selective gradient detach (blocks residual→theory gradient but leaves shared encoder intact).
- **Why they fail**: All operate within a shared-encoder architecture. The encoder's shared FFN layer mixes component representations by design; any auxiliary loss or gradient modification can only constrain the mixing, not prevent it.

### G2: Fixed architectural priors (physics tokens, graph bias, richer decoders) amplify overfitting on heterogeneous multi-portfolio data
- **Statement**: While component-token separation (C11) provides a clean architectural prior, the ablation chain A2-A5 shows that ANY additional fixed prior — physics tokens, graph-biased attention, richer decoders, weather conditioning — systematically degrades Test MAE while improving Val MSE (C12). The root mechanism: fixed priors impose uniform coupling forms across heterogeneous portfolios, memorizing training-distribution-specific patterns rather than learning transferable structure.
- **Caused by**: O2 (heterogeneous portfolio coupling), O3 (physics equations are incomplete)
- **Existing attempts**: A2 (+physics token), A3 (+twin+constraint tokens), A4 (+graph bias), A5 (+horizon decoder) — all degrade Test MAE. Phase B MCP replaces fixed priors with data-driven coupling discovery via self-supervised pretraining.
- **Why they fail**: Fixed priors provide training-set-specific guidance that fails to generalize across portfolios with different DER compositions, weather regimes, and load behaviors. The Val-better → Test-worse divergence (C12) is a universal pattern across 4 architecture variants.

### G3: Load component has no effective physics prior
- **Statement**: Unlike PV/Wind/Battery which have clear physical equations, Load has no first-principles model. The "Load physics" branch is essentially a learned function with calendar features — not true physics guidance.
- **Caused by**: O5
- **Existing attempts**: Calendar embeddings + load proxy feature — helps but Load error remains 18x Wind.
- **Why they fail**: Calendar provides temporal context but does not encode the actual behavioral determinants of load (pricing, occupancy, industrial schedules).

## Key Insight

- **Insight**: Tokenize each DER component independently and apply self-attention across component tokens rather than time steps. Under standard Transformer architectures, a shared encoder mixes all component representations — the residual head can send gradient through the encoder to induce cross-component error correlations that cancel under net = load − pv − wind + batt (C08 capacity regime) and mask individual component inaccuracy. Component-token separation (inverted attention, iTransformer paradigm) gives each DER its own representation space, preventing this cancellation at the architecture level rather than through gradient hacks (C10 detach) or fixed physics priors (C12). The real-unit power balance decoder preserves physical consistency without requiring explicit physics equations as model inputs.
- **Derived from**: O1, O2, O4 (shared-encoder cross-component error contamination), O3 (physics equations exist but are incomplete — motivating data-driven over physics-guided approaches)
- **Enables**: A forecasting model that is (a) architecturally free of shared-encoder cancellation (C10→C11), (b) accurate at the aggregate level with dramatically smaller cross-seed variance (C11), (c) decomposable into learned per-component forecasts for dispatch preparation when augmented with self-supervised component pretraining (C13).

## Assumptions

- A1: The four-component decomposition (Load - PV - Wind + Battery) is a complete and correct representation of the VPP net power balance. No significant unmodeled components exist.
- A2: Physics equations (PV irradiance model, Wind cubic law, Battery SOC dynamics) are directionally correct even if magnitude-imprecise.
- A3: Per-component ground truth (or reasonable proxy) is available for training supervision. Without it, component-consistent residual cannot be trained.
- A4: The temporal resolution (e.g., 15-min or 1-hour) is sufficient to capture the relevant dynamics of all four components.
- A5: Standard SGD with AdamW is sufficient as the optimizer; no second-order or specialized solver is required.
