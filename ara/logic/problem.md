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

### G1: No mechanism to prevent cross-component error propagation in aggregated forecast
- **Statement**: When a single scalar residual corrects the aggregate net injection, an error in one component (e.g., Load under-prediction) can be "compensated" by the residual in a way that corrupts the predictions of other components (e.g., PV over-correction).
- **Caused by**: O2
- **Existing attempts**: V3/V4 scalar residual — functional but allows cross-contamination.
- **Why they fail**: A single degree of freedom cannot disentangle corrections for 4 independent error sources.

### G2: No principled method to set the physics-vs-data trade-off
- **Statement**: The optimal component loss weight varies with architecture, data distribution, and training stage. There is no theoretical guidance for setting this weight.
- **Caused by**: O3, O4
- **Existing attempts**: Manual grid search (0.01, 0.05, 0.1) — heuristic, not principled.
- **Why they fail**: The optimal weight is non-stationary during training (early training benefits from more physics guidance; late training benefits from more data freedom).

### G3: Load component has no effective physics prior
- **Statement**: Unlike PV/Wind/Battery which have clear physical equations, Load has no first-principles model. The "Load physics" branch is essentially a learned function with calendar features — not true physics guidance.
- **Caused by**: O5
- **Existing attempts**: Calendar embeddings + load proxy feature — helps but Load error remains 18x Wind.
- **Why they fail**: Calendar provides temporal context but does not encode the actual behavioral determinants of load (pricing, occupancy, industrial schedules).

## Key Insight

- **Insight**: Embed physics equations as learnable FiLM conditioning (not hard constraints) with per-component residual corrections (not scalar), and use curriculum training to progressively shift from physics-driven to data-driven optimization. This gives each component its own gradient path, prevents cross-contamination, and lets the model discover when to trust physics vs. when to override it.
- **Derived from**: O1, O2, O3, O4
- **Enables**: A forecasting model that is (a) physically consistent at the component level, (b) accurate at the aggregate level, (c) honest about which components benefit from physics and which don't.

## Assumptions

- A1: The four-component decomposition (Load - PV - Wind + Battery) is a complete and correct representation of the VPP net power balance. No significant unmodeled components exist.
- A2: Physics equations (PV irradiance model, Wind cubic law, Battery SOC dynamics) are directionally correct even if magnitude-imprecise.
- A3: Per-component ground truth (or reasonable proxy) is available for training supervision. Without it, component-consistent residual cannot be trained.
- A4: The temporal resolution (e.g., 15-min or 1-hour) is sufficient to capture the relevant dynamics of all four components.
- A5: Standard SGD with AdamW is sufficient as the optimizer; no second-order or specialized solver is required.
