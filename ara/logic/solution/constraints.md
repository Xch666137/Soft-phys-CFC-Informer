# Constraints

## Power Balance
- **Constraint**: $\hat{P}_{\text{net}} = \hat{P}_{\text{load}} - \hat{P}_{\text{pv}} - \hat{P}_{\text{wind}} + \hat{P}_{\text{batt}}$
- **Enforcement**: Architectural — the aggregation layer implements this identity exactly. It is not a soft constraint.
- **Implication**: Any error in the net forecast is exactly the sum of signed component errors. The decomposition is exact, not approximate.

## Battery SOC Accumulation
- **Constraint**: $SOC_t = SOC_{t-1} + \eta_{\text{ch}} P_{\text{ch},t} \Delta t - \frac{1}{\eta_{\text{dis}}} P_{\text{dis},t} \Delta t$
- **Enforcement**: Soft — the battery theory branch computes this recursively, but errors can accumulate over long horizons.
- **Bounds**: $SOC_{\text{min}} \leq SOC_t \leq SOC_{\text{max}}$ (typically 0.1–0.9 for Li-ion)
- **Current status**: This is historical V5 physics-branch context. In the current iGT/A1/B1 evidence chain, Battery SOC is an input/state token and placeholder diagnostic, not a learned output head. SOC must not be cited as learned component evidence for C13.

## Dimension Matching (Residual Shortcut)
- **Constraint**: Residual $r_i$ and theory $P_i^{\text{theory}}$ must have identical shape $(B, L_{\text{out}}, 1)$ for element-wise addition.
- **Enforcement**: Architectural — both theory branch and residual head output the same shape.
- **When dimensions change** (V4 scalar → V5 per-component): The residual dimension expands from 1 to 5, with each channel mapped to a specific component in the aggregation formula.

## Component Sign Convention
- **Constraint**: Load contributes positively (+) to net injection; PV and Wind contribute negatively (-); Battery can be either (+ charging, - discharging).
- **Enforcement**: Architectural — hardcoded in the aggregation formula.
- **Implication**: The model cannot learn a wrong sign convention. This is an invariant baked into the architecture.

## Non-Negativity of Generation
- **Constraint**: $P_{\text{pv}} \geq 0$, $P_{\text{wind}} \geq 0$
- **Enforcement**: None currently. Theory branches can theoretically output negative values if physics features are anomalous. In practice, ReLU activations in theory branches prevent this, but residual corrections could push predictions negative.
- **Risk**: Low — PV/Wind are far from zero in typical operating conditions. Only relevant at night (PV) or calm conditions (Wind).

## Gradient Isolation
- **Constraint**: Gradient from component loss should flow to the corresponding theory branch without cross-component interference.
- **Enforcement**: Per-component residual heads (V5) provide independent gradient paths. Scalar residual (V4) allows cross-contamination.
- **Remaining coupling**: The shared Transformer encoder couples all components — a gradient from Load component loss still affects representations used by PV theory branch. This is a fundamental trade-off between parameter efficiency and gradient isolation.

## Data Availability
- **Constraint**: Per-component ground truth is required for component supervision.
- **Current status**: Available for all 5 components (Load, PV, Wind, Battery Power, Battery SOC).
- **Implication**: The approach requires component-level metering. If only aggregate net injection is available, component supervision is not possible and the model must fall back to pure net MSE training (V3 mode).

## iGT Learned Component Boundary
- **Constraint**: Current PhysFormer-iGT decomposable evidence covers four learned output components: Load, PV, Wind, and Battery Power.
- **Enforcement**: Architectural/reporting. C13 and E13 tables exclude Battery SOC from learned component claims.
- **Implication**: Any dispatch-preparation statement can use R1-reg's four learned component forecasts, but SOC-aware battery feasibility remains unvalidated unless a future model learns SOC dynamics or E14 imports external SOC bounds.

## Dispatch Proxy Scope (E14)
- **Constraint**: Operational dispatch value cannot be inferred from component MAE alone.
- **Enforcement**: Evidence boundary. C13's dispatch-value part remains pending until E14 converts forecasts into adjustment commands and evaluates realized outcomes.
- **Minimum bounds for proxy-only validation**: shared component command limits, estimated or measured headroom, common cost weights, and a fixed target sequence chosen before scoring.
- **Pass boundary**: R1-reg component-aware allocation must beat A1 net-only allocation baselines on realized deviation or cost without increasing infeasible-command rate.
- **Implication**: If headroom, SOC limits, or cost curves are estimated from observed histories rather than measured asset metadata, E14 supports only a proxy operational claim, not field-ready dispatch optimization.
