# PhysFormer Architecture Updates (Logic Loop Closure)

> **AI Context/Prompt Inject**: This document details the core architectural modifications made to the `PhysFormer` codebase to achieve "logical loop closure" (逻辑闭环) and strict physical adherence. It intentionally omits numerical stabilization bugfixes (e.g., NaN patches, FP16/AMP mixed precision underflow fixes). When analyzing the codebase or writing academic sections (such as Experiments or Methodology), strictly adhere to the mechanistic designs described below.

## 1. Strict Physical Gating (Causal_coupling.py)
**Objective**: Guarantee that theoretical zero-power states (e.g., PV at night, Wind outside cut-in/cut-out speeds) are strictly enforced in the final physical feature fusion, without being compromised or diluted by neural network soft-gating.
*   **Mechanism**: The physical feature gates for PV and Wind are computed as a strict element-wise product of the hard physical prior and the learned soft gate: `gate = prior * soft_gate`.
*   **Key Modification**: Removed previous "floor" scalings (e.g., `0.5 + 0.5 * gate`) that artificially prevented the gate from reaching true zero. This ensures that when the physical prior dictates 0 output (e.g., irradiance is 0), the resulting `gate_pv` is exactly 0, blocking all irrelevant statistical features from contaminating the physics-guided stream.

## 2. Future Weather Injection / Residual Blind Spot Fix (model.py)
**Objective**: Provide the model's prediction heads with explicit awareness of future weather conditions, addressing the "blind spot" where historical statistical embeddings were forced to extrapolate physical dynamics without future meteorological context.
*   **Mechanism**: Implemented a Gated Linear Unit (GLU) fusion mechanism.
*   **Process**:
    1.  The future weather sequence `x_weather_future` is processed by `phys_layer` to extract future theoretical physics and `weather_feat_future`.
    2.  `future_feat_history` (the temporal projection of historical causal-coupled features) is concatenated with `weather_feat_future`.
    3.  A `weather_fusion_gate` (Sigmoid) and `weather_fusion_proj` compute the unified representation via a residual connection: `FusionGate * Proj(Concat) + HistoryFeat`.
*   **Result**: The deep decoupled prediction heads (Load/PV/Wind) now iteratively condition their temporal forecasting on accurate future meteorological context, significantly enhancing step-wise accuracy.

## 3. Kinematic Smoothing / Physical Inertia (model.py)
**Objective**: Eliminate high-frequency, non-physical jitter (noise) intrinsically introduced by standard Transformer point-wise mapping and attention mechanisms, natively imparting "physical inertia" to the raw residual generation process.
*   **Mechanism**: Applied a 1D Average Pooling (`avg_pool1d`) low-pass filter with `kernel_size=3` and `stride=1` operating over the temporal sequence dimension of the separated `res_load`, `res_pv`, and `res_wind` predictions.
*   **Key Detail**: Utilizes replication padding (`mode='replicate'`) on sequence boundaries to maintain the exact sequence length and preserve edge stability. This aligns the raw neural network temporal output with real-world thermodynamic and mechanical inertia before the final scale summation.

## 4. Bounded Physical Act-Residual / Zero-Bound Act (model.py)
**Objective**: Structurally and mathematically prevent the network from predicting physically impossible negative power generation/load values, replacing post-hoc ReLU/Clamp truncations that cause downstream gradient dead-zones.
*   **Mechanism**: A rigorous physics-informed lower-bound projection using `Softplus`.
*   **Process**:
    1.  Calculates the exact scalar representation of "0 MW" in the continuously normalized latent space: `zero_val = - (target_mean / target_std)`.
    2.  Computes the raw unbounded composite prediction: `raw_val = theory_future + activity_mask * res_pred`.
    3.  Applies the bounded activation envelope: `final_val = zero_val + Softplus(raw_val - zero_val)`.
*   **Result**: The final output function becomes strictly lower-bounded at the exact physical zero-point (0 MW). As the network prediction geometrically approaches 0 MW, the gradient smoothly decays rather than encountering a non-differentiable cliff, naturally eliminating Negative-Boundary Violations (BVR) directly within the architectural logic constraint.
