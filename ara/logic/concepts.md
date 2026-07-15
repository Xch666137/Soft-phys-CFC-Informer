# Concepts

## C01: Aggregated Net Power Injection
- **Notation**: $P_{\text{net}}(t) = P_{\text{load}}(t) - P_{\text{pv}}(t) - P_{\text{wind}}(t) + P_{\text{batt}}(t)$
- **Definition**: The net power exchanged between the VPP and the grid at time $t$. Positive = net consumption (load dominates); negative = net generation (DER dominates).
- **Boundary conditions**: $P_{\text{net}}$ is bounded by the VPP's grid connection capacity. $P_{\text{pv}} \geq 0$, $P_{\text{wind}} \geq 0$, $P_{\text{batt}}$ can be positive (charging) or negative (discharging).
- **Related concepts**: Power balance, VPP dispatch, aggregated forecasting

## C02: FiLM Conditioning (Feature-wise Linear Modulation)
- **Notation**: $\mathbf{h}' = \gamma(\mathbf{x}_{\text{phys}}) \odot \mathbf{h} + \beta(\mathbf{x}_{\text{phys}})$
- **Definition**: A conditioning mechanism where physics features $\mathbf{x}_{\text{phys}}$ are mapped through an MLP to produce channel-wise scale $\gamma$ and shift $\beta$ parameters that modulate the Transformer's hidden representations.
- **Boundary conditions**: $\gamma \in \mathbb{R}^{d_{\text{model}}}$, $\beta \in \mathbb{R}^{d_{\text{model}}}$. Applied after each self-attention sublayer, before residual add + layer norm.
- **Related concepts**: Adaptive conditioning, affine transformation, physics-informed neural networks

## C03: Theory Branch
- **Definition**: A per-component module that computes a physics-based estimate of that component's power output from meteorological and temporal features. PV branch uses irradiance×temperature model; Wind branch uses cubic wind-speed law; Battery branch uses SOC dynamics; Load branch uses calendar embeddings.
- **Boundary conditions**: Theory branches have no access to historical net injection data — they only see physics features. This forces them to learn physics rather than memorize patterns.
- **Related concepts**: Physics prior, inductive bias, domain knowledge embedding

## C04: Component-Consistent Residual
- **Notation**: $r_i(t)$ for component $i \in \{\text{load}, \text{pv}, \text{wind}, \text{batt}\}$, total residual $\mathbf{r} \in \mathbb{R}^{L_{\text{out}} \times 4}$
- **Definition**: Per-component learnable correction added to each theory branch output before aggregation. Unlike a scalar residual (which corrects only the aggregate), component-consistent residuals independently correct each component, preventing cross-component error propagation.
- **Boundary conditions**: Each $r_i$ is produced by an independent MLP head. The aggregate power balance is preserved: $P_{\text{net}} = \sum_i s_i \cdot (P_i^{\text{theory}} + r_i)$ where $s_i = \pm 1$ per sign convention.
- **Related concepts**: Residual learning, disentangled representation, per-component correction

## C05: Theory Deviation (Theory MAE)
- **Notation**: $\text{TheoryMAE} = \frac{1}{N} \sum |P_{\text{theory}} - P_{\text{true}}|$
- **Definition**: Mean absolute error between the theory branch output (physics-only prediction) and the true component value. Measures how much correction the residual head must provide.
- **Boundary conditions**: Theory MAE $\geq 0$. Lower is better (physics is more accurate). A high Theory MAE means the residual head must do most of the work.
- **Related concepts**: Physics-model accuracy, residual burden, forecast error decomposition

## C06: Curriculum Training (Physics-to-Data Annealing)
- **Notation**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{net\_mse}} + \lambda(\text{epoch}) \cdot \mathcal{L}_{\text{component\_mae}}$
- **Definition**: A training schedule where the component loss weight $\lambda$ starts high (physics emphasis), then linearly decays to a lower value (data emphasis). This establishes a physics-informed initialization before allowing the model to override physics with data-driven corrections.
- **Boundary conditions**: $\lambda \in [\lambda_{\text{min}}, \lambda_{\text{max}}]$, typically $\lambda_{\text{max}} = 0.03\text{--}0.10$, $\lambda_{\text{min}} = 0.001\text{--}0.01$. The decay schedule is piecewise linear over epoch ranges.
- **Related concepts**: Training curriculum, loss annealing, multi-task learning, Pareto optimization

## C07: Component Loss Weight Pareto Frontier
- **Notation**: $\mathcal{P} = \{(\text{TheoryMAE}(\lambda), \text{TestMAE}(\lambda)) \mid \lambda \in [0, \lambda_{\text{max}}]\}$
- **Definition**: The set of non-dominated (Theory MAE, Test MAE) pairs achievable by varying the component loss weight $\lambda$. Points on the frontier represent optimal trade-offs — improving one metric requires degrading the other.
- **Boundary conditions**: The frontier is empirically observed, not theoretically derived. Its shape depends on architecture, data, and training duration.
- **Related concepts**: Multi-objective optimization, trade-off analysis, hyperparameter sensitivity

## C08: Cross-Component Error Contamination
- **Definition**: When a scalar residual corrects the aggregate net injection, an error in one component's prediction can be compensated by adjusting the residual in a way that corrupts other components' effective predictions, without the loss function being aware of the contamination.
- **Boundary conditions**: Occurs only with scalar residual (V3, V4). Component-consistent residual (V5) eliminates this by giving each component its own correction pathway.
- **Related concepts**: Gradient interference, disentangled optimization, component-level supervision

## C09: Shared-Encoder Cancellation Channel
- **Notation**: $e_{\text{net}} = e_{\text{load}} - e_{\text{pv}} - e_{\text{wind}} + e_{\text{batt}}$
- **Definition**: A shared temporal encoder can learn cross-component error correlations that reduce aggregate net error through signed cancellation, even when individual component forecasts are inaccurate. This explains why a model can improve aggregate MAE while degrading component MAE.
- **Boundary conditions**: Applies when multiple DER components share a representation space and the training objective mainly supervises aggregate net injection. It is diagnosed through component-error covariance rather than aggregate error alone.
- **Related concepts**: Component-aggregate paradox, covariance cancellation, C08, C10

## C10: Component-Token Separation
- **Notation**: $T = [T_{\text{load}}, T_{\text{pv}}, T_{\text{wind}}, T_{\text{batt}}, T_{\text{soc}}, T_{\text{temp}}, T_{\text{irr}}, T_{\text{windspd}}]$
- **Definition**: An inverted-token formulation where each DER component and weather variable is encoded as a semantic token before self-attention. Attention operates across variables rather than across time positions.
- **Boundary conditions**: In the current A1 contract, five component-history tokens and three future-weather tokens form an 8-token encoder input. Fixed physics tokens, graph bias, twin tokens, and horizon cross-attention are excluded from the mainline because C12 shows they overfit.
- **Related concepts**: Inverted attention, iTransformer, component-token forecasting, cancellation-channel removal

## C11: Masked Component Pretraining (MCP)
- **Notation**: $\mathcal{L}_{\text{MCP}} = \mathcal{L}_{\text{masked-comp-mae}} + \lambda_{\text{net}}\mathcal{L}_{\text{net-mse}}$
- **Definition**: A self-supervised pretraining objective that masks selected component histories and trains the model to reconstruct future component trajectories while retaining a net-MSE anchor. It keeps the A1 architecture unchanged and changes only the training signal.
- **Boundary conditions**: The repaired B1 protocol uses the same 8-token contract for pretraining, finetuning, and testing; missing pretrained checkpoints are fatal; Battery SOC is not treated as a learned output component.
- **Related concepts**: Self-supervised learning, component decomposability, B1, R1-reg

## C12: Decomposable Forecasting vs. Dispatchability
- **Definition**: Decomposable forecasting means the model outputs learned Load/PV/Wind/Battery-Power trajectories that can be aggregated by the real-unit power-balance identity. Dispatchability, in the operational sense, requires a further step: converting forecasts into feasible DER adjustment commands and measuring realized deviation, cost, and infeasibility.
- **Boundary conditions**: C13 currently supports decomposable forecasting for dispatch preparation. It does not prove dispatch cost reduction or field-ready optimizer performance until E14 or a stronger optimizer validation is executed.
- **Related concepts**: Dispatch proxy, Pareto frontier, aggregate-vs-component tradeoff, E14
