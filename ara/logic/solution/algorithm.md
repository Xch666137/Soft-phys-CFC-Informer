# Algorithm

## Core Formulation

### Net Power Decomposition

The VPP aggregated net power at time $t$ is:

$$P_{\text{net}}(t) = P_{\text{load}}(t) - P_{\text{pv}}(t) - P_{\text{wind}}(t) + P_{\text{batt}}(t)$$

PhysFormer predicts each component as theory + residual:

$$P_i(t) = P_i^{\text{theory}}(t) + r_i(t), \quad i \in \{\text{load}, \text{pv}, \text{wind}, \text{batt}\}$$

The residual $r_i$ is a learned correction to the physics-based theory estimate.

### Theory Branch: PV

$$P_{\text{pv}}^{\text{theory}}(t) = \eta \cdot G(t) \cdot [1 + \alpha (T(t) - T_{\text{ref}})] \cdot N_{\text{panels}}$$

Where $G(t)$ = irradiance (W/m²), $T(t)$ = module temperature, $\eta$ = panel efficiency, $\alpha$ = temperature coefficient.

### Theory Branch: Wind

$$P_{\text{wind}}^{\text{theory}}(t) = f_{\theta}(v(t))$$

Where $v(t)$ = wind speed at hub height, $f_{\theta}$ is a learnable cubic-like function parameterized by a small MLP.

### Theory Branch: Battery

$$SOC_t = SOC_{t-1} + \eta_{\text{ch}} \cdot P_{\text{ch},t} \cdot \Delta t - \frac{1}{\eta_{\text{dis}}} \cdot P_{\text{dis},t} \cdot \Delta t$$

$$P_{\text{batt}}^{\text{theory}}(t) = P_{\text{ch},t} - P_{\text{dis},t}$$

Subject to: $SOC_{\text{min}} \leq SOC_t \leq SOC_{\text{max}}$, $0 \leq P_{\text{ch}} \leq P_{\text{rated}}$, $0 \leq P_{\text{dis}} \leq P_{\text{rated}}$.

### Theory Branch: Load

$$P_{\text{load}}^{\text{theory}}(t) = g_{\phi}(\text{calendar}(t))$$

Where $\text{calendar}(t)$ = [hour, weekday, month, holiday_flag], $g_{\phi}$ is a learned temporal embedding + MLP.

### FiLM Conditioning

For encoder layer $l$, hidden state $\mathbf{h}^{(l)}$:

$$\gamma^{(l)}, \beta^{(l)} = \text{MLP}_{\text{FiLM}}(\mathbf{x}_{\text{phys}})$$

$$\mathbf{h}^{(l)} = \gamma^{(l)} \odot \mathbf{h}^{(l)} + \beta^{(l)}$$

Applied after self-attention, before residual add.

### Residual Heads

For component $i$, the residual at timestep $\tau$ (in the prediction horizon):

$$r_i(\tau) = \text{MLP}_{\text{res},i}(\mathbf{d}_\tau)$$

Where $\mathbf{d}_\tau$ is the decoder output at step $\tau$, time-conditioned by $\text{time\_proj}(\mathbf{y}_{\text{mark},\tau})$.

### Loss Function

$$\mathcal{L} = \underbrace{\frac{1}{T}\sum_{t}(P_{\text{net}}(t) - \hat{P}_{\text{net}}(t))^2}_{\text{Aggregate MSE}} + \lambda \cdot \underbrace{\frac{1}{T}\sum_{t}\sum_{i}|P_i(t) - \hat{P}_i(t)|}_{\text{Component MAE}}$$

With curriculum schedule:

$$\lambda(\text{epoch}) = \begin{cases} \lambda_{\text{max}} & \text{epoch} < E_1 \\ \lambda_{\text{max}} - (\lambda_{\text{max}} - \lambda_{\text{min}}) \cdot \frac{\text{epoch} - E_1}{E_2 - E_1} & E_1 \leq \text{epoch} < E_2 \\ \lambda_{\text{min}} & \text{epoch} \geq E_2 \end{cases}$$

## Pseudocode

```
def physformer_forward(x_net, x_phys, y_mark):
    # Encode
    h = encoder(x_net, x_phys, y_mark[:L_in])  # FiLM-conditioned Transformer

    # Theory branches (physics-only, no gradient to encoder for theory computation)
    pv_theory = pv_branch(x_phys)       # irradiance × temp model
    wind_theory = wind_branch(x_phys)    # cubic wind speed fit
    batt_theory = batt_branch(x_phys)    # SOC dynamics
    load_theory = load_branch(y_mark)    # calendar embeddings

    # Decode with time conditioning
    d = temporal_decoder(h, time_proj(y_mark[L_in:]))

    # Per-component residuals
    pv_res = pv_residual_head(d)
    wind_res = wind_residual_head(d)
    batt_res = batt_residual_head(d)
    load_res = load_residual_head(d)

    # Component predictions
    pv_pred = pv_theory + pv_res
    wind_pred = wind_theory + wind_res
    batt_pred = batt_theory + batt_res
    load_pred = load_theory + load_res

    # Aggregate (power balance)
    net_pred = load_pred - pv_pred - wind_pred + batt_pred

    return net_pred, (load_pred, pv_pred, wind_pred, batt_pred)

def compute_loss(net_pred, net_true, comp_preds, comp_trues, epoch):
    loss_net = MSE(net_pred, net_true)
    loss_comp = sum(MAE(pred, true) for pred, true in zip(comp_preds, comp_trues))
    lambda = curriculum_schedule(epoch)
    return loss_net + lambda * loss_comp
```

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Self-attention (encoder) | $O(L_{\text{in}}^2 \cdot d)$ | $O(L_{\text{in}}^2)$ |
| FiLM conditioning | $O(L_{\text{in}} \cdot d \cdot d_{\text{phys}})$ | $O(d)$ |
| Theory branches | $O(L_{\text{out}} \cdot d_{\text{theory}})$ | $O(d_{\text{theory}})$ |
| Temporal decoder | $O(L_{\text{out}}^2 \cdot d)$ | $O(L_{\text{out}}^2)$ |
| Residual heads | $O(L_{\text{out}} \cdot d \cdot 5)$ | $O(d \cdot 5)$ |
| **Total (V5)** | Dominated by attention: $O((L_{\text{in}}^2 + L_{\text{out}}^2) \cdot d)$ | ~3.5M params |
