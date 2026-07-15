# Architecture

## Current Mainline: PhysFormer-iGT

PhysFormer-iGT is the current thesis mainline. It replaces the historical
shared-encoder FiLM PhysFormer with an inverted-token Transformer for DVPP net
forecasting and decomposable component prediction.

The central design choice is component-token separation: each DER component has
its own token before attention. This removes the shared-encoder cancellation
channel documented in C08-C10, while preserving the real-unit power-balance
decoder.

## A1 Aggregate Forecaster

### Inputs

- **Component history**: `(B, L_in, 5)` for Load, PV, Wind, Battery Power, Battery SOC.
- **Future weather**: `(B, L_out, 3)` for temperature, irradiance, and wind speed.
- **Time marks / portfolio metadata**: available in the data pipeline, but A1 does not add
  fixed physics tokens, graph priors, or a separate horizon decoder.

### Tokenization

- **5 component tokens**: a single batched one-layer bidirectional GRU encodes all five
  component histories as `(B, 5, d_model)`.
- **3 weather tokens**: a batched MLP encodes the three future weather channels as
  `(B, 3, d_model)`.
- **8-token contract**: component tokens and weather tokens are concatenated into
  `(B, 8, d_model)`.

This 8-token contract is preserved across pretraining, finetuning, and testing after
the N131 repair. The token-encoder enhancement path is closed by N135: the A0 BiGRU
readout fix and hidden=128 capacity expansion did not beat the A1 baseline.

### Inverted Attention Encoder

The encoder applies self-attention across tokens, not across time steps. Each token is a
domain-semantic variable representation rather than a time-position representation.

This is the architectural mechanism behind C11:

- it prevents the shared temporal encoder from mixing all components into one latent stream;
- it blocks the encoder-depth -> cancellable-covariance channel identified in C10;
- it avoids fixed physics priors, which C12 shows to be overfitting amplifiers.

### Component Decoders

A1 uses independent FFN projectors for the four learned dispatch-relevant components:

- Load
- PV
- Wind
- Battery Power

Battery SOC is not a learned output in the current iGT evidence chain. It is excluded
from the learned decomposable-forecasting claim and should be treated as placeholder
diagnostics only.

### Real-Unit Power-Balance Decoder

Predicted components are denormalized into MW and aggregated by the sign convention:

```text
P_net = P_load - P_pv - P_wind + P_batt
```

This gives A1 a hard architectural power-balance identity without using explicit PV,
wind, or battery physics equations as model inputs.

## Phase B: Masked Component Pretraining

B1 keeps the same A1 8-token architecture and changes only the training signal.

### MCP Pretraining

- Randomly mask one or more component history channels among Load/PV/Wind/Battery Power.
- Replace masked channels with a learnable mask token before GRU tokenization.
- Predict masked future components through component MAE in auxiliary-scaler space.
- Add a net-MSE anchor with `lambda_net=1.0` so the checkpoint remains compatible with
  aggregate forecasting.

### Downstream Arms

- **R0**: direct test of the repaired pretrained checkpoint.
- **R1**: low-LR net-MSE finetuning.
- **R1-reg**: low-LR finetuning with a tiny component anchor; currently the best repaired
  B1 downstream arm in N132.
- **R2**: few-shot target-prefix adaptation; N132 shows this degrades aggregate MAE.

## Historical Architecture Context

The earlier PhysFormer line used:

- shared Transformer encoder;
- FiLM-conditioned physics branches;
- theory + residual decomposition;
- component-consistent residual heads;
- component-loss curriculum.

Those designs remain important historical evidence for C01-C07, but they are no longer
the current thesis mainline. The final narrative uses them as a research path that
revealed the shared-encoder cancellation problem and motivated PhysFormer-iGT.

## Dispatch Proxy Layer

C13 does not claim that the model is already a dispatch optimizer. It claims that
component-decomposable forecasts are the information layer needed for dispatch preparation.

Operational dispatch value must be tested by E14:

- compare R1-reg component-aware allocation against A1 net-only allocation;
- evaluate realized net deviation, infeasible-command rate, and dispatch cost under the
  same constraints;
- treat any result as proxy-only unless real asset headroom, SOC bounds, and cost curves
  are available.
