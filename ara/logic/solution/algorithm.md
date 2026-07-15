# Algorithm

## A1 PhysFormer-iGT Forward Pass

### Inputs

Let:

- `X_comp in R^{B x L_in x 5}` be component history:
  `[Load, PV, Wind, BatteryPower, BatterySOC]`.
- `X_weather in R^{B x L_out x 3}` be future weather:
  `[temperature, irradiance, wind_speed]`.
- `L_out = 96` be the prediction horizon.

### Component Tokenization

All component histories are encoded in one batched GRU call:

```text
X_comp: (B, L_in, 5)
transpose/reshape -> (B * 5, L_in, 1)
BiGRU -> project -> T_comp: (B, 5, d_model)
```

All weather futures are encoded in one batched MLP call:

```text
X_weather: (B, L_out, 3)
transpose/reshape -> (B * 3, L_out)
MLP -> T_weather: (B, 3, d_model)
```

The A1 token set is:

```text
T = concat(T_comp, T_weather)  # (B, 8, d_model)
```

### Inverted Self-Attention

The encoder applies self-attention across the 8 semantic tokens:

```text
Z = InvertedEncoder(T)
```

There are no physics tokens, graph-bias terms, twin tokens, constraint tokens, or
horizon cross-attention decoder in the A1 mainline.

### Component Prediction

Four independent FFN projectors decode the four learned components:

```text
Y_load_norm = FFN_load(Z_load)
Y_pv_norm   = FFN_pv(Z_pv)
Y_wind_norm = FFN_wind(Z_wind)
Y_batt_norm = FFN_batt(Z_batt_power)
```

These normalized predictions are denormalized into MW with the auxiliary scaler:

```text
Y_i_real = Y_i_norm * aux_std_i + aux_mean_i
```

Battery SOC is not decoded as a learned dispatch component in the current evidence chain.

### Power-Balance Aggregation

The predicted net injection is computed by the hard sign convention:

```text
Y_net_real = Y_load_real - Y_pv_real - Y_wind_real + Y_batt_real
```

For training against normalized net targets, `Y_net_real` is mapped through the target
scaler into normalized target space.

### A1 Loss

A1 is trained from scratch with net MSE only:

```text
L_A1 = MSE(Y_net_norm, Y_net_true_norm)
```

This is the aggregate-accuracy baseline for C11-C13.

## B1 Masked Component Pretraining

B1 keeps the same forward architecture and adds component masking before tokenization.

For a mask `M` over the first four components:

```text
X_comp_masked[..., i] = mask_token, if M_i = 1
X_comp_masked[..., i] = X_comp[..., i], otherwise
```

The model predicts all four learned components, but component loss is applied only to
masked components:

```text
L_comp = mean_i,t |Y_i_norm - Y_i_true_norm| over masked i
L_net  = MSE(Y_net_norm, Y_net_true_norm)
L_B1_pretrain = L_comp + lambda_net * L_net
```

The repaired mainline uses `lambda_net=1.0` and the N131 protocol:

- canonical checkpoint path;
- fatal failure on missing pretrained checkpoint;
- no pretrain-only calendar token;
- `use_compile=false`;
- iGT SOC metric treated as placeholder.

## Downstream Finetuning

### R0 Direct Test

Load the repaired MCP checkpoint and evaluate without additional training.

### R1 Low-LR Finetune

Finetune with net MSE only:

```text
L_R1 = MSE(Y_net_norm, Y_net_true_norm)
```

### R1-Reg Tiny Component Anchor

Finetune with net MSE plus a small component anchor:

```text
L_R1_reg = MSE(Y_net_norm, Y_net_true_norm)
           + epsilon * mean_component_MAE(Y_comp_norm, Y_comp_true_norm)
```

N132 identifies R1-reg as the best repaired B1 downstream arm by aggregate MAE while
also producing the strongest learned 4-component metrics among repaired B1 arms.

### R2 Few-Shot Adaptation

Use a target-prefix subset for adaptation. N132 shows that the tested 5%, 10%, and 20%
few-shot settings degrade aggregate performance, so R2 is not the current mainline.

## Dispatch Proxy Algorithm (E14, Pending)

The dispatch proxy is not yet evidence. It is the proposed way to test the operational
part of C13.

### Inputs

- A1 net prediction and optional component diagnostics.
- R1-reg component predictions.
- Ground-truth test-set components for realized-outcome evaluation.
- A fixed target sequence such as peak-shaving, ramp-smoothing, or reserve tracking.
- Shared component bounds and cost weights.

### Net-Only Baseline

A1 provides a scalar net forecast. The required adjustment is allocated by a simple rule:

```text
Delta_i = w_i * Delta_net
```

where `w_i` is either historical component share or uniform share, clipped by the same
component bounds used for all arms.

### Component-Aware Arm

R1-reg allocates `Delta_net` according to predicted component availability/headroom:

```text
Delta_i = allocate(Delta_net, predicted_headroom_i, cost_i, bounds_i)
```

### Evaluation

Use ground truth, not predictions, to score the realized schedule:

```text
realized_net_error = target_net - realized_net_after_dispatch
total_cost = sum_i cost_i * |Delta_i|
infeasible_rate = count(bound violations) / count(commands)
```

Pass condition for C13 operational validation:

- R1-reg beats A1 net-only baselines on realized net deviation or cost;
- it does not increase infeasible-command rate;
- the result holds in at least two dispatch target scenarios, or the claim is scoped to
  the single tested scenario.
