---
name: python-best-practices
description: Enforce high-quality Python coding standards specially for PyTorch, VPP simulation, and Time-series forecasting.
---

# Python & PyTorch Research Best Practices

When writing or refactoring code for this user (Electrical Engineering Researcher), YOU MUST follow these rules:

## 1. PyTorch & Model Architecture
- **Tensor Shapes:** ALWAYS comment the expected shape of tensors at each layer transformation. 
  - e.g., `# x: [batch_size, seq_len, embed_dim]`
- **Device Agnostic:** Use `.to(device)` logic. Never hardcode `.cuda()` or `.cpu()`.
- **Reproducibility:** Always include a `seed_everything()` function that sets seeds for torch, numpy, and random.

## 2. Scientific Computing (Pandapower/Simbench)
- **Vectorization:** Avoid `for` loops when processing time-series data. Use `numpy` or `torch` vector operations.
- **Units:** Explicitly state units in variable names or docstrings (e.g., `p_mw`, `q_mvar`, `voltage_kv`).

## 3. Code Structure
- **Type Hints:** Use `typing.List`, `typing.Dict`, `torch.Tensor` strictly.
- **Config Management:** Do not hardcode hyperparameters inside classes. Use `argparse` or a configuration class/dict.

## 4. Debugging & Logging
- For RL (DDPG) training loops, ensure loss values and Q-values are logged to TensorBoard or a CSV, not just printed.