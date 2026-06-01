# Training Configuration

## V4 Training Recipe (Best Aggregate)

| Parameter | Value | Rationale | Search Range | Sensitivity |
|-----------|-------|-----------|-------------|-------------|
| Optimizer | AdamW | Standard for Transformer training | — | low |
| Learning rate | 1e-4 | Standard starting point | [5e-5, 5e-4] | medium |
| LR schedule | CosineAnnealing | Smooth decay without plateau stalls | — | low |
| Batch size | 32 | GPU memory limit on 5090 | [16, 64] | low |
| Epochs | 70 | Saturation observed by epoch 50-60 | [50, 100] | low |
| Weight decay | 1e-4 | Standard regularization | [1e-5, 1e-3] | low |
| Gradient clip | 1.0 | Prevent occasional gradient spikes | [0.5, 5.0] | low |
| Component loss weight | 0.05 | Empirically optimal for V4 (scalar residual) | [0.01, 0.20] | **high** |
| Component loss type | MAE (L1) | Linear penalty in kW — prevents Load dominance | {MAE, MSE, Huber} | medium |
| Warmup epochs | 3 | Linear LR warmup from 1e-7 | [0, 10] | low |
| Early stopping patience | 10 | Prevent overfitting | [5, 20] | medium |
| Seed | 42 or 3407 | Reproducibility | — | low |

## V5 Training Recipe (Component-Consistent Residual + Curriculum)

| Parameter | Phase 1 (Physics Warmup) | Phase 2 (Joint Annealing) | Phase 3 (Net MSE Fine-tune) |
|-----------|--------------------------|---------------------------|----------------------------|
| Epochs | 1–15 | 16–40 | 41–70 |
| Component loss weight | 0.10 | 0.10 → 0.005 (linear decay) | 0 (pure net_mse) |
| Residual grad to theory | Restricted | Full | Full |
| LR | 1e-4 | CosineAnnealing from 1e-4 | 1e-5 |
| Phase 3 status | — | — | **Dead end** (no validation improvement) |

## V5.5 Tuning Adjustments (In Progress)

| Parameter | V5 | V5.5 | Rationale |
|-----------|-----|------|-----------|
| Phase 1 component weight | 0.10 | 0.03 | Per-component residual multiplies effective supervision 5×; 0.03 ≈ V4's 0.05/5× |
| Phase 1 epochs | 15 | 5–8 | Faster physics initialization sufficient |
| Phase 3 | 30 epochs | Removed | Dead end — no validation gain (N12) |
| Residual init std | 0.01 | 0.05 | Faster residual learning in Phase 1 |
