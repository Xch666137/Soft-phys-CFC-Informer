# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhysFormer — a physics-guided Transformer for Virtual Power Plant (VPP) multi-target load forecasting. Predicts Load, PV Power, and Wind Power 24 hours ahead (96 steps at 15-min resolution) from 7 days of history (672 steps). The core innovation is structurally embedding physical constraints (solar/wind bounds, ramp rates) into the architecture rather than using post-hoc clamping.

## Commands

### Training
```bash
# PhysFormer (main model) via unified entry point
python run.py --config configs/physformer_default.yaml

# With CLI overrides
python run.py --config configs/physformer_default.yaml --lr 1e-4 --epochs 100

# Ablation experiments
python run.py --config configs/physformer_ablation_v1.yaml   # V1: no physics stream
python run.py --config configs/physformer_ablation_v2.yaml   # V2: naive concat fusion
python run.py --config configs/physformer_ablation_v3.yaml   # V3: no future weather GLU
python run.py --config configs/physformer_ablation_v4.yaml   # V4: no curriculum learning
python run.py --config configs/physformer_ablation_v5.yaml   # V5: freeze physics params

# Test only (skip training)
python run.py --config configs/physformer_default.yaml --test_only

# Baselines
python run.py --config configs/baselines/informer.yaml
python run.py --config configs/baselines/lstm.yaml

# Legacy scripts (still functional)
python scripts/run_PhysFormer.py
python scripts/run_benchmark.py
```

### Evaluation & Visualization
```bash
python analysis/generate_paper_results.py        # Official benchmark metrics table
python analysis/evaluate_extreme_weather.py       # Extreme weather scenario testing
python analysis/plot_gate_correlation.py          # Causal gate correlation visualization
python analysis/compute_gate_corr.py              # Gate-weather correlation analysis
```

### Setup
```bash
pip install -e .              # Install as editable package (recommended)
pip install -r requirements.txt  # Or install dependencies only
```

## Architecture

### Core Pipeline

```
run.py  →  physformer/exp/exp_physformer.py  →  physformer/models/physformer.py
(config)        (train/val/test loop)              (forward pass)
```

Data flows through `physformer/data/data_factory.py` (`PhysFormerDataset`), which loads `data/vpp_dataset_3years.csv`, applies StandardScaler normalization, and produces sin/cos time encodings.

### PhysFormer Model Components (`physformer/models/`)

- **`physformer.py`** — Main model. Dual-stream architecture: statistical Transformer encoder + physical stream, fused via PGCC, with CFC temporal smoothing and bounded output heads.
- **`physical_layer.py`** — `ExplicitPhysicalMapping`. Computes physics baselines using learnable parameters: PV efficiency/temperature coefficient, wind power curve thresholds, load base/temperature sensitivity. 7 total learnable physical parameters.
- **`causal_coupling.py`** — `PhysicsGuidedCausalCoupling` (PGCC). Multi-head cross-attention between statistical queries and physical keys/values. Learns per-target soft gates (0–1) with curriculum scheduling. Gate = hard_prior × learned_soft_gate.
- **`cfc.py`** — Continuous Function-based RNN (ODE layer via torchdiffeq). Models physical inertia for temporal smoothing of residual predictions.
- **`flatten_head.py`** — Efficient projection head: `[B, S, D] → [B, P, D]` via linear time-dimension projection (~64k params).

### Shared Layers (`physformer/layers/`)

- **`attention.py`** — ProbAttention, FullAttention (with RoPE support)
- **`embedding.py`** — DataEmbedding, TokenEmbedding, TemporalEmbedding
- **`encoder.py`** — Encoder, EncoderLayer, FeedForward, AttentionLayer
- **`decoder.py`** — Decoder, DecoderLayer
- **`positional.py`** — PositionalEncoding, RoPEPositionalEncoding
- **`revin.py`** — RevIN (Reversible Instance Normalization)

### Key Design Patterns

- **BPAR (Bounded Physical Act-Residual)**: `output = zero_val + Softplus(raw - zero_val)` — structurally prevents negative power predictions without gradient-killing clamping.
- **Curriculum learning**: Soft gates start at 0 (pure physics) and linearly increase per epoch, gradually allowing the network to override physical priors. This is the single most impactful component (8.7% accuracy impact in ablation).
- **Activity masking**: Prevents learning spurious patterns during zero-output periods (nighttime PV, calm wind).
- **Channel-aware NRMSE**: Used for early stopping to prevent collapse toward large-scale channels.

### Loss & Metrics (`physformer/utils/`)

- **`losses.py`** — `PhysLoss`: prediction error + boundary violation penalty (L_BVR) + ramp rate violation penalty (L_RVR) + `GateResponseRegularization` (Pearson correlation between gates and physical priors).
- **`metrics.py`** — Standard (MAE, RMSE, RSE, CORR) + physics-specific (BVR = boundary violation rate, MVS = mean violation size, `PhysicsComplianceMetrics`).

### Baseline Models (`physformer/models/`)

Informer, Autoformer, LSTM, GRU, PINN, DLinear, PatchTST, iTransformer — all trained/evaluated via `physformer/exp/exp_baseline.py` (`Exp_Baselines`, inherits from `Exp_PhysFormer`).

### Default Hyperparameters

seq_len=672, pred_len=96, d_model=512, n_heads=8, e_layers=3, d_ff=2048, batch_size=128, lr=3e-4, dropout=0.10, use_rope=True, use_amp=True, patience=10. All defaults stored in `configs/physformer_default.yaml`.

## Key Conventions

- Install as a package with `pip install -e .` — no `sys.path` hacks needed.
- YAML configs in `configs/` provide defaults; CLI args override them.
- Dataset: 6 input features (Load, PV, Wind, Temp, Irradiance, WindSpeed), 3 targets (Load, PV, Wind), sequential train/val/test split (no shuffling).
- Checkpoints saved to `checkpoints/`.
- Figures/plots saved to `visualization/output/`.
- Analysis scripts live in `analysis/`.
- Comments and argparse help strings are a mix of Chinese and English.
- Paper materials live in `paper/en/` (English) and `paper/zh/` (Chinese).
