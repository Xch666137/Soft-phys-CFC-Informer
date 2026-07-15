---
title: "PhysFormer: Physics-Guided Transformer for VPP Aggregated Net Power Forecasting"
authors: ["Xingyu Liu"]
year: 2026
venue: "Master's Thesis, Shanghai University of Electric Power"
doi: "TBD"
ara_version: "0.6"
domain: "power systems / physics-guided deep learning"
keywords:
  [
    "virtual power plant",
    "net power forecasting",
    "physics-guided learning",
    "transformer",
    "component-consistent residual",
    "curriculum training",
    "FiLM conditioning",
    "distributed energy resources",
  ]
claims_summary:
  - "Physics-guided FiLM conditioning with per-component theory branches improves physical consistency of VPP net load forecasts over black-box Transformers."
  - "Component-consistent residual learning disentangles per-asset-type error propagation, reducing PV/Wind/Battery component errors while preserving aggregate accuracy."
  - "Curriculum training with moderate component-loss annealing improves robustness to loss-weight choice, but does not by itself prove a Phase1-to-Phase2 refinement mechanism."
  - "Load forecasting is the dominant error bottleneck: behavioral modeling reduced the initial 18x Load/Wind gap to ~6x, but residual load error remains larger than weather-driven components."
  - "Selective gradient detach produces robust aggregate-accuracy gains across seeds with the smallest cross-seed variance (C09); mechanism = disabling encoder-depth→cancellation channel (C10)."
  - "Component-token separation via inverted attention eliminates the shared-encoder cancellation channel, achieving −12.5% aggregate MAE vs PhysFormer c23 baseline with 20× smaller cross-seed variance (C11)."
  - "Fixed physics priors (tokens, graph bias, weather conditioning, horizon decoder) are monotonic overfitting amplifiers — every architectural addition beyond component-token separation systematically degrades Test MAE (C12)."
  - "DVPP dispatch preparation requires per-component decomposability; pure net injection accuracy is a means, not an end. Repaired MCP pretrain+finetune trades about +1% aggregate MAE for learned Load/PV/Wind/Battery-Power forecasts, while operational dispatch value remains pending E14 (C13)."
abstract: "Distributed Virtual Power Plant (DVPP) aggregated net power forecasting is critical for dispatch and ancillary market participation, yet pure end-to-end deep learning approaches collapse 4-dimensional component information (load, PV, wind, battery) into a scalar net injection prediction that cannot answer the operator's fundamental question: 'which DER should I adjust to meet the grid requirement?' We propose PhysFormer-iGT, an inverted Transformer that tokenizes each component independently through self-attention across 8 tokens (5 components + 3 weather), eliminating the shared-encoder cross-component error cancellation channel that masks individual component inaccuracy in aggregate metrics. Phase A establishes that component-token separation achieves -12.5% aggregate MAE vs shared-encoder baselines with 20x smaller variance, while fixed physics priors monotonically degrade Test MAE. Phase B introduces Masked Component Pretraining (MCP) to learn data-driven component coupling from partially observed DER histories, then finetunes on net injection to obtain decomposable Load/PV/Wind/Battery-Power forecasts at about +1% aggregate MAE relative to A1. The contribution is twofold: (1) architectural proof that component-token separation outperforms physics guidance for VPP forecasting (Phase A, validated across 3 seeds for each claim); (2) a self-supervised pretraining paradigm that preserves component-level information needed for downstream DVPP dispatch preparation. The actual dispatch cost or feasibility benefit is hypothesized, not yet proven, and is assigned to the pending E14 proxy validation."
status: "in-progress; B1/R1-reg component metrics obtained; C13 scoped to decomposable forecasting, with dispatch optimizer value pending E14"
phase: "Phase B rescue (2026-06-10). N135 (2026-06-14): Token encoder exploration closed: A0 (BiGRU readout fix, 3-seed mean MAE=0.001824) and A1 (hidden=128, MAE=0.001845) both FAILED to beat A1 baseline 0.001811. Bottleneck is NOT temporal encoding; iGT gain comes from component-token separation (C11 strengthened by exclusion). Thesis mainline: A1 = aggregate model, B1/R1-reg = decomposable forecasting tradeoff (C13), E14 = dispatch proxy validation."
---
# PhysFormer: Physics-Guided Transformer for VPP Aggregated Net Power Forecasting

## Overview

PhysFormer-iGT addresses DVPP aggregated net power forecasting by replacing the
historical shared-encoder physics-guided Transformer with an inverted-token
architecture. The current mainline tokenizes Load, PV, Wind, Battery Power, Battery
SOC, and future weather as semantic tokens, applies self-attention across tokens, and
aggregates learned Load/PV/Wind/Battery-Power predictions through the real-unit power
balance identity. This removes the shared-encoder cancellation channel behind the
component/aggregate paradox (C08-C11) without adding fixed physics priors, which Phase A
showed to be overfitting amplifiers (C12).

This artifact captures both the historical PhysFormer path (FiLM theory branches,
component-consistent residuals, curriculum training, and their dead ends) and the current
PhysFormer-iGT path. Phase B adds Masked Component Pretraining (MCP) to recover
component-level decomposability at a small aggregate-MAE cost, with direct operational
dispatch value left to the pending E14 proxy validation.

## Layer Index

### Cognitive Layer (`/logic`)
| File | Description |
|------|-------------|
| [problem.md](logic/problem.md) | Shared-encoder cancellation, fixed-prior overfitting, and component-token separation gap framing |
| [claims.md](logic/claims.md) | Falsifiable claims from historical PhysFormer through C13 decomposable forecasting |
| [concepts.md](logic/concepts.md) | Formal definitions for net power, historical physics guidance, and component-token forecasting |
| [experiments.md](logic/experiments.md) | Declarative experiments E01-E14, including Phase A/B and pending dispatch proxy |
| [solution/architecture.md](logic/solution/architecture.md) | Current PhysFormer-iGT architecture, MCP pretraining, and dispatch proxy layer |
| [solution/algorithm.md](logic/solution/algorithm.md) | A1/B1/R1-reg forward and loss algorithms plus E14 dispatch proxy algorithm |
| [solution/constraints.md](logic/solution/constraints.md) | Power balance, SOC accumulation, dimension matching |
| [solution/heuristics.md](logic/solution/heuristics.md) | Design heuristics (identity over gated residual, MAE for component loss, warmup, etc.) |
| [related_work.md](logic/related_work.md) | Typed dependency graph: physics-guided DL, VPP forecasting, Transformer variants |

### Physical Layer (`/src`)
| File | Description |
|------|-------------|
| [configs/training.md](src/configs/training.md) | Hyperparameters with rationale for V4 and V5 runs |
| [configs/model.md](src/configs/model.md) | Model architecture configs |
| [configs/v5_curriculum.yaml](src/configs/v5_curriculum.yaml) | V5 curriculum training schedule |
| [execution/physformer.py](src/execution/physformer.py) | PhysFormer forward pass stub |
| [execution/losses.py](src/execution/losses.py) | Loss function stubs (component loss, curriculum schedule) |
| [environment.md](src/environment.md) | Python version, conda env, hardware (AutoDL 5090), seeds |

### Exploration Graph (`/trace`)
| File | Description |
|------|-------------|
| [exploration_tree.yaml](trace/exploration_tree.yaml) | 20+ node research DAG from V3 baseline through V5 tuning, with dead ends and decisions |

### Evidence (`/evidence`)
| File | Description |
|------|-------------|
| [README.md](evidence/README.md) | Index of evidence tables and their claim bindings |
| [tables/version_comparison.md](evidence/tables/version_comparison.md) | V3–V5 aggregate metrics comparison |
| [tables/v5_component_metrics.md](evidence/tables/v5_component_metrics.md) | V5 per-component MAE comparison |
