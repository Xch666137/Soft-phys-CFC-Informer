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
abstract: "Virtual power plant (VPP) aggregated net power forecasting is critical for dispatch and market participation, yet pure deep-learning approaches produce physically inconsistent predictions. We propose PhysFormer, a physics-guided Transformer that embeds domain equations via FiLM-conditioned theory branches for photovoltaic, wind, and battery components while modeling load with behavioral/temporal structure. A component-consistent residual mechanism preserves per-asset-type gradient isolation, and curriculum training improves robustness to component-loss weight choice. Experiments on real VPP data demonstrate improved component-level physical consistency, while variance decomposition reveals that aggregate accuracy can improve through signed component-error cancellation even when physical component forecasts degrade."
status: "in-progress"
phase: "Phase B rescue implementation (2026-05-31): N113 remains a dead end only for MCP pretrain -> high-LR pure net_mse finetune on the static split. N114 implemented scaler-buffer repair, best-Val-Net checkpointing, low-LR/regularized finetune arms, and a DVPP target-portfolio few-shot adaptation gate. Next: run B1-R0 old-vs-fixed scaler direct test, B1-R1 3-seed rescue arms, then B1-R2 5/10/20% target adaptation vs A1 scratch."
---
# PhysFormer: Physics-Guided Transformer for VPP Aggregated Net Power Forecasting

## Overview

PhysFormer addresses the VPP aggregated net power forecasting problem by combining
physics equations for DER components (PV, Wind, Battery) with behavioral/temporal
load modeling inside a Transformer architecture. Unlike pure black-box approaches,
PhysFormer decomposes the forecast into theory-driven and residual components, with
per-asset-type FiLM conditioning and a component-consistent residual mechanism that
prevents cross-component gradient contamination. Curriculum training is currently
interpreted as improving robustness to component-loss weight choice rather than as
evidence for a strict Phase1-to-Phase2 refinement path.

This artifact captures the research journey from baseline (V3) through V7/P1 and
Phase A, including dead ends (sigmoid gate, excessive component loss, invalid
curriculum Phase 3, AMD ROCm training instability) and key design decisions
(identity over gated residual, MAE over MSE for component loss, per-component over
scalar residual, behavioral Load modeling, NVIDIA-only formal training).

## Layer Index

### Cognitive Layer (`/logic`)
| File | Description |
|------|-------------|
| [problem.md](logic/problem.md) | Observations on VPP forecasting gaps, physics-data integration challenges |
| [claims.md](logic/claims.md) | Falsifiable claims on physics guidance, residual learning, curriculum training |
| [concepts.md](logic/concepts.md) | Formal definitions: FiLM conditioning, component-consistent residual, theory/aggregate net |
| [experiments.md](logic/experiments.md) | Declarative experiment plans (E01–E06) |
| [solution/architecture.md](logic/solution/architecture.md) | Component graph: encoder, FiLM, theory branches, temporal decoder, residual heads |
| [solution/algorithm.md](logic/solution/algorithm.md) | Math formulation of net power decomposition and residual correction |
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
