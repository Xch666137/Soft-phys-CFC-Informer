# Environment

## Python Environment
- **Python version**: 3.10 (conda environment: PhysFormer)
- **Package manager**: pip + conda
- **Key dependencies**: See `requirements.txt` at project root
- **Framework**: PyTorch (CUDA-enabled for GPU training)

## Hardware

### Training (AutoDL Cloud)
- **Current NVIDIA instance**: `root@connect.westd.seetacloud.com`, SSH port `45015`
- **Verified GPU**: NVIDIA GeForce RTX 4080 SUPER, 32760 MiB, driver 580.142
- **Verified host**: `autodl-container-7j2mj0bz0c-e7dd7a39`
- **Remote project dir**: `/root/autodl-tmp/physformer`
- **Remote conda env**: `physformer`
- **Remote PyTorch**: 2.7.1+cu118, `torch.cuda.is_available() == True`
- **Remote data status**: `data_raw/` and `data_processed/multi_portfolio/` are present; no `.autodl_upload_complete` stamp was found, so use `-SkipDataUpload` only after verifying data files still exist.
- **Status**: cloned replacement instance for the unavailable previous remote GPU
- **CPU/RAM/Storage**: AutoDL-configured; use AutoDL data disk for datasets and checkpoints
- **Operational note**: formal training should use NVIDIA/AutoDL, not local AMD ROCm. Scripts are parameterized with `-RemoteHost` and `-Port`; pass the current endpoint explicitly.

### Local Development
- **Machine**: Laptop/Desktop (Windows 11)
- **CPU only**: conda environment for local import validation, config testing
- **No GPU training locally**: All training runs on AutoDL

## Dataset
- **Source**: VPP portfolio data (proprietary)
- **Components**: Load, PV, Wind, Battery (power + SOC)
- **Resolution**: 15-minute intervals
- **Features**: Historical net injection, irradiance, temperature, wind speed, calendar (hour, weekday, month)
- **Split**: Train/Val/Test (temporal split, no random shuffling)
- **Sequence**: Input 96 steps (24 hours), Output 96 steps (24 hours)

## Seeds & Reproducibility
- **Random seed**: Set per experiment in config YAML (typically seed=42 or seed=3407)
- **Deterministic operations**: `torch.backends.cudnn.deterministic = True` where possible
- **Note**: Full reproducibility across GPU architectures not guaranteed due to CUDA non-determinism in some ops

## Version Control
- **Git worktree**: `codex/thesis-mainline` branch
- **Key commits**:
  - `280c93f` — V5: Component-consistent residual with curriculum training
  - `df4661c` — Phase 1 + 1.1: Load-aware enhancements
  - `5ce164e` — Baseline benchmarks, three-stage training pipeline
