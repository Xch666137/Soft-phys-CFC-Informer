#!/bin/bash
set -euo pipefail

# Thin Linux wrapper for the thesis runner.
#
# Examples:
#   bash scripts/train.sh --config configs/physformer_default.yaml
#   bash scripts/train.sh --config configs/physformer_time_generalization.yaml
#   bash scripts/train.sh --config configs/baselines/tide_net_injection.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Soft-phys-CFC-Informer

python run.py train "$@"
