#!/bin/bash
set -euo pipefail

# Thin Linux wrapper for PhysFormer ablation runs.
#
# Example:
#   bash scripts/ablation.sh --config configs/drivers/physformer_ablation.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Soft-phys-CFC-Informer

python run.py ablation "$@"
