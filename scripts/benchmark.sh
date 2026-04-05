#!/bin/bash
set -euo pipefail

# Thin Linux wrapper for thesis benchmark drivers.
#
# Examples:
#   bash scripts/benchmark.sh --config configs/drivers/benchmark_net_injection.yaml
#   bash scripts/benchmark.sh --config configs/drivers/benchmark_net_injection_time_generalization.yaml
#   bash scripts/benchmark.sh --config configs/drivers/benchmark_net_injection_appendix.yaml
#
# Main benchmark drivers run the paper-strong matrix:
#   PhysFormer v2 / DLinear / TiDE / TimeXer / TFT
# with 3 fixed seeds:
#   2024 / 2025 / 2026
#
# Reports are written to:
#   runs/reports/<driver_name>_summary_raw.csv
#   runs/reports/<driver_name>_summary_grouped.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Soft-phys-CFC-Informer

python run.py benchmark "$@"
