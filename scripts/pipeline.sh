#!/bin/bash
set -euo pipefail

# Thin Linux wrapper for the full thesis pipeline:
# build multi-portfolio dataset -> train -> test -> export -> validate.
#
# Example:
#   bash scripts/pipeline.sh \
#     --config configs/physformer_default.yaml \
#     --mapping-csv templates/network_mapping.csv \
#     --nextgen-dir data_raw/nextgen \
#     --act-weather-csv data_raw/era5/act_canberra_hourly.csv \
#     --rye-generation-csv data_raw/rye/rye_generation_and_load.csv \
#     --rye-weather-csv data_raw/era5/rye_template_hourly.csv \
#     --output-dir data_processed/multi_portfolio

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Soft-phys-CFC-Informer

python run.py pipeline "$@"
