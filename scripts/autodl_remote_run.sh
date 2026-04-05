#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/autodl_remote_run.sh [options]

Options:
  --project-dir PATH
  --env-name NAME
  --python-version VERSION
  --stages CSV
  --log-root PATH
  --validate-config PATH
  --validate-run-name NAME
  --mapping-csv PATH
  --help

Default stages:
  verify,build_dataset,benchmark_main,benchmark_time
EOF
}

PROJECT_DIR="/root/autodl-tmp/Soft-phys-CFC-Informer"
ENV_NAME="Soft-phys-CFC-Informer"
PYTHON_VERSION="3.10"
STAGES="verify,build_dataset,benchmark_main,benchmark_time"
LOG_ROOT=""
VALIDATE_CONFIG=""
VALIDATE_RUN_NAME=""
MAPPING_CSV=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-dir)
      PROJECT_DIR="$2"
      shift 2
      ;;
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --python-version)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --stages)
      STAGES="$2"
      shift 2
      ;;
    --log-root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --validate-config)
      VALIDATE_CONFIG="$2"
      shift 2
      ;;
    --validate-run-name)
      VALIDATE_RUN_NAME="$2"
      shift 2
      ;;
    --mapping-csv)
      MAPPING_CSV="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[autodl-remote] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$LOG_ROOT" ]]; then
  LOG_ROOT="$PROJECT_DIR/logs/autodl"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/$timestamp"
MASTER_LOG="$LOG_DIR/master.log"
mkdir -p "$LOG_DIR"

log() {
  echo "[autodl-remote] $*" | tee -a "$MASTER_LOG"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[autodl-remote] Required file not found: $path" | tee -a "$MASTER_LOG"
    exit 1
  fi
}

run_stage_cmd() {
  local stage_name="$1"
  local command="$2"
  local stage_log="$LOG_DIR/${stage_name}.log"
  log "START stage=$stage_name"
  (
    set -euo pipefail
    cd "$PROJECT_DIR"
    bash -lc "$command"
  ) 2>&1 | tee -a "$stage_log" "$MASTER_LOG"
  log "DONE stage=$stage_name"
}

ensure_conda_env() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[autodl-remote] conda was not found in PATH." | tee -a "$MASTER_LOG"
    exit 1
  fi

  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"

  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Creating conda environment: $ENV_NAME (python=$PYTHON_VERSION)"
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" | tee -a "$MASTER_LOG"
  fi

  conda activate "$ENV_NAME"
  require_file "$PROJECT_DIR/requirements.txt"

  local stamp_file="$PROJECT_DIR/.autodl_requirements_sha"
  local req_hash
  req_hash="$(sha256sum "$PROJECT_DIR/requirements.txt" | awk '{print $1}')"
  local current_hash=""
  if [[ -f "$stamp_file" ]]; then
    current_hash="$(cat "$stamp_file")"
  fi

  if [[ "$current_hash" != "$req_hash" ]]; then
    log "Installing Python requirements into $ENV_NAME"
    python -m pip install --upgrade pip | tee -a "$MASTER_LOG"
    pip install -r "$PROJECT_DIR/requirements.txt" | tee -a "$MASTER_LOG"
    echo "$req_hash" > "$stamp_file"
  else
    log "Requirements already up to date for $ENV_NAME"
  fi
}

log "project_dir=$PROJECT_DIR"
log "env_name=$ENV_NAME"
log "stages=$STAGES"
log "log_dir=$LOG_DIR"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "[autodl-remote] Project directory not found: $PROJECT_DIR" | tee -a "$MASTER_LOG"
  exit 1
fi

ensure_conda_env

IFS=',' read -r -a stage_array <<< "$STAGES"

for raw_stage in "${stage_array[@]}"; do
  stage="$(echo "$raw_stage" | xargs)"
  case "$stage" in
    verify)
      run_stage_cmd "verify" "source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate '$ENV_NAME' && python verify_imports.py && python run.py train --config configs/physformer_default.yaml --print-config && python run.py train --config configs/baselines/tide_net_injection.yaml --print-config"
      ;;
    build_dataset)
      run_stage_cmd "build_dataset" "source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate '$ENV_NAME' && python run.py build-dataset --nextgen-dir data_raw/nextgen --act-weather-csv data_raw/era5/act_canberra_hourly.csv --rye-generation-csv data_raw/rye/rye_generation_and_load.csv --rye-weather-csv data_raw/era5/rye_template_hourly.csv --output-dir data_processed/multi_portfolio"
      ;;
    benchmark_main)
      run_stage_cmd "benchmark_main" "source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate '$ENV_NAME' && python run.py benchmark --config configs/drivers/benchmark_net_injection.yaml"
      ;;
    benchmark_time)
      run_stage_cmd "benchmark_time" "source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate '$ENV_NAME' && python run.py benchmark --config configs/drivers/benchmark_net_injection_time_generalization.yaml"
      ;;
    ablation)
      run_stage_cmd "ablation" "source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate '$ENV_NAME' && python run.py ablation --config configs/drivers/physformer_ablation.yaml"
      ;;
    appendix)
      run_stage_cmd "appendix_main" "source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate '$ENV_NAME' && python run.py benchmark --config configs/drivers/benchmark_net_injection_appendix.yaml"
      run_stage_cmd "appendix_time" "source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate '$ENV_NAME' && python run.py benchmark --config configs/drivers/benchmark_net_injection_appendix_time_generalization.yaml"
      ;;
    validate_powerflow)
      if [[ -z "$VALIDATE_CONFIG" || -z "$VALIDATE_RUN_NAME" || -z "$MAPPING_CSV" ]]; then
        echo "[autodl-remote] validate_powerflow requires --validate-config, --validate-run-name, and --mapping-csv" | tee -a "$MASTER_LOG"
        exit 1
      fi
      run_stage_cmd "export_forecast" "source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate '$ENV_NAME' && python run.py export-forecast --config '$VALIDATE_CONFIG' --run-name '$VALIDATE_RUN_NAME'"
      run_stage_cmd "validate_powerflow" "source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate '$ENV_NAME' && python run.py validate-powerflow --config '$VALIDATE_CONFIG' --run-name '$VALIDATE_RUN_NAME' --mapping-csv '$MAPPING_CSV'"
      ;;
    "")
      ;;
    *)
      echo "[autodl-remote] Unknown stage: $stage" | tee -a "$MASTER_LOG"
      exit 1
      ;;
  esac
done

log "All requested stages completed successfully."
