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
  --stage-a-config PATH
  --stage-a-run-name NAME
  --operational-config PATH
  --operational-init-run PATH
  --operational-run-name NAME
  --force-rebuild-dataset
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
STAGE_A_CONFIG=""
STAGE_A_RUN_NAME=""
OPERATIONAL_CONFIG=""
OPERATIONAL_INIT_RUN=""
OPERATIONAL_RUN_NAME=""
FORCE_REBUILD_DATASET="false"

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
    --stage-a-config)
      STAGE_A_CONFIG="$2"
      shift 2
      ;;
    --stage-a-run-name)
      STAGE_A_RUN_NAME="$2"
      shift 2
      ;;
    --operational-config)
      OPERATIONAL_CONFIG="$2"
      shift 2
      ;;
    --operational-init-run)
      OPERATIONAL_INIT_RUN="$2"
      shift 2
      ;;
    --operational-run-name)
      OPERATIONAL_RUN_NAME="$2"
      shift 2
      ;;
    --force-rebuild-dataset)
      FORCE_REBUILD_DATASET="true"
      shift 1
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

dataset_ready() {
  local output_dir="$PROJECT_DIR/data_processed/multi_portfolio"
  [[ -f "$output_dir/portfolio_dataset_for_training.csv" ]] &&
  [[ -f "$output_dir/portfolio_dataset_for_time_generalization.csv" ]] &&
  [[ -f "$output_dir/multi_portfolio_metadata.json" ]]
}

run_stage_cmd() {
  local stage_name="$1"
  local command="$2"
  local stage_log="$LOG_DIR/${stage_name}.log"
  log "START stage=$stage_name"
  (
    set -euo pipefail
    cd "$PROJECT_DIR"
    eval "$command"
  ) 2>&1 | tee -a "$stage_log" "$MASTER_LOG"
  log "DONE stage=$stage_name"
}

run_batch_audit() {
  local audit_log="$LOG_DIR/audit_batch.log"
  local summary_file="$LOG_DIR/audit_batch_summary.txt"
  : > "$summary_file"

  audit_one_model() {
    local model_name="$1"
    local config_path="$2"
    shift 2
    local best_batch=""

    for batch_size in "$@"; do
      local audit_run_name="${model_name}_audit_bs${batch_size}"
      log "AUDIT start model=$model_name batch_size=$batch_size"
      set +e
      (
        set -euo pipefail
        cd "$PROJECT_DIR"
        python run.py train --config "$config_path" --epochs 2 --batch-size "$batch_size" --run-name "$audit_run_name"
      ) 2>&1 | tee -a "$audit_log" "$MASTER_LOG"
      local status=${PIPESTATUS[0]}
      set -e

      if [[ "$status" -eq 0 ]]; then
        best_batch="$batch_size"
        log "AUDIT success model=$model_name batch_size=$batch_size"
      else
        log "AUDIT fail model=$model_name batch_size=$batch_size status=$status"
        break
      fi
    done

    if [[ -n "$best_batch" ]]; then
      echo "$model_name best_stable_batch=$best_batch" | tee -a "$summary_file" "$MASTER_LOG"
    else
      echo "$model_name best_stable_batch=NONE" | tee -a "$summary_file" "$MASTER_LOG"
    fi
  }

  audit_one_model "physformer" "configs/physformer_default.yaml" 96 128 160
  audit_one_model "tide" "configs/baselines/tide_net_injection.yaml" 96 128 160
  audit_one_model "timexer" "configs/baselines/timexer_net_injection.yaml" 64 96 128
  audit_one_model "tft" "configs/baselines/tft_net_injection.yaml" 64 96 128
  audit_one_model "dlinear" "configs/baselines/dlinear_net_injection.yaml" 256 512
}

run_physformer_hparam_probe() {
  local probe_run_name="physformer_hparam_probe_5090"
  local probe_run_dir="$PROJECT_DIR/runs/$probe_run_name"
  rm -rf "$probe_run_dir"

  run_stage_cmd "probe_hparams_train" "python run.py train --config configs/physformer_probe_5090.yaml"
  run_stage_cmd "probe_hparams_analyze" "python tools/analyze_physformer_probe.py --run-dir '$probe_run_dir' --warmup-epochs 5"
}

run_stage_a_single() {
  local config_path="${STAGE_A_CONFIG:-configs/physformer_default.yaml}"
  local run_name="${STAGE_A_RUN_NAME:-physformer_net_injection__s2024}"

  run_stage_cmd "stage_a_train" "python run.py train --config '$config_path' --run-name '$run_name' --batch-size 128 --num-workers 12 --lr 1e-4 --patience 25 --epochs 100"
  run_stage_cmd "stage_a_test" "python run.py test --config '$config_path' --run-name '$run_name'"
}

print_stage_a_config() {
  local config_path="${STAGE_A_CONFIG:-configs/physformer_default.yaml}"
  local run_name="${STAGE_A_RUN_NAME:-physformer_net_injection__s2024}"
  python run.py train --config "$config_path" --run-name "$run_name" --batch-size 128 --num-workers 12 --lr 1e-4 --patience 25 --epochs 100 --print-config
}

run_operational_fit() {
  local config_path="${OPERATIONAL_CONFIG:-configs/physformer_operational_fit.yaml}"
  local init_run="${OPERATIONAL_INIT_RUN:-}"
  local run_name="${OPERATIONAL_RUN_NAME:-physformer_operational_fit}"

  if [[ -z "$init_run" && -n "$STAGE_A_RUN_NAME" ]]; then
    init_run="$PROJECT_DIR/runs/$STAGE_A_RUN_NAME"
  fi

  if [[ -z "$init_run" ]]; then
    echo "[autodl-remote] operational_fit requires --operational-init-run" | tee -a "$MASTER_LOG"
    exit 1
  fi

  run_stage_cmd "operational_fit_train" "python run.py train --config '$config_path' --init-from-run '$init_run' --run-name '$run_name'"
  run_stage_cmd "operational_fit_test" "python run.py test --config '$config_path' --init-from-run '$init_run' --run-name '$run_name'"
  run_stage_cmd "operational_fit_analyze" "python tools/analyze_operational_interface.py --run-dir '$PROJECT_DIR/runs/$run_name'"
}

ensure_conda_env() {
  local conda_bin=""
  local conda_base=""
  local candidate=""

  if command -v conda >/dev/null 2>&1; then
    conda_bin="$(command -v conda)"
  else
    for candidate in \
      "/root/miniconda3/bin/conda" \
      "$HOME/miniconda3/bin/conda" \
      "/opt/conda/bin/conda"
    do
      if [[ -x "$candidate" ]]; then
        conda_bin="$candidate"
        break
      fi
    done
  fi

  if [[ -z "$conda_bin" ]]; then
    echo "[autodl-remote] conda was not found in PATH or common AutoDL locations." | tee -a "$MASTER_LOG"
    exit 1
  fi

  conda_base="$("$conda_bin" info --base)"
  export PATH="$conda_base/bin:$PATH"

  if [[ -f "$conda_base/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$conda_base/etc/profile.d/conda.sh"
  else
    eval "$("$conda_bin" shell.bash hook)"
  fi

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
      run_stage_cmd "verify" "python -c \"import sys; print(sys.executable)\" && python verify_imports.py && print_stage_a_config"
      ;;
    build_dataset)
      if [[ "$FORCE_REBUILD_DATASET" != "true" ]] && dataset_ready; then
        log "SKIP stage=build_dataset reason=existing_outputs"
      else
        run_stage_cmd "build_dataset" "python run.py build-dataset --nextgen-dir data_raw/nextgen --act-weather-csv data_raw/era5/act_canberra_hourly.csv --rye-generation-csv data_raw/rye/rye_generation_and_load.csv --rye-weather-csv data_raw/era5/rye_template_hourly.csv --output-dir data_processed/multi_portfolio"
      fi
      ;;
    benchmark_main)
      run_stage_cmd "benchmark_main" "python run.py benchmark --config configs/drivers/benchmark_net_injection_5090.yaml"
      ;;
    benchmark_time)
      run_stage_cmd "benchmark_time" "python run.py benchmark --config configs/drivers/benchmark_net_injection_time_generalization_5090.yaml"
      ;;
    stage_a_single)
      log "START stage=stage_a_single"
      run_stage_a_single
      log "DONE stage=stage_a_single"
      ;;
    probe_hparams)
      log "START stage=probe_hparams"
      run_physformer_hparam_probe
      log "DONE stage=probe_hparams"
      ;;
    audit_batch)
      log "START stage=audit_batch"
      run_batch_audit
      log "DONE stage=audit_batch"
      ;;
    ablation)
      run_stage_cmd "ablation" "python run.py ablation --config configs/drivers/physformer_ablation.yaml"
      ;;
    operational_fit)
      log "START stage=operational_fit"
      run_operational_fit
      log "DONE stage=operational_fit"
      ;;
    export_operational)
      validate_config="${VALIDATE_CONFIG:-${OPERATIONAL_CONFIG:-configs/physformer_operational_fit.yaml}}"
      validate_run_name="${VALIDATE_RUN_NAME:-${OPERATIONAL_RUN_NAME:-physformer_operational_fit}}"
      if [[ -z "$validate_config" || -z "$validate_run_name" ]]; then
        echo "[autodl-remote] export_operational requires a config and run name" | tee -a "$MASTER_LOG"
        exit 1
      fi
      run_stage_cmd "export_operational" "python run.py export-forecast --config '$validate_config' --run-name '$validate_run_name' --include-operational-interface"
      ;;
    appendix)
      run_stage_cmd "appendix_main" "python run.py benchmark --config configs/drivers/benchmark_net_injection_appendix.yaml"
      run_stage_cmd "appendix_time" "python run.py benchmark --config configs/drivers/benchmark_net_injection_appendix_time_generalization.yaml"
      ;;
    validate_powerflow)
      if [[ -z "$VALIDATE_CONFIG" || -z "$VALIDATE_RUN_NAME" || -z "$MAPPING_CSV" ]]; then
        echo "[autodl-remote] validate_powerflow requires --validate-config, --validate-run-name, and --mapping-csv" | tee -a "$MASTER_LOG"
        exit 1
      fi
      run_stage_cmd "export_forecast" "python run.py export-forecast --config '$VALIDATE_CONFIG' --run-name '$VALIDATE_RUN_NAME'"
      run_stage_cmd "validate_powerflow" "python run.py validate-powerflow --config '$VALIDATE_CONFIG' --run-name '$VALIDATE_RUN_NAME' --mapping-csv '$MAPPING_CSV'"
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
