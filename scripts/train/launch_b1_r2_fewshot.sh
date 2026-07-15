#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/physformer}"
ENV_NAME="${ENV_NAME:-physformer}"
SESSION_TAG="${SESSION_TAG:-b1_r2_fewshot_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_DIR/logs/$SESSION_TAG"
RUN_ROOT="$PROJECT_DIR/runs"
CONFIGS=(
  "configs/physformer_igt_b1_r2_fewshot_adapt_f05.yaml"
  "configs/physformer_igt_b1_r2_fewshot_adapt.yaml"
  "configs/physformer_igt_b1_r2_fewshot_adapt_f20.yaml"
)
RUNS=(
  "physformer_igt_b1_r2_fewshot_adapt_f05_s2025"
  "physformer_igt_b1_r2_fewshot_adapt_f10_s2025"
  "physformer_igt_b1_r2_fewshot_adapt_f20_s2025"
)

find_conda() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi
  for candidate in /root/miniconda3/bin/conda "$HOME/miniconda3/bin/conda" /opt/conda/bin/conda; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done
  return 1
}

CONDA_BIN="$(find_conda)"
CONDA_BASE="$("$CONDA_BIN" info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR" "$RUN_ROOT"

if [[ ! -f results/physformer_igt_b1_pretrain_lam10/best_val_net_checkpoint.pth ]]; then
  echo "Missing pretrained checkpoint: results/physformer_igt_b1_pretrain_lam10/best_val_net_checkpoint.pth" >&2
  exit 1
fi

printf "session_tag\t%s\n" "$SESSION_TAG" | tee "$LOG_DIR/jobs.tsv"
printf "started_at\t%s\n" "$(date -Is)" | tee -a "$LOG_DIR/jobs.tsv"

status=0
for i in "${!RUNS[@]}"; do
  run="${RUNS[$i]}"
  config="${CONFIGS[$i]}"
  if [[ -d "$RUN_ROOT/$run" ]]; then
    mv "$RUN_ROOT/$run" "$RUN_ROOT/${run}.bak_${SESSION_TAG}"
  fi
  printf "%s\t%s\t%s\n" "$run" "$config" "$LOG_DIR/${run}.launcher.log" | tee -a "$LOG_DIR/jobs.tsv"
  if (
    set -euo pipefail
    python run.py train --config "$config"
    python run.py test --config "$config"
  ) >"$LOG_DIR/${run}.launcher.log" 2>&1; then
    printf "%s\tfinished\t0\t%s\n" "$run" "$(date -Is)" | tee -a "$LOG_DIR/jobs.tsv"
  else
    status=1
    printf "%s\tfailed\t1\t%s\n" "$run" "$(date -Is)" | tee -a "$LOG_DIR/jobs.tsv"
  fi
done

echo "Finished ${#RUNS[@]} B1-R2 few-shot runs. Logs: $LOG_DIR"

printf "finished_at\t%s\n" "$(date -Is)" | tee -a "$LOG_DIR/jobs.tsv"
exit "$status"
