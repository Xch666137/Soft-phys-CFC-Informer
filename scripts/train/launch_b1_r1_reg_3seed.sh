#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/physformer}"
ENV_NAME="${ENV_NAME:-physformer}"
SESSION_TAG="${SESSION_TAG:-b1_r1_reg_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_DIR/logs/$SESSION_TAG"
RUN_ROOT="$PROJECT_DIR/runs"
CONFIGS=(
  "configs/physformer_igt_b1_r1_reg_finetune.yaml"
  "configs/physformer_igt_b1_r1_reg_finetune_s2026.yaml"
  "configs/physformer_igt_b1_r1_reg_finetune_s2027.yaml"
)
RUNS=(
  "physformer_igt_b1_r1_reg_finetune_s2025"
  "physformer_igt_b1_r1_reg_finetune_s2026"
  "physformer_igt_b1_r1_reg_finetune_s2027"
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

if [[ ! -f results/physformer_igt_b1_pretrain/pretrained_checkpoint.pth ]]; then
  echo "Missing pretrained checkpoint: results/physformer_igt_b1_pretrain/pretrained_checkpoint.pth" >&2
  exit 1
fi

printf "session_tag\t%s\n" "$SESSION_TAG" | tee "$LOG_DIR/jobs.tsv"
printf "started_at\t%s\n" "$(date -Is)" | tee -a "$LOG_DIR/jobs.tsv"

pids=()
for i in "${!RUNS[@]}"; do
  run="${RUNS[$i]}"
  config="${CONFIGS[$i]}"
  if [[ -d "$RUN_ROOT/$run" ]]; then
    mv "$RUN_ROOT/$run" "$RUN_ROOT/${run}.bak_${SESSION_TAG}"
  fi
  (
    set -euo pipefail
    python run.py train --config "$config"
    python run.py test --config "$config"
  ) >"$LOG_DIR/${run}.launcher.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  printf "%s\t%s\t%s\t%s\n" "$run" "$config" "$pid" "$LOG_DIR/${run}.launcher.log" | tee -a "$LOG_DIR/jobs.tsv"
done

echo "Launched ${#RUNS[@]} B1-R1-reg seeds. Logs: $LOG_DIR"
echo "Monitor with:"
echo "  python scripts/train/monitor_b1_3seed_epochs.py --project-dir '$PROJECT_DIR' --session-tag '$SESSION_TAG'"

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

printf "finished_at\t%s\n" "$(date -Is)" | tee -a "$LOG_DIR/jobs.tsv"
exit "$status"
