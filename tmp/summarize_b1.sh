#!/bin/bash
# B1 Finetune Monitor — reports whenever ANY seed completes a new epoch.
# Shows each seed's latest completed epoch stats.
# State file tracks per-seed last reported epoch.

LOGDIR=/root/autodl-tmp/physformer/logs
SEEDS=(2025 2026 2027)
PREFIX=b1_finetune_s
STATE_FILE=/tmp/b1_monitor_eps

# ---- helpers ----
get_curr_epoch() {
  local log="$1"
  if [ ! -f "$log" ]; then echo "0"; return; fi
  local last
  last=$(grep -P '^Epoch:\s+\d+\s+\|' "$log" | tail -1 2>/dev/null)
  if [ -z "$last" ]; then echo "0"; else echo "$last" | grep -oP '^Epoch:\s+\K\d+'; fi
}

get_epoch_line() {
  local log="$1" epoch="$2"
  grep -P "^Epoch:\s+${epoch}\s+\|" "$log" | head -1
}

# ---- load last reported per seed ----
declare -A LAST_REP
for s in "${SEEDS[@]}"; do
  LAST_REP[$s]=0
done
if [ -f "$STATE_FILE" ]; then
  while IFS='=' read -r seed ep; do
    LAST_REP[$seed]=${ep:-0}
  done < "$STATE_FILE"
fi

# ---- check for new completions ----
NEW=false
declare -A CURR
for s in "${SEEDS[@]}"; do
  LOG="$LOGDIR/${PREFIX}${s}.log"
  CURR[$s]=$(get_curr_epoch "$LOG")
  if [ "${CURR[$s]}" -gt "${LAST_REP[$s]}" ]; then
    NEW=true
  fi
done

# Also check if all done, regardless
DONE_COUNT=0
STOPPED=0
for s in "${SEEDS[@]}"; do
  LOG="$LOGDIR/${PREFIX}${s}.log"
  if grep -q 'Early stopping' "$LOG" 2>/dev/null; then STOPPED=$((STOPPED+1)); fi
  if grep -q 'Test Metrics' "$LOG" 2>/dev/null; then DONE_COUNT=$((DONE_COUNT+1)); fi
done

if [ "$NEW" = false ] && [ "$DONE_COUNT" -lt 3 ]; then
  exit 0  # silent
fi

# ---- output ----
echo "=== B1 Finetune — $(date '+%H:%M:%S') ==="
GPU_INFO=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null)
echo "GPU: $GPU_INFO"
echo

printf "%-6s | %-6s | %-8s | %-10s | %-10s | %-10s | %-10s | %-12s\n" \
  "Seed" "Epoch" "Train" "Val Loss" "Val MSE(N)" "Val MSE(MW²)" "Best" "ES"
echo "------- | ------ | -------- | ---------- | ---------- | ---------- | ---------- | ------------"

for s in "${SEEDS[@]}"; do
  LOG="$LOGDIR/${PREFIX}${s}.log"
  ep=${CURR[$s]}

  if [ "$ep" -eq 0 ]; then
    printf "%-6s | %-6s | %-8s | %-10s | %-10s | %-10s | %-10s | %-12s\n" \
      "$s" "---" "---" "---" "---" "---" "---" "loading"
    continue
  fi

  LINE=$(get_epoch_line "$LOG" "$ep")
  TRAIN=$(echo "$LINE"  | grep -oP 'Train:\s+\K[\d.]+')
  V_LOSS=$(echo "$LINE" | grep -oP 'Val Loss:\s+\K[\d.]+')
  V_MSE=$(echo "$LINE"  | grep -oP 'Val MSE:\s+\K[\d.]+')
  V_MSER=$(echo "$LINE" | sed -n 's/.*Val MSE(MW.*): \([0-9.e+-]\+\).*/\1/p')
  BEST=$(grep -P '^Epoch:\s+\d+\s+\|' "$LOG" | grep -oP 'Val MSE:\s+\K[\d.]+' | sort -n | head -1)

  ES="0/12"
  ES_LINE=$(grep "EarlyStopping counter" "$LOG" 2>/dev/null | tail -1)
  if [ -n "$ES_LINE" ]; then
    ES=$(echo "$ES_LINE" | grep -oP '\d+ out of \d+' || echo "0/12")
  fi

  TAG=""
  if grep -q 'Early stopping' "$LOG" 2>/dev/null; then TAG="⏹"; fi
  if grep -q 'Test Metrics' "$LOG" 2>/dev/null; then TAG="✓"; fi

  printf "%-6s | %-6s | %-8s | %-10s | %-10s | %-10s | %-10s | %-12s\n" \
    "$s" "E$ep" "${TRAIN:----}" "${V_LOSS:----}" "${V_MSE:----}" "${V_MSER:----}" "${BEST:----}" "$ES $TAG"
done

# Update state
for s in "${SEEDS[@]}"; do
  echo "${s}=${CURR[$s]}"
done > "$STATE_FILE"

# If all done, signal
if [ "$DONE_COUNT" -eq 3 ] || [ "$STOPPED" -eq 3 ]; then
  echo
  echo "=== ALL DONE ==="
fi
