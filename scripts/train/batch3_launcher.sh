#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/physformer
mkdir -p logs/p1_batch3

CONFIGS=(
  "physformer_p1a_detach_a03"
  "physformer_p1a_detach_a05"
  "physformer_p1a_detach_a07"
)
PIDS=()

echo "[b3_launcher] Launching Batch 3 (gradient scaling α=0.3/0.5/0.7)..."
for cfg in "${CONFIGS[@]}"; do
  echo "[b3_launcher] Starting $cfg..."
  nohup /root/miniconda3/bin/conda run -n physformer python run.py train --config configs/${cfg}.yaml > /dev/null 2>&1 &
  pid=$!
  PIDS+=($pid)
  echo "  $cfg PID=$pid"
  sleep 5
done

echo "[b3_launcher] All 3 launched. PIDs: ${PIDS[*]}"
echo "[b3_launcher] Waiting 30s for trainer initialization..."

# Verify all 3 are actually running
sleep 30
FAILED=()
for cfg in "${CONFIGS[@]}"; do
  LOG="/root/autodl-tmp/physformer/runs/${cfg}/train.log"
  if [ -f "$LOG" ] && [ -s "$LOG" ]; then
    echo "[b3_launcher] VERIFIED: $cfg — train.log exists and non-empty"
  else
    echo "[b3_launcher] FAILED: $cfg — train.log missing or empty"
    FAILED+=("$cfg")
  fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "[b3_launcher] ERROR: ${#FAILED[@]} experiments failed to start: ${FAILED[*]}"
  exit 1
fi

echo "[b3_launcher] All 3 experiments verified running. GPU status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

echo "[b3_launcher] Deploying auto-finisher..."
# Write finisher inline
cat > /root/autodl-tmp/physformer/scripts/train/batch3_finisher.sh << 'FINISHER'
#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/physformer
CONFIGS=(physformer_p1a_detach_a03 physformer_p1a_detach_a05 physformer_p1a_detach_a07)

for cfg in "${CONFIGS[@]}"; do
  echo "[b3_finisher] Waiting for $cfg training..."
  while ps aux | grep "run.py train.*$cfg" | grep -v grep > /dev/null 2>&1; do
    sleep 30
  done
  echo "[b3_finisher] $cfg training DONE, running test..."
  /root/miniconda3/bin/conda run -n physformer python run.py test --config configs/${cfg}.yaml > runs/${cfg}/test.log 2>&1
  echo "[b3_finisher] $cfg test DONE"
done

echo "=== BATCH3 RESULTS ==="
for cfg in "${CONFIGS[@]}"; do
  echo "--- $cfg ---"
  /root/miniconda3/bin/conda run -n physformer python -c "import json; m=json.load(open('runs/$cfg/metrics.json')); print(f'MAE={m[\"mae\"]:.6f} MSE={m[\"mse\"]:.6e} TheoryMAE={m[\"theory_mae\"]:.6f} ResMean={m[\"residual_mean_real_mw\"]:.6f}')"
done
echo "=== BATCH3 COMPLETE ==="
FINISHER
chmod +x /root/autodl-tmp/physformer/scripts/train/batch3_finisher.sh
nohup bash /root/autodl-tmp/physformer/scripts/train/batch3_finisher.sh > /root/autodl-tmp/physformer/logs/p1_batch3/finisher.log 2>&1 &
echo "[b3_launcher] Auto-finisher deployed (PID=$!). All done."
