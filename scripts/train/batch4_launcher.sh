#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/physformer
mkdir -p logs/p1_batch4

CONFIGS=(
  "physformer_p1b_d512"
  "physformer_p1b_e3"
  "physformer_p1b_d512_e3"
)
PIDS=()

echo "[b4_launcher] Launching Batch 4 (encoder bottleneck: d512 / e3 / d512+e3)..."
for cfg in "${CONFIGS[@]}"; do
  echo "[b4_launcher] Starting $cfg..."
  nohup /root/miniconda3/bin/conda run -n physformer python run.py train --config configs/${cfg}.yaml > /dev/null 2>&1 &
  pid=$!
  PIDS+=($pid)
  echo "  $cfg PID=$pid"
  sleep 5
done

echo "[b4_launcher] All 3 launched. PIDs: ${PIDS[*]}"
echo "[b4_launcher] Waiting 30s for trainer initialization..."

sleep 30
FAILED=()
for cfg in "${CONFIGS[@]}"; do
  LOG="/root/autodl-tmp/physformer/runs/${cfg}/train.log"
  if [ -f "$LOG" ] && [ -s "$LOG" ]; then
    echo "[b4_launcher] VERIFIED: $cfg"
  else
    echo "[b4_launcher] FAILED: $cfg"
    FAILED+=("$cfg")
  fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "[b4_launcher] ERROR: ${#FAILED[@]} experiments failed: ${FAILED[*]}"
  exit 1
fi

echo "[b4_launcher] All 3 verified. GPU:"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

cat > /root/autodl-tmp/physformer/scripts/train/batch4_finisher.sh << 'FINISHER'
#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/physformer
CONFIGS=(physformer_p1b_d512 physformer_p1b_e3 physformer_p1b_d512_e3)

for cfg in "${CONFIGS[@]}"; do
  echo "[b4_finisher] Waiting for $cfg training..."
  while ps aux | grep "run.py train.*$cfg" | grep -v grep > /dev/null 2>&1; do sleep 30; done
  echo "[b4_finisher] $cfg training DONE, running test..."
  /root/miniconda3/bin/conda run -n physformer python run.py test --config configs/${cfg}.yaml > runs/${cfg}/test.log 2>&1
  echo "[b4_finisher] $cfg test DONE"
done

echo "=== BATCH4 RESULTS ==="
for cfg in "${CONFIGS[@]}"; do
  echo "--- $cfg ---"
  /root/miniconda3/bin/conda run -n physformer python -c "import json; m=json.load(open('runs/$cfg/metrics.json')); print(f'MAE={m[\"mae\"]:.6f} MSE={m[\"mse\"]:.6e} TheoryMAE={m[\"theory_mae\"]:.6f} ResMean={m[\"residual_mean_real_mw\"]:.6f}')"
done
echo "=== BATCH4 COMPLETE ==="
FINISHER
chmod +x /root/autodl-tmp/physformer/scripts/train/batch4_finisher.sh
nohup bash /root/autodl-tmp/physformer/scripts/train/batch4_finisher.sh > /root/autodl-tmp/physformer/logs/p1_batch4/finisher.log 2>&1 &
echo "[b4_launcher] Auto-finisher deployed (PID=$!). Done."
