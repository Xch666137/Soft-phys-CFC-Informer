#!/bin/bash
cd /root/autodl-tmp/physformer
CONFIGS=(physformer_p1a_baseline_s2026 physformer_p1a_detach_s2026 physformer_p1a_notemp_s2026)

for cfg in "${CONFIGS[@]}"; do
  echo "[finisher] Waiting for $cfg training..."
  while ps aux | grep "run.py train.*$cfg" | grep -v grep > /dev/null 2>&1; do
    sleep 30
  done
  echo "[finisher] $cfg training DONE, running test..."
  /root/miniconda3/bin/conda run -n physformer python run.py test --config configs/${cfg}.yaml > runs/${cfg}/test.log 2>&1
  echo "[finisher] $cfg test DONE"
done

echo "=== BATCH2 RESULTS ==="
for cfg in "${CONFIGS[@]}"; do
  echo "--- $cfg ---"
  /root/miniconda3/bin/conda run -n physformer python -c "import json; m=json.load(open('runs/$cfg/metrics.json')); print(f'MAE={m[\"mae\"]:.6f} MSE={m[\"mse\"]:.6e} TheoryMAE={m[\"theory_mae\"]:.6f}')"
done
echo "=== BATCH2 COMPLETE ==="
