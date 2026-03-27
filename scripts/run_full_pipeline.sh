#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

mkdir -p logs

echo "================================================================"
echo " PhysFormer full experiment pipeline"
echo " Time: $(date)"
echo "================================================================"

run_experiment() {
    local config_path=$1
    local name=$2
    local gpu_id=$3
    echo "--- ${name} (GPU ${gpu_id}) ---"
    CUDA_VISIBLE_DEVICES=$gpu_id python run.py \
      --config "$config_path" \
      --checkpoint_name "$name" \
      --override hardware.gpu=0 \
      > "logs/${name}.log" 2>&1
}

run_experiment "configs/physformer_default.yaml"      "PhysFormer_full_seed2024"       0 &
PID_FULL=$!
run_experiment "configs/physformer_ablation_v1.yaml"  "PhysFormer_ablation_V1_no_phys" 1 &
PID_V1=$!
run_experiment "configs/physformer_ablation_v2.yaml"  "PhysFormer_ablation_V2_no_pgcc" 2 &
PID_V2=$!
run_experiment "configs/physformer_ablation_v3.yaml"  "PhysFormer_ablation_V3_no_future_glu" 3 &
PID_V3=$!
run_experiment "configs/physformer_ablation_v4.yaml"  "PhysFormer_ablation_V4_no_curriculum" 4 &
PID_V4=$!

FAILED=0
for PID in $PID_FULL $PID_V1 $PID_V2 $PID_V3 $PID_V4; do
    wait $PID || FAILED=$((FAILED + 1))
done

if [ $FAILED -gt 0 ]; then
    echo "[warning] $FAILED experiments failed. Check logs/*.log"
else
    echo "All experiments completed"
fi
