#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

mkdir -p logs

echo "================================================================"
echo " Start PhysFormer single training"
echo " Target: PhysFormer_full_seed2024"
echo " Time: $(date)"
echo "================================================================"

CKPT_DIR="exp_results/PhysFormer/checkpoints/PhysFormer_full_seed2024"
if [ -d "$CKPT_DIR" ]; then
    rm -f "$CKPT_DIR"/vis_*.npy
    rm -f "$CKPT_DIR"/pred.npy
    rm -f "$CKPT_DIR"/true.npy
    rm -f "$CKPT_DIR"/metrics.npy
fi

CUDA_VISIBLE_DEVICES=0 python run.py \
  --config configs/physformer_default.yaml \
  --checkpoint_name "PhysFormer_full_seed2024" \
  --resume \
  --override data.freq=t \
  --override training.physics_prior_weight=0.05 \
  --override hardware.gpu=0 \
  --override hardware.num_workers=4

echo "================================================================"
echo " Training finished at $(date)"
echo "================================================================"
