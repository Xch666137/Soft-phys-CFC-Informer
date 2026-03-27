#!/bin/bash
# Remote server verification — run on GPU server after refactoring
set -e

echo "=== 1. Install package ==="
pip install -e .

echo "=== 2. Local Windows/CPU verification suite ==="
python scripts/verify_local_all.py

echo "=== 3. PhysFormer single epoch training (GPU server required) ==="
python run.py --config configs/physformer_default.yaml \
    --epochs 1 --batch_size 32

echo "=== 4. PhysFormer inference (load checkpoint) ==="
python run.py --config configs/physformer_default.yaml --test_only

echo "=== 5. Baseline single epoch (Informer) ==="
python run.py --config configs/baselines/informer.yaml \
    --epochs 1 --batch_size 32

echo "=== 6. Ablation V1 single epoch ==="
python run.py --config configs/physformer_ablation_v1.yaml \
    --epochs 1 --batch_size 32

echo "=== ALL PASSED ==="
