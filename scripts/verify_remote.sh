#!/bin/bash
# Remote server verification — run on GPU server after refactoring
set -e

echo "=== 1. Install package ==="
pip install -e .

echo "=== 2. Import verification ==="
python verify_imports.py

echo "=== 3. Config loading test ==="
python run.py --config configs/physformer_default.yaml --print_config

echo "=== 4. PhysFormer single epoch training ==="
python run.py --config configs/physformer_default.yaml \
    --epochs 1 --batch_size 32

echo "=== 5. PhysFormer inference (load checkpoint) ==="
python run.py --config configs/physformer_default.yaml --test_only

echo "=== 6. Baseline single epoch (Informer) ==="
python run.py --config configs/baselines/informer.yaml \
    --epochs 1 --batch_size 32

echo "=== 7. Ablation V1 single epoch ==="
python run.py --config configs/physformer_ablation_v1.yaml \
    --epochs 1 --batch_size 32

echo "=== ALL PASSED ==="
