#!/bin/bash
set -e

echo "======================================================="
echo "         PhysFormer Ablation Study Pipeline"
echo "======================================================="

echo
echo "[1/6] Evaluating Full PhysFormer..."
python run.py --config configs/physformer_default.yaml --checkpoint_name "PhysFormer_full_seed2024" --test_only

echo
echo "[2/6] Running Ablation: w/o Physics Stream..."
python run.py --config configs/physformer_ablation_v1.yaml --checkpoint_name "PhysFormer_No_Phys"

echo
echo "[3/6] Running Ablation: w/o PGCC..."
python run.py --config configs/physformer_ablation_v2.yaml --checkpoint_name "PhysFormer_No_PGCC"

echo
echo "[4/6] Running Ablation: w/o Future GLU..."
python run.py --config configs/physformer_ablation_v3.yaml --checkpoint_name "PhysFormer_No_Future_GLU"

echo
echo "[5/6] Running Ablation: w/o Curriculum..."
python run.py --config configs/physformer_ablation_v4.yaml --checkpoint_name "PhysFormer_No_Curriculum"

echo
echo "[6/6] Running Ablation: Fixed Phys..."
python run.py --config configs/physformer_ablation_v5.yaml --checkpoint_name "PhysFormer_Fixed_Phys"

echo
echo "======================================================="
echo "      Generating Ablation Results Table..."
echo "======================================================="
python analysis/collect_ablation_results.py
