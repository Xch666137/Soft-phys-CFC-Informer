@echo off
set OMP_NUM_THREADS=1
call conda activate rx6800-rocm
python run.py train --config configs/physformer_c23_baseline_local.yaml
