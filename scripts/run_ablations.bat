@echo off
setlocal
echo =======================================================
echo          PhysFormer Ablation Study Pipeline
echo =======================================================

echo.
echo [1/6] Evaluating Full PhysFormer (Baseline)...
echo.
python scripts\run_PhysFormer.py --is_training 0 --checkpoint_name "PhysFormer_ensemble_seed2024"

echo.
echo =======================================================
echo [2/6] Running Ablation: w/o Physics Stream...
echo =======================================================
python scripts\run_PhysFormer.py --is_training 1 --checkpoint_name "PhysFormer_No_Phys" --ablation_no_phys_stream

echo.
echo =======================================================
echo [3/6] Running Ablation: w/o PGCC...
echo =======================================================
python scripts\run_PhysFormer.py --is_training 1 --checkpoint_name "PhysFormer_No_PGCC" --ablation_no_pgcc

echo.
echo =======================================================
echo [4/6] Running Ablation: w/o Future GLU...
echo =======================================================
python scripts\run_PhysFormer.py --is_training 1 --checkpoint_name "PhysFormer_No_Future_GLU" --ablation_no_future_glu

echo.
echo =======================================================
echo [5/6] Running Ablation: w/o Curriculum...
echo =======================================================
python scripts\run_PhysFormer.py --is_training 1 --checkpoint_name "PhysFormer_No_Curriculum" --ablation_no_curriculum

echo.
echo =======================================================
echo [6/6] Running Ablation: Fixed Thresholds (No Data-Driven Fine-Tuning)...
echo =======================================================
python scripts\run_PhysFormer.py --is_training 1 --checkpoint_name "PhysFormer_Fixed_Phys" --ablation_fixed_phys

echo.
echo =======================================================
echo       All Ablation Experiments Finished!
echo =======================================================

echo.
echo =======================================================
echo       Generating Ablation Results Table...
echo =======================================================
python scripts\collect_ablation_results.py
endlocal
pause
