#!/bin/bash
# ===============================================================
# run_full_pipeline.sh
# PhysFormer 完整实验流水线：Full Model → 消融 → 集成
# 用法：
#   cd /root/autodl-tmp/Soft-phys-cfc-Informer
#   chmod +x scripts/run_full_pipeline.sh
#   nohup bash scripts/run_full_pipeline.sh > logs/pipeline.log 2>&1 &
# ===============================================================

set -e  # 任何一步失败立即中止
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

mkdir -p logs

# ---------------------------------------------------------------
# 公共参数（与 run_PhysFormer.py 的 default 值完全一致）
# ---------------------------------------------------------------
COMMON_ARGS="
  --root_path ./
  --data_path data/vpp_dataset_3years.csv
  --seq_len 672
  --pred_len 96
  --enc_in 6
  --d_model 256
  --n_heads 8
  --e_layers 3
  --d_ff 1024
  --factor 5
  --dropout 0.10
  --attn full
  --embed custom
  --activation gelu
  --use_rope
  --freq h
  --batch_size 32
  --train_epochs 100
  --learning_rate 1e-4
  --weight_decay 1e-5
  --physics_prior_weight 0.05
  --grad_clip 1.0
  --patience 10
  --use_amp 1
  --use_gpu 1
  --gpu 0
  --num_workers 8
"

echo "================================================================"
echo " PhysFormer 完整实验流水线 开始"
echo " 时间: $(date)"
echo "================================================================"

# ===============================================================
# Full Model + 4 个消融变体，分配到 GPU 0-4 并行执行
# ===============================================================
echo ""
echo ">>> Full Model + 消融实验 (5 路 GPU 并行) <<<"
echo "================================================================"

run_experiment() {
    local name=$1
    local gpu_id=$2
    local extra_flags=$3
    echo ""
    echo "--- 实验: $name (GPU $gpu_id) ---"
    # 清理旧的可视化文件，防止数据残留
    local ckpt_dir="exp_results/PhysFormer/checkpoints/${name}"
    if [ -d "$ckpt_dir" ]; then
        rm -f "$ckpt_dir"/vis_*.npy
        echo "  [清理] 已删除 $ckpt_dir 中的旧可视化文件"
    fi
    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/run_PhysFormer.py \
      $COMMON_ARGS \
      --gpu 0 \
      --checkpoint_name "${name}" \
      --is_training 1 \
      $extra_flags \
      > "logs/${name}.log" 2>&1
    echo "[$name] 完成 (GPU $gpu_id)"
}

# GPU 0: Full Model
run_experiment "PhysFormer_full_seed2024"       0 ""                             &
PID_FULL=$!

# GPU 1-4: 4 个消融变体
run_experiment "PhysFormer_ablation_V1_no_phys"       1 "--ablation_no_phys_stream"  &
PID_V1=$!
run_experiment "PhysFormer_ablation_V2_no_pgcc"       2 "--ablation_no_pgcc"         &
PID_V2=$!
run_experiment "PhysFormer_ablation_V3_no_future_glu" 3 "--ablation_no_future_glu"   &
PID_V3=$!
run_experiment "PhysFormer_ablation_V4_no_curriculum" 4 "--ablation_no_curriculum"    &
PID_V4=$!

echo ""
echo "已启动 5 路并行实验:"
echo "  GPU 0: Full Model       (PID: $PID_FULL)"
echo "  GPU 1: V1_no_phys       (PID: $PID_V1)"
echo "  GPU 2: V2_no_pgcc       (PID: $PID_V2)"
echo "  GPU 3: V3_no_future_glu (PID: $PID_V3)"
echo "  GPU 4: V4_no_curriculum (PID: $PID_V4)"
echo "  等待全部完成..."

# 等待全部后台任务完成
FAILED=0
for PID in $PID_FULL $PID_V1 $PID_V2 $PID_V3 $PID_V4; do
    wait $PID || FAILED=$((FAILED + 1))
done

if [ $FAILED -gt 0 ]; then
    echo "[警告] 有 $FAILED 个实验失败，请检查 logs/*.log"
else
    echo "全部实验完成"
fi

echo ""
echo "================================================================"
echo " 实验完成！时间: $(date)"
echo "================================================================"
echo ""
echo " 日志目录: logs/"
echo " Checkpoint 目录: exp_results/PhysFormer/checkpoints/"
echo ""
echo " 汇总报告: python scripts/collect_ablation_results.py"

# ===============================================================
# 第三阶段：多 Seed 集成（暂时停用）
# ===============================================================
# echo ""
# echo ">>> [阶段 3] 多 Seed 集成训练 (seeds: 2024, 2025, 2026) <<<"
# echo "================================================================"
#
# for SEED in 2024 2025 2026; do
#     echo ""
#     echo "--- Seed: $SEED ---"
#     python scripts/run_PhysFormer.py \
#       $COMMON_ARGS \
#       --checkpoint_name "PhysFormer_ensemble_seed${SEED}" \
#       --is_training 1 \
#       2>&1 | tee "logs/ensemble_seed${SEED}.log"
#     echo "[Seed $SEED] 完成"
# done
