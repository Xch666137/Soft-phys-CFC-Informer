#!/bin/bash
# ===============================================================
# run_single_train.sh
# PhysFormer 单次训练启动脚本 (覆盖 full_2024 种子)
# 用法：
#   cd /root/autodl-tmp/Soft-phys-cfc-Informer (根据你的服务器路径修改)
#   chmod +x scripts/run_single_train.sh
#   nohup bash scripts/run_single_train.sh > logs/PhysFormer_full_seed2024.log 2>&1 &
# ===============================================================

set -e  # 任何一步失败立即中止
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

mkdir -p logs

echo "================================================================"
echo " 启动 PhysFormer (物理强约束重构版) 单次训练"
echo " 目标模型: PhysFormer_full_seed2024"
echo " 时间: $(date)"
echo "================================================================"

# 删除旧的 checkpoint / 可视化残余防止冲突
CKPT_DIR="exp_results/PhysFormer/checkpoints/PhysFormer_full_seed2024"
if [ -d "$CKPT_DIR" ]; then
    echo "  [清理] 检测到旧的 Checkpoint 目录 $CKPT_DIR, 正在清理残留的验证文件..."
    rm -f "$CKPT_DIR"/vis_*.npy
    rm -f "$CKPT_DIR"/pred.npy
    rm -f "$CKPT_DIR"/true.npy
    rm -f "$CKPT_DIR"/metrics.npy
fi

# 启动训练
CUDA_VISIBLE_DEVICES=0 python scripts/run_PhysFormer.py \
  --root_path ./ \
  --data_path data/vpp_dataset_3years.csv \
  --seq_len 672 \
  --pred_len 96 \
  --enc_in 6 \
  --d_model 512 \
  --n_heads 8 \
  --e_layers 3 \
  --d_ff 2048 \
  --factor 5 \
  --dropout 0.10 \
  --attn full \
  --embed custom \
  --activation gelu \
  --use_rope \
  --freq h \
  --batch_size 128 \
  --train_epochs 100 \
  --learning_rate 3e-4 \
  --weight_decay 1e-5 \
  --physics_prior_weight 0.05 \
  --grad_clip 1.0 \
  --patience 10 \
  --use_amp 1 \
  --use_gpu 1 \
  --gpu 0 \
  --num_workers 8 \
  --checkpoint_name "PhysFormer_full_seed2024" \
  --is_training 1

echo "================================================================"
echo " 训练环节结束，时间: $(date)"
echo " 请检查 logs/ 下的日志文件查看具体进度。"
echo "================================================================"
