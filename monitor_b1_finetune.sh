#!/bin/bash
# B1 finetune 进度监控 — 3 seed 同步 epoch 后汇总汇报
# 用法: bash monitor_b1_finetune.sh

HOST="root@connect.westd.seetacloud.com"
PORT="16846"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $PORT"
RUNS_DIR="/root/autodl-tmp/physformer/runs"
SEEDS="2025 2026 2027"
INTERVAL=60
MAX_EPOCH=50

# 动态获取起始最小 epoch
init_result=$(ssh $SSH_OPTS "$HOST" \
  "for s in $SEEDS; do \
     grep 'Epoch:' ${RUNS_DIR}/physformer_igt_b1_finetune_s\${s}/train.log 2>/dev/null | tail -1; \
   done" 2>/dev/null)

last_min_epoch=999
while read -r line; do
  [ -z "$line" ] && continue
  ep=$(echo "$line" | sed -n 's/.*Epoch: *\([0-9]*\).*/\1/p')
  [ -n "$ep" ] && [ "$ep" -lt "$last_min_epoch" ] && last_min_epoch=$ep
done <<< "$init_result"
[ "$last_min_epoch" -eq 999 ] && last_min_epoch=0

echo "[monitor] B1 finetune 监控启动，间隔=${INTERVAL}s，起始追踪 epoch=$last_min_epoch"
echo "[monitor] 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo

while true; do
  # 一次 SSH 取所有 seed 的最新 epoch summary 行
  result=$(ssh $SSH_OPTS "$HOST" \
    "for s in $SEEDS; do \
       grep 'Epoch:' ${RUNS_DIR}/physformer_igt_b1_finetune_s\${s}/train.log 2>/dev/null | tail -1; \
     done" 2>/dev/null)

  if [ -z "$result" ]; then
    echo "[monitor] $(date '+%H:%M:%S') SSH 失败，跳过本轮"
    sleep $INTERVAL
    continue
  fi

  # 用 sed 解析（不用 grep -P，避免 locale 问题）
  ep2025=$(echo "$result" | sed -n '1s/.*Epoch: *\([0-9]*\).*/\1/p')
  ep2026=$(echo "$result" | sed -n '2s/.*Epoch: *\([0-9]*\).*/\1/p')
  ep2027=$(echo "$result" | sed -n '3s/.*Epoch: *\([0-9]*\).*/\1/p')

  vmse2025=$(echo "$result" | sed -n '1s/.*Val MSE: *\([0-9.]*\).*/\1/p')
  vmse2026=$(echo "$result" | sed -n '2s/.*Val MSE: *\([0-9.]*\).*/\1/p')
  vmse2027=$(echo "$result" | sed -n '3s/.*Val MSE: *\([0-9.]*\).*/\1/p')

  es2025=$(echo "$result" | sed -n '1s/.*EarlyStopping counter: *\([0-9]*\).*/\1/p')
  es2026=$(echo "$result" | sed -n '2s/.*EarlyStopping counter: *\([0-9]*\).*/\1/p')
  es2027=$(echo "$result" | sed -n '3s/.*EarlyStopping counter: *\([0-9]*\).*/\1/p')

  tl2025=$(echo "$result" | sed -n '1s/.*Train: *\([0-9.]*\).*/\1/p')
  tl2026=$(echo "$result" | sed -n '2s/.*Train: *\([0-9.]*\).*/\1/p')
  tl2027=$(echo "$result" | sed -n '3s/.*Train: *\([0-9.]*\).*/\1/p')

  mse2025=$(echo "$result" | sed -n '1s/.*Val MSE(\([^)]*\)): *\([0-9.e+\-]*\).*/\2/p')
  mse2026=$(echo "$result" | sed -n '2s/.*Val MSE(\([^)]*\)): *\([0-9.e+\-]*\).*/\2/p')
  mse2027=$(echo "$result" | sed -n '3s/.*Val MSE(\([^)]*\)): *\([0-9.e+\-]*\).*/\2/p')

  # 当前最小 epoch
  min_ep=$ep2025
  [ "$ep2026" -lt "$min_ep" ] && min_ep=$ep2026
  [ "$ep2027" -lt "$min_ep" ] && min_ep=$ep2027

  if [ "$min_ep" -gt "$last_min_epoch" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ✓ Epoch $min_ep 全部完成  |  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "═══════════════════════════════════════════════════════════════"
    printf "  %-6s  %10s  %12s  %13s  %s\n" "Seed" "Train Loss" "Val MSE" "MSE(MW²)" "ES"
    printf "  %-6s  %10s  %12s  %13s  %s\n" "------" "----------" "------------" "-------------" "----"
    printf "  s2025   %10s  %12s  %13s  %s/12\n" "$tl2025" "$vmse2025" "$mse2025" "$es2025"
    printf "  s2026   %10s  %12s  %13s  %s/12\n" "$tl2026" "$vmse2026" "$mse2026" "$es2026"
    printf "  s2027   %10s  %12s  %13s  %s/12\n" "$tl2027" "$vmse2027" "$mse2027" "$es2027"
    # 取最优 Val MSE
    best=$(printf '%s\n' "$vmse2025" "$vmse2026" "$vmse2027" | sort -n | head -1)
    echo "  ───────────────────────────────────────────────────────────"
    echo "  Best Val MSE: $best"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    last_min_epoch=$min_ep
  else
    # 安静模式：只打印一行进度
    printf "[monitor] %s  s2025=%s/%s s2026=%s/%s s2027=%s/%s  (等待全部完成 epoch %s)\n" \
      "$(date '+%H:%M:%S')" "$ep2025" "$MAX_EPOCH" "$ep2026" "$MAX_EPOCH" "$ep2027" "$MAX_EPOCH" "$((last_min_epoch + 1))"
  fi

  # 全部完成？
  if [ "$ep2025" -ge "$MAX_EPOCH" ] && [ "$ep2026" -ge "$MAX_EPOCH" ] && [ "$ep2027" -ge "$MAX_EPOCH" ]; then
    echo ""
    echo "[monitor] ========== 全部 epoch 完成！=========="
    echo "[monitor] $(date '+%Y-%m-%d %H:%M:%S')"
    break
  fi

  sleep $INTERVAL
done
