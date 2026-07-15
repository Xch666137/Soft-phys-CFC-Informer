#!/bin/bash
# Monitor PhysFormer pretraining on AutoDL every 30 minutes
# Reports: epoch/step, loss, running status, errors

LOG_FILE="/root/autodl-tmp/physformer/logs/pretrain_pv_masking.log"
SSH_HOST="AutoDL-vGPU-32GB"
INTERVAL=1800  # 30 minutes in seconds
CHECK_NUM=0

echo "=== PhysFormer Pretrain Monitor Started at $(date) ==="
echo "Checking every 30 minutes until epoch 50/50 or failure"
echo ""

while true; do
    CHECK_NUM=$((CHECK_NUM + 1))
    echo "=========================================="
    echo "CHECK #$CHECK_NUM - $(date)"
    echo "=========================================="
    
    # 1. Check if training process is running
    echo "--- Process Status ---"
    PROC_INFO=$(ssh $SSH_HOST 'ps aux | grep "run.py pretrain" | grep -v grep | head -1')
    if [ -z "$PROC_INFO" ]; then
        echo "WARNING: Training process NOT FOUND! Training may have stopped."
        echo "ALERT: Training stopped unexpectedly at check #$CHECK_NUM"
        # Check last log lines for completion or error
        echo "--- Last 30 lines of log ---"
        ssh $SSH_HOST "tail -30 $LOG_FILE"
        echo ""
        echo "=== Monitor exiting: process not running ==="
        exit 1
    else
        echo "Training process is RUNNING (PID: $(echo $PROC_INFO | awk '{print $2}'))"
    fi
    
    # 2. Check training log
    echo ""
    echo "--- Training Log (last 20 lines) ---"
    LOG_TAIL=$(ssh $SSH_HOST "tail -20 $LOG_FILE")
    echo "$LOG_TAIL"
    
    # 3. Extract current epoch and loss
    echo ""
    echo "--- Progress Summary ---"
    LAST_EPOCH_LINE=$(ssh $SSH_HOST "grep '^Epoch:' $LOG_FILE | tail -1")
    if [ -n "$LAST_EPOCH_LINE" ]; then
        echo "Last completed epoch: $LAST_EPOCH_LINE"
    fi
    
    CURRENT_PROGRESS=$(ssh $SSH_HOST "grep -E '^Epoch [0-9]+/[0-9]+ \[' $LOG_FILE | tail -1")
    if [ -n "$CURRENT_PROGRESS" ]; then
        echo "Current step: $CURRENT_PROGRESS"
    fi
    
    # 4. Check for errors
    echo ""
    echo "--- Error/Warning Check ---"
    ERRORS=$(ssh $SSH_HOST "grep -i -E 'error|exception|traceback|killed|oom|out of memory' $LOG_FILE | tail -5")
    if [ -n "$ERRORS" ]; then
        echo "ALERT: Errors found in log:"
        echo "$ERRORS"
    else
        echo "No errors detected in log."
    fi
    
    # 5. Check if training completed (epoch 50/50)
    COMPLETION=$(ssh $SSH_HOST "grep 'Epoch: 50' $LOG_FILE" 2>/dev/null)
    if [ -n "$COMPLETION" ]; then
        echo ""
        echo "*** TRAINING COMPLETE: Epoch 50/50 reached! ***"
        echo "Final epoch info: $COMPLETION"
        echo ""
        echo "--- Last 30 lines of log ---"
        ssh $SSH_HOST "tail -30 $LOG_FILE"
        echo ""
        echo "=== Monitor exiting: training completed successfully ==="
        exit 0
    fi
    
    # 6. GPU status
    echo ""
    echo "--- GPU Status ---"
    ssh $SSH_HOST 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'
    
    echo ""
    echo "Next check in 30 minutes..."
    echo ""
    
    sleep $INTERVAL
done
