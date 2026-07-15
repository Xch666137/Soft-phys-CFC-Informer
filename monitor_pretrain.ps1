# Monitor PhysFormer pretraining on AutoDL every 30 minutes
# Reports: epoch/step, loss, running status, errors

$SSH_HOST = "AutoDL-vGPU-32GB"
$LOG_FILE = "/root/autodl-tmp/physformer/logs/pretrain_pv_masking.log"
$INTERVAL = 1800  # 30 minutes in seconds
$CHECK_NUM = 0

Write-Host "=== PhysFormer Pretrain Monitor Started at $(Get-Date) ==="
Write-Host "Checking every 30 minutes until epoch 50/50 or failure"
Write-Host ""

while ($true) {
    $CHECK_NUM++
    Write-Host "=========================================="
    Write-Host "CHECK #$CHECK_NUM - $(Get-Date)"
    Write-Host "=========================================="
    
    # 1. Check if training process is running
    Write-Host "--- Process Status ---"
    $procInfo = ssh $SSH_HOST 'ps aux | grep "run.py pretrain" | grep -v grep | head -1'
    if ([string]::IsNullOrWhiteSpace($procInfo)) {
        Write-Host "WARNING: Training process NOT FOUND! Training may have stopped."
        Write-Host "ALERT: Training stopped unexpectedly at check #$CHECK_NUM"
        Write-Host "--- Last 30 lines of log ---"
        $lastLog = ssh $SSH_HOST "tail -30 $LOG_FILE"
        Write-Host $lastLog
        Write-Host ""
        Write-Host "=== Monitor exiting: process not running ==="
        exit 1
    } else {
        $pid = ($procInfo -split '\s+')[1]
        Write-Host "Training process is RUNNING (PID: $pid)"
    }
    
    # 2. Check training log
    Write-Host ""
    Write-Host "--- Training Log (last 20 lines) ---"
    $logTail = ssh $SSH_HOST "tail -20 $LOG_FILE"
    Write-Host $logTail
    
    # 3. Extract current epoch and loss
    Write-Host ""
    Write-Host "--- Progress Summary ---"
    $lastEpochLine = ssh $SSH_HOST "grep '^Epoch:' $LOG_FILE | tail -1"
    if (-not [string]::IsNullOrWhiteSpace($lastEpochLine)) {
        Write-Host "Last completed epoch: $lastEpochLine"
    }
    
    $currentProgress = ssh $SSH_HOST "grep -E '^Epoch [0-9]+/[0-9]+ \[' $LOG_FILE | tail -1"
    if (-not [string]::IsNullOrWhiteSpace($currentProgress)) {
        Write-Host "Current step: $currentProgress"
    }
    
    # 4. Check for errors
    Write-Host ""
    Write-Host "--- Error/Warning Check ---"
    $errors = ssh $SSH_HOST "grep -i -E 'error|exception|traceback|killed|oom|out of memory' $LOG_FILE | tail -5"
    if (-not [string]::IsNullOrWhiteSpace($errors)) {
        Write-Host "ALERT: Errors found in log:"
        Write-Host $errors
    } else {
        Write-Host "No errors detected in log."
    }
    
    # 5. Check if training completed (epoch 50/50)
    $completion = ssh $SSH_HOST "grep 'Epoch: 50' $LOG_FILE" 2>$null
    if (-not [string]::IsNullOrWhiteSpace($completion)) {
        Write-Host ""
        Write-Host "*** TRAINING COMPLETE: Epoch 50/50 reached! ***"
        Write-Host "Final epoch info: $completion"
        Write-Host ""
        Write-Host "--- Last 30 lines of log ---"
        $finalLog = ssh $SSH_HOST "tail -30 $LOG_FILE"
        Write-Host $finalLog
        Write-Host ""
        Write-Host "=== Monitor exiting: training completed successfully ==="
        exit 0
    }
    
    # 6. GPU status
    Write-Host ""
    Write-Host "--- GPU Status ---"
    $gpuInfo = ssh $SSH_HOST 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'
    Write-Host $gpuInfo
    
    Write-Host ""
    Write-Host "Next check in 30 minutes..."
    Write-Host ""
    
    Start-Sleep -Seconds $INTERVAL
}
