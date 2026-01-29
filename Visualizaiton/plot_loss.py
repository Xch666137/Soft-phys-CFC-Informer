import matplotlib.pyplot as plt
import re
import os


def plot_training_logs(log_file_path):
    train_losses = []
    vali_scores = []
    epochs = []

    # 检查文件是否存在
    if not os.path.exists(log_file_path):
        print(f"Error: 文件未找到 - {log_file_path}")
        return

    # 正则表达式匹配 (根据你的日志格式)
    train_pattern = re.compile(r"Train Loss: (\d+\.\d+)")
    vali_pattern = re.compile(r"Vali Score \(Avg NRMSE\): (\d+\.\d+)")

    print(f"正在读取日志: {log_file_path} ...")

    with open(log_file_path, 'r') as f:
        lines = f.readlines()
        current_epoch = 0

        for line in lines:
            # 只有当 Epoch 完成时才计数，保证和指标对齐
            if "Epoch:" in line and "Cost Time:" in line:
                current_epoch += 1

            t_match = train_pattern.search(line)
            if t_match:
                train_losses.append(float(t_match.group(1)))
                # 只有收集到 Train Loss 时才记录 Epoch，保证数据长度一致
                epochs.append(current_epoch)

            v_match = vali_pattern.search(line)
            if v_match:
                vali_scores.append(float(v_match.group(1)))

    # 确保数据长度对齐 (防止日志不完整导致报错)
    min_len = min(len(epochs), len(train_losses), len(vali_scores))
    epochs = epochs[:min_len]
    train_losses = train_losses[:min_len]
    vali_scores = vali_scores[:min_len]

    # --- 绘图开始 ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    plt.title('Training Progress: Loss vs Validation NRMSE')
    plt.grid(True, alpha=0.3)

    # 1. 绘制左侧 Y 轴 (Train Loss) - 蓝色
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss (MSE)', color=color, fontsize=12)
    l1, = ax1.plot(epochs, train_losses, color=color, label='Train Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    # 2. 创建右侧 Y 轴 (Validation Score) - 橙色
    ax2 = ax1.twinx()  # 关键：共享 X 轴
    color = 'tab:orange'
    ax2.set_ylabel('Vali Score (Avg NRMSE)', color=color, fontsize=12)
    l2, = ax2.plot(epochs, vali_scores, color=color, label='Vali NRMSE', linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    # 合并图例
    plt.legend([l1, l2], ['Train Loss (Left)', 'Vali NRMSE (Right)'], loc='upper right')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 使用绝对路径 (请确保这里是你电脑上的真实路径)
    # 记得加 r 防止转义
    log_path = r"E:\Py_program\Soft-phys-CFC-Informer\exp_results\informer\checkpoints\logs\train_log_20251205_160240.txt"

    plot_training_logs(log_path)