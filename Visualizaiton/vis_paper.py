import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as font_manager


# ==========================================
# 1. IEEE Transactions 绘图风格配置
# ==========================================
def set_ieee_style():
    """配置 Matplotlib 以符合 IEEE Transactions 标准"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,  # 正文字号
        'axes.labelsize': 12,  # 轴标签字号
        'axes.titlesize': 12,  # 标题字号
        'legend.fontsize': 10,  # 图例字号
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'in',  # 刻度朝内
        'ytick.direction': 'in',
        'lines.linewidth': 1.5,  # 线宽
        'axes.grid': True,  # 开启网格
        'grid.alpha': 0.3,  # 网格透明度
        'grid.linestyle': '--',  # 网格样式
        'figure.constrained_layout.use': True,  # 自动布局
        'savefig.dpi': 300,  # 保存分辨率
        'savefig.bbox': 'tight'  # 去除白边
    })


# 定义 IEEE 推荐的高对比度配色 (适合黑白打印)
COLORS = {
    'pred': '#D7191C',  # 预测值 (红色)
    'true': '#2C7BB6',  # 真实值 (蓝色)
    'gate': '#1A9641',  # Gate值 (绿色)
    'phys': '#FDAE61',  # 物理环境 (橙色/填充)
    'net': '#404040'  # 净负荷 (深灰)
}


# ==========================================
# 2. 数据加载器
# ==========================================
def load_data(folder_path):
    """加载预测结果、真实值以及Gate可视化数据"""
    print(f"Loading data from: {folder_path}")

    data = {}
    try:
        # 基础预测数据
        data['pred'] = np.load(os.path.join(folder_path, 'pred.npy'))
        data['true'] = np.load(os.path.join(folder_path, 'true.npy'))

        # Gate 可视化数据 (如果存在)
        # 注意：这里假设文件名与 exp_PhysFormer.py 中保存的一致
        if os.path.exists(os.path.join(folder_path, 'vis_gate_pv.npy')):
            data['gate_pv'] = np.load(os.path.join(folder_path, 'vis_gate_pv.npy'))
            data['gate_wind'] = np.load(os.path.join(folder_path, 'vis_gate_wind.npy'))
            data['irr'] = np.load(os.path.join(folder_path, 'vis_irr.npy'))
            data['speed'] = np.load(os.path.join(folder_path, 'vis_speed.npy'))
            print(">> Gate visualization data loaded successfully.")
        else:
            print("!! Warning: Gate visualization data not found. Skipping Gate plots.")

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    return data


# ==========================================
# 3. 绘图函数：VPP 全局时序预测 (含净负荷)
# ==========================================
def plot_vpp_forecast(data, sample_idx=0, save_path='./'):
    """
    绘制 Load, PV, Wind 以及 Net Load 的预测对比图
    """
    preds = data['pred']
    trues = data['true']

    # 如果数据是 [B, Seq, D]，取指定样本
    if len(preds.shape) == 3:
        pred_sample = preds[sample_idx]  # [Seq, 3]
        true_sample = trues[sample_idx]
    else:
        # 如果是拼接好的长序列，取前96个点演示
        pred_sample = preds[:96]
        true_sample = trues[:96]

    # 计算净负荷 (Net Load = Load - PV - Wind)
    # 假设列顺序: 0:Load, 1:PV, 2:Wind
    net_pred = pred_sample[:, 0] - pred_sample[:, 1] - pred_sample[:, 2]
    net_true = true_sample[:, 0] - true_sample[:, 1] - true_sample[:, 2]

    seq_len = pred_sample.shape[0]
    time_steps = np.arange(seq_len)

    # 创建 2x2 子图
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Load
    ax = axs[0, 0]
    ax.plot(time_steps, true_sample[:, 0], color=COLORS['true'], label='Ground Truth', linestyle='-')
    ax.plot(time_steps, pred_sample[:, 0], color=COLORS['pred'], label='Prediction', linestyle='--')
    ax.set_title('(a) Load Forecast', loc='left', fontweight='bold')
    ax.set_ylabel('Power (MW)')
    ax.legend(loc='upper right', frameon=False)

    # 2. PV
    ax = axs[0, 1]
    ax.plot(time_steps, true_sample[:, 1], color=COLORS['true'], linestyle='-')
    ax.plot(time_steps, pred_sample[:, 1], color=COLORS['pred'], linestyle='--')
    ax.set_title('(b) PV Power Forecast', loc='left', fontweight='bold')

    # 3. Wind
    ax = axs[1, 0]
    ax.plot(time_steps, true_sample[:, 2], color=COLORS['true'], linestyle='-')
    ax.plot(time_steps, pred_sample[:, 2], color=COLORS['pred'], linestyle='--')
    ax.set_title('(c) Wind Power Forecast', loc='left', fontweight='bold')
    ax.set_xlabel('Time Steps (15 min)')
    ax.set_ylabel('Power (MW)')

    # 4. Net Load (VPP 关键指标)
    ax = axs[1, 1]
    ax.plot(time_steps, net_true, color='black', alpha=0.6, label='Net Truth', linestyle='-')
    ax.plot(time_steps, net_pred, color=COLORS['net'], label='Net Pred', linestyle='--')
    # 填充误差区域
    ax.fill_between(time_steps, net_true, net_pred, color='gray', alpha=0.2, label='Error')
    ax.set_title('(d) VPP Net Load (Aggregation)', loc='left', fontweight='bold')
    ax.set_xlabel('Time Steps (15 min)')
    ax.legend(loc='upper right', frameon=False)

    plt.savefig(os.path.join(save_path, 'IEEE_VPP_Forecast.pdf'))
    plt.savefig(os.path.join(save_path, 'IEEE_VPP_Forecast.png'))
    print(">> VPP Forecast plot saved.")
    plt.close()


# ==========================================
# 4. 绘图函数：PhysFormer 核心机制 (Gate vs Physics)
# ==========================================
def plot_gate_mechanism(data, sample_idx=0, save_path='./'):
    """
    绘制 Gate 开启程度与物理环境因子的双轴对比图
    """
    if 'gate_pv' not in data:
        return

    # 提取数据
    gate_pv = data['gate_pv'][sample_idx]  # [Seq]
    gate_wind = data['gate_wind'][sample_idx]  # [Seq]
    irr = data['irr'][sample_idx]  # [Seq]
    speed = data['speed'][sample_idx]  # [Seq]

    seq_len = len(gate_pv)
    time_steps = np.arange(seq_len)

    # 创建 2x1 子图
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Subplot 1: PV Gate vs Irradiance ---
    # 右轴：物理环境 (Irradiance) - 作为背景
    ax2 = ax1.twinx()
    ln2 = ax2.fill_between(time_steps, 0, irr, color='#FDAE61', alpha=0.4, label='Irradiance (Normalized)')
    ax2.set_ylabel('Irradiance Intensity', color='#E69500')
    ax2.tick_params(axis='y', labelcolor='#E69500')
    ax2.set_ylim(0, max(irr) * 1.2)  # 留出顶部空间

    # 左轴：Gate 值 - 作为前景
    ln1 = ax1.plot(time_steps, gate_pv, color=COLORS['gate'], linewidth=2, label='PV Gate ($g_{pv}$)')
    ax1.set_ylabel('Gate Activation $\in [0, 1]$', color=COLORS['gate'])
    ax1.tick_params(axis='y', labelcolor=COLORS['gate'])
    ax1.set_ylim(-0.1, 1.1)

    # 标题与图例
    ax1.set_title('(a) PV Mechanism: Gate Response to Irradiance', loc='left', fontweight='bold')

    # 合并图例 (Matplotlib 双轴图例合并技巧)
    # fill_between 返回的是 PolyCollection，不能直接用于 legend，需要创建一个 proxy
    import matplotlib.patches as mpatches
    patch_irr = mpatches.Patch(color='#FDAE61', alpha=0.4, label='Irradiance')
    lines = ln1 + [patch_irr]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=False)

    # --- Subplot 2: Wind Gate vs Wind Speed ---
    # 右轴：物理环境 (Wind Speed)
    ax4 = ax3.twinx()
    ln4 = ax4.fill_between(time_steps, 0, speed, color='#abd9e9', alpha=0.5, label='Wind Speed')
    ax4.plot(time_steps, speed, color='#2c7bb6', alpha=0.3, linewidth=1)  # 增加一条细线轮廓
    ax4.set_ylabel('Wind Speed (Normalized)', color='#2c7bb6')
    ax4.tick_params(axis='y', labelcolor='#2c7bb6')

    # 左轴：Gate 值
    ln3 = ax3.plot(time_steps, gate_wind, color=COLORS['gate'], linewidth=2, label='Wind Gate ($g_{wind}$)')
    ax3.set_ylabel('Gate Activation $\in [0, 1]$', color=COLORS['gate'])
    ax3.tick_params(axis='y', labelcolor=COLORS['gate'])
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_xlabel('Time Steps (15 min)')

    ax3.set_title('(b) Wind Mechanism: Gate Response to Wind Speed', loc='left', fontweight='bold')

    # 合并图例
    patch_wind = mpatches.Patch(color='#abd9e9', alpha=0.5, label='Wind Speed')
    lines_w = ln3 + [patch_wind]
    labels_w = [l.get_label() for l in lines_w]
    ax3.legend(lines_w, labels_w, loc='upper left', frameon=False)

    plt.savefig(os.path.join(save_path, 'IEEE_PhysFormer_Mechanism.pdf'))
    plt.savefig(os.path.join(save_path, 'IEEE_PhysFormer_Mechanism.png'))
    print(">> Gate Mechanism plot saved.")
    plt.close()


# ==========================================
# 主程序
# ==========================================
if __name__ == '__main__':
    # 设置风格
    set_ieee_style()

    # 路径配置 (请根据实际情况修改 experiment_name)
    # 假设脚本放在项目根目录下
    experiment_name = 'PhysFormer_experiment_v1.0'
    results_path = f'exp_results/PhysFormer/checkpoints/{experiment_name}/'

    # 加载数据
    data = load_data(results_path)

    if data is not None:
        # 1. 绘制 VPP 预测总览
        plot_vpp_forecast(data, sample_idx=0, save_path=results_path)

        # 2. 绘制 PhysFormer 机制图 (核心图表)
        plot_gate_mechanism(data, sample_idx=0, save_path=results_path)

        print("\nAll plots generated successfully following IEEE style.")