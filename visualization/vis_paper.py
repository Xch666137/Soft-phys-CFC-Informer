import numpy as np
import matplotlib.pyplot as plt
import os
import json
import matplotlib.font_manager as font_manager

from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm


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
# 5. 绘图函数：净负荷相关性散点图 (Hexbin)
# ==========================================
def plot_net_load_correlation(data, save_path='./'):
    """
    绘制真实净负荷 vs 预测净负荷的 Hexbin 密度散点图
    证明模型在 VPP 总出口处的预测无偏性
    """
    preds = data['pred'].reshape(-1, 3)
    trues = data['true'].reshape(-1, 3)

    net_pred = preds[:, 0] - preds[:, 1] - preds[:, 2]
    net_true = trues[:, 0] - trues[:, 1] - trues[:, 2]

    fig, ax = plt.subplots(figsize=(6, 5))

    # 使用 Hexbin 绘制高密度散点图
    hb = ax.hexbin(net_true, net_pred, gridsize=50, cmap='Blues',
                   mincnt=1, norm=LogNorm())  # LogNorm 凸显低密度区域的异常点
    cb = fig.colorbar(hb, ax=ax, label='Density (log scale)')

    # 绘制 y=x 理想基准线
    min_val = min(net_true.min(), net_pred.min())
    max_val = max(net_true.max(), net_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal ($y=x$)')

    # 计算 R^2 和 RMSE
    correlation_matrix = np.corrcoef(net_true, net_pred)
    r2 = correlation_matrix[0, 1] ** 2
    rmse = np.sqrt(np.mean((net_true - net_pred) ** 2))

    # 在图中标注指标
    text_str = f'$R^2$ = {r2:.4f}\nRMSE = {rmse:.2f} MW'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.set_title('Net Load Prediction Correlation', fontweight='bold')
    ax.set_xlabel('True Net Load (MW)')
    ax.set_ylabel('Predicted Net Load (MW)')
    ax.legend(loc='lower right', frameon=False)

    plt.savefig(os.path.join(save_path, 'IEEE_Correlation_Hexbin.pdf'))
    plt.savefig(os.path.join(save_path, 'IEEE_Correlation_Hexbin.png'))
    print(">> Correlation Hexbin plot saved.")
    plt.close()


# ==========================================
# 6. 绘图函数：物理爬坡率合规性分布 (Ramp Rate PDF)
# ==========================================
def plot_ramp_compliance(data, ramp_limits=[5.0, 2.0, 3.0], save_path='./'):
    """
    绘制预测值的爬坡率概率密度分布，叠加物理极限红线
    需要根据你的数据集实际情况传入 ramp_limits (MW/step)
    """
    preds = data['pred'].reshape(-1, 3)  # 展平为连续序列

    # 计算一阶差分 (爬坡率)
    diff_pred = preds[1:] - preds[:-1]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    titles = ['Load Ramp Rate', 'PV Ramp Rate', 'Wind Ramp Rate']
    colors = ['#404040', '#E69500', '#2C7BB6']

    for i in range(3):
        ax = axs[i]
        var_diff = diff_pred[:, i]
        limit = ramp_limits[i]

        # 使用 KDE 绘制平滑的概率密度曲线
        kde = gaussian_kde(var_diff)
        x_range = np.linspace(var_diff.min() - 1, var_diff.max() + 1, 500)
        ax.plot(x_range, kde(x_range), color=colors[i], lw=2)
        ax.fill_between(x_range, 0, kde(x_range), color=colors[i], alpha=0.3)

        # 绘制物理极限红线
        ax.axvline(x=limit, color='red', linestyle='--', lw=1.5, label='Upper Phys-Limit')
        ax.axvline(x=-limit, color='red', linestyle='--', lw=1.5, label='Lower Phys-Limit')

        # 统计越限率
        violation_rate = np.sum(np.abs(var_diff) > limit) / len(var_diff) * 100
        ax.set_title(f'({chr(97 + i)}) {titles[i]}\nViolations: {violation_rate:.2f}%', fontweight='bold')
        ax.set_xlabel('$\Delta P$ (MW / 15min)')
        if i == 0:
            ax.set_ylabel('Density')
        ax.legend(loc='upper right', frameon=False)

    plt.savefig(os.path.join(save_path, 'IEEE_Physical_Compliance.pdf'))
    print(">> Physical Compliance plot saved.")
    plt.close()


# ==========================================
# 7. 绘图函数：课程学习门控演化 (Curriculum Gate Evolution)
# ==========================================
def plot_curriculum_evolution(save_path='./'):
    """
    读取 gate_history.npy，绘制训练过程中 Gate 的演化
    展示物理约束是如何“逐步放权”给数据驱动的
    """
    history_file = os.path.join(save_path, 'gate_history.npy')
    if not os.path.exists(history_file):
        print("!! Warning: gate_history.npy not found. Skipping curriculum plot.")
        return

    history = np.load(history_file, allow_pickle=True).item()
    epochs = history['epoch']

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(epochs, history['pv'], color='#E69500', marker='o', markersize=4, label='PV Gate Avg')
    ax.plot(epochs, history['wind'], color='#2C7BB6', marker='s', markersize=4, label='Wind Gate Avg')
    ax.plot(epochs, history['load'], color='#404040', marker='^', markersize=4, label='Load Gate Avg')

    # 标注课程学习的三个阶段 (对应代码中的 stage1, stage2, stage3)
    ax.axvspan(0, 5, color='gray', alpha=0.1, label='Stage 1: Frozen Phys Prior')
    ax.axvspan(5, 15, color='yellow', alpha=0.1, label='Stage 2: Thaw & Relax')
    ax.axvspan(15, max(epochs), color='green', alpha=0.05, label='Stage 3: Full Data-Driven')

    ax.set_title('Gate Evolution During Curriculum Learning', fontweight='bold')
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Average Gate Activation')
    ax.set_ylim(0, 1.05)

    # 将图例放在外面防止遮挡数据
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    plt.savefig(os.path.join(save_path, 'IEEE_Curriculum_Evolution.pdf'), bbox_inches='tight')
    print(">> Curriculum Evolution plot saved.")
    plt.close()


# ==========================================
# 8. 绘图函数：极端工况柱状图 (Extreme Scenarios)
# ==========================================
def plot_extreme_scenarios_bar(save_path='./'):
    """
    读取 extreme_scenarios/scenario_metrics.json 绘制表现
    """
    json_path = os.path.join(save_path, 'extreme_scenarios', 'scenario_metrics.json')
    if not os.path.exists(json_path):
        print("!! Warning: scenario_metrics.json not found. Skipping extreme scenario plot.")
        return

    with open(json_path, 'r') as f:
        metrics = json.load(f)

    # 准备数据
    scenarios = [k for k in metrics.keys() if metrics[k]['sample_count'] > 0 and k != 'irr_drop']
    scenarios = ['high_temp', 'high_wind', 'low_irr', 'high_temp_wind', 'low_irr_high_temp']
    # 过滤掉不存在或样本为0的场景
    scenarios = [s for s in scenarios if s in metrics and metrics[s]['sample_count'] > 0]

    if not scenarios:
        return

    maes = [metrics[s]['mae'] for s in scenarios]
    rmses = [metrics[s]['rmse'] for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))

    rects1 = ax.bar(x - width / 2, maes, width, label='MAE', color='#2C7BB6', edgecolor='black')
    rects2 = ax.bar(x + width / 2, rmses, width, label='RMSE', color='#D7191C', edgecolor='black')

    # 格式化 X 轴标签，使其更易读
    display_names = {
        'high_temp': 'High\nTemp',
        'high_wind': 'High\nWind',
        'low_irr': 'Low\nIrradiance',
        'high_temp_wind': 'Temp +\nWind',
        'low_irr_high_temp': 'Temp +\nLow Irr'
    }

    ax.set_ylabel('Error (MW)')
    ax.set_title('Model Robustness in Extreme Weather Scenarios', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([display_names.get(s, s) for s in scenarios])
    ax.legend(frameon=False)

    # 在柱子上打上具体数值
    ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=8)
    ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=8)

    plt.savefig(os.path.join(save_path, 'IEEE_Extreme_Scenarios.pdf'))
    print(">> Extreme Scenarios Bar plot saved.")
    plt.close()


# ==========================================
# 主程序
# ==========================================
if __name__ == '__main__':
    # 设置风格
    set_ieee_style()

    # 路径配置
    experiment_name = 'PhysFormer_ensemble_seed2024'
    results_path = f'exp_results/PhysFormer/checkpoints/{experiment_name}/'

    # 加载数据
    data = load_data(results_path)

    if data is not None:
        # --- 原始图表 ---
        plot_vpp_forecast(data, sample_idx=0, save_path=results_path)
        plot_gate_mechanism(data, sample_idx=0, save_path=results_path)

        plot_net_load_correlation(data, save_path=results_path)

        plot_ramp_compliance(data, ramp_limits=[1.20527597, 0.23828998, 0.36192739], save_path=results_path)

        plot_curriculum_evolution(save_path=results_path)
        plot_extreme_scenarios_bar(save_path=results_path)

        print("\nAll plots generated successfully following IEEE style.")