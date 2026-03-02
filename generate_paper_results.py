import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# IEEE 论文绘图全局样式
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ==========================================
# 1. 路径配置
# ==========================================
base_dir = './exp_results'

model_paths = {
    'LSTM':       f'{base_dir}/LSTM_vpp_dataset_3years_sl672_pl96_vpp',
    'GRU':        f'{base_dir}/GRU_vpp_dataset_3years_sl672_pl96_vpp',
    'PINN':       f'{base_dir}/PINN_vpp_dataset_3years_sl672_pl96_vpp',
    'Informer':   f'{base_dir}/Informer_vpp_dataset_3years_sl672_pl96_vpp',
    'Autoformer': f'{base_dir}/Autoformer_vpp_dataset_3years_sl672_pl96_vpp',
    'DLinear':    f'{base_dir}/DLinear_vpp_dataset_3years_sl672_pl96_vpp',
    'PatchTST':   f'{base_dir}/PatchTST_vpp_dataset_3years_sl672_pl96_vpp',
    'PhysFormer': f'{base_dir}/PhysFormer/checkpoints/PhysFormer_full_seed2024',
}

# 颜色方案（IEEE 风格，高对比度低饱和学术风）
MODEL_COLORS = {
    'LSTM':       '#8c564b', # 棕色
    'GRU':        '#9467bd', # 紫色
    'PINN':       '#e377c2', # 粉色
    'Informer':   '#d62728', # 红色
    'Autoformer': '#ff7f0e', # 橙色
    'DLinear':    '#2ca02c', # 绿色
    'PatchTST':   '#1f77b4', # 蓝色
    'PhysFormer': '#d73027', # 高亮正红（主角强调，与其他拉开反差）
}

MODEL_STYLES = {
    'LSTM':       (':',   1.5),
    'GRU':        ('-.',  1.5),
    'PINN':       (':',   1.5),
    'Informer':   ('--',  1.8),
    'Autoformer': (':',   1.5),
    'DLinear':    ('-.',  1.5),
    'PatchTST':   ('--',  1.8),
    'PhysFormer': ('-',   2.5),
}

CHANNEL_NAMES = ['Load', 'PV', 'Wind']
CHANNEL_UNITS = 'Power (MW)'

# ==========================================
# 2. 加载所有 metrics
# ==========================================
results = {}
pred_data = {}
true_data = {}

for model, path in model_paths.items():
    metrics_path = os.path.join(path, 'metrics.npy')
    pred_path = os.path.join(path, 'pred.npy')
    true_path = os.path.join(path, 'true.npy')

    if os.path.exists(metrics_path):
        metrics = np.load(metrics_path, allow_pickle=True)
        results[model] = {
            'MAE': float(metrics[0]),
            'RMSE': float(metrics[2]),
            'BVR (%)': float(metrics[5]),
        }

        # 计算 MVS: PV/Wind 通道负值预测的平均违规幅度
        if os.path.exists(pred_path):
            pred = np.load(pred_path, allow_pickle=True)
            pred_pv_wind = pred[:, :, 1:3]  # PV 和 Wind 通道
            violations = pred_pv_wind[pred_pv_wind < 0]
            mvs = float(np.mean(np.abs(violations))) if len(violations) > 0 else 0.0
            results[model]['MVS'] = mvs
        else:
            results[model]['MVS'] = float('nan')
    else:
        print(f"[警告] metrics.npy 未找到: {metrics_path}")

    if os.path.exists(pred_path) and os.path.exists(true_path):
        pred_data[model] = np.load(pred_path, allow_pickle=True)  # [N, T, C]
        true_data[model]  = np.load(true_path, allow_pickle=True)

# ==========================================
# 3. Table I
# ==========================================
if results:
    df = pd.DataFrame(results).T[['MAE', 'RMSE', 'BVR (%)', 'MVS']]
    print("\n" + "=" * 60)
    print("      IEEE TRANSACTIONS ON SMART GRID - TABLE I")
    print("=" * 60)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 60 + "\n")
    df.to_csv("IEEE_Table_I_Results.csv", float_format="%.4f")
    print("已生成表格文件: IEEE_Table_I_Results.csv\n")

# ==========================================
# 4. Fig 6: BVR 合规性对比 (对数坐标)
# ==========================================
plot_models_compliance = ['Informer', 'Autoformer', 'DLinear', 'PatchTST', 'PhysFormer']
avail_compliance = [m for m in plot_models_compliance if m in results]

if avail_compliance:
    bvr_data = [max(results[m]['BVR (%)'], 1e-3) for m in avail_compliance]
    x = np.arange(len(avail_compliance))
    width = 0.5

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, bvr_data, width, label='Boundary Violation Rate (BVR)',
                   color='#1f77b4', edgecolor='black', linewidth=0.7)
    ax.set_yscale('log')

    # 标注数值
    for bar in bars:
        h = bar.get_height()
        if h > 1e-2:
            ax.text(bar.get_x() + bar.get_width()/2., h * 1.3,
                    f'{h:.2f}%', ha='center', va='bottom', fontsize=7.5, rotation=45)

    ax.set_ylabel('Violation Rate (%) [Log Scale]', fontweight='bold')
    ax.set_title('Physical Constraint Compliance Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(avail_compliance, fontweight='bold')
    ticklabels = ax.get_xticklabels()
    ticklabels[-1].set_color('red')  # 突出 PhysFormer
    ax.legend(loc='upper right')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('IEEE_Fig6_Compliance_LogScale.pdf', bbox_inches='tight')
    plt.savefig('IEEE_Fig6_Compliance_LogScale.png', bbox_inches='tight')
    plt.close()
    print("已生成: IEEE_Fig6_Compliance_LogScale.pdf")


# ==========================================
# 5. Fig 7: PhysFormer 单模型预测对比图
#    (Load / PV / Wind / Net Load 四子图)
#    对应论文中展示的主要可视化图
# ==========================================
def plot_physformer_forecast(pred, true, sample_idx=0, save_prefix='IEEE_Fig7_PhysFormer'):
    """
    绘制 PhysFormer 四通道预测对比图
    pred/true: [N, T, C]，C=3 对应 [Load, PV, Wind]
    """
    p = pred[sample_idx]   # [T, 3]
    t = true[sample_idx]   # [T, 3]
    T = p.shape[0]
    time_steps = np.arange(T)

    # 净负荷 = Load - PV - Wind
    net_pred = p[:, 0] - p[:, 1] - p[:, 2]
    net_true = t[:, 0] - t[:, 1] - t[:, 2]

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    subplot_configs = [
        (gs[0, 0], 0, '(a) Load Forecast',    '#2166ac'),
        (gs[0, 1], 1, '(b) PV Power Forecast', '#d6604d'),
        (gs[1, 0], 2, '(c) Wind Power Forecast','#4dac26'),
    ]

    for gs_pos, ch_idx, title, color in subplot_configs:
        ax = fig.add_subplot(gs_pos)
        ax.plot(time_steps, t[:, ch_idx], color='#333333', lw=1.5,
                label='Ground Truth', zorder=3)
        ax.plot(time_steps, p[:, ch_idx], color=color, lw=1.5,
                linestyle='--', label='Prediction', zorder=4)
        # 填充误差区域
        ax.fill_between(time_steps, t[:, ch_idx], p[:, ch_idx],
                        alpha=0.15, color=color)
        ax.set_title(title, fontweight='bold', pad=6)
        ax.set_xlabel('Time Steps (15 min)')
        ax.set_ylabel(CHANNEL_UNITS)
        ax.legend(loc='upper left', framealpha=0.8)

    # 净负荷子图
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time_steps, net_true, color='#333333', lw=1.5, label='Net Truth', zorder=3)
    ax4.plot(time_steps, net_pred, color='#555555', lw=1.5, linestyle='--',
             label='Net Pred', zorder=4)
    ax4.fill_between(time_steps, net_true, net_pred, alpha=0.2, color='gray', label='Error')
    ax4.set_title('(d) VPP Net Load (Aggregation)', fontweight='bold', pad=6)
    ax4.set_xlabel('Time Steps (15 min)')
    ax4.set_ylabel(CHANNEL_UNITS)
    ax4.legend(loc='upper left', framealpha=0.8)

    fig.suptitle('PhysFormer: VPP Multi-Channel Forecasting Results',
                 fontsize=13, fontweight='bold', y=1.01)

    plt.savefig(f'{save_prefix}.pdf', bbox_inches='tight')
    plt.savefig(f'{save_prefix}.png', bbox_inches='tight')
    plt.close()
    print(f"已生成: {save_prefix}.pdf")


if 'PhysFormer' in pred_data:
    # 自动选取一个有代表性的样本（PV有明显日间峰值的）
    pv_max_per_sample = pred_data['PhysFormer'][:, :, 1].max(axis=1)
    # 选 PV 峰值中等的样本，更有代表性
    sorted_idx = np.argsort(pv_max_per_sample)
    sample_idx = sorted_idx[len(sorted_idx) // 2]
    plot_physformer_forecast(pred_data['PhysFormer'], true_data['PhysFormer'],
                             sample_idx=sample_idx)


# ==========================================
# 6. Fig 8: 多模型横向对比预测图
#    对同一个测试样本，展示所有模型的预测曲线
#    每行一个通道，每列……（实际上同一图内叠加）
# ==========================================
def plot_multi_model_comparison(pred_data, true_data, results,
                                 channel_idx=0, channel_name='Load',
                                 sample_idx=None, save_prefix='IEEE_Fig4_ForecastComparison'):
    """
    对单个通道，将所有模型的预测曲线画在同一张图上
    同时附带 MSE 排名标注
    """
    # 找所有模型都有 pred 的情况
    avail_models = [m for m in pred_data if m in true_data]
    if not avail_models:
        print("[跳过] 没有可用的 pred/true 数据")
        return

    # 用 PhysFormer 的 true 作为基准（或第一个可用的）
    ref_model = 'PhysFormer' if 'PhysFormer' in avail_models else avail_models[0]
    true_ref = true_data[ref_model]   # [N, T, C]

    if sample_idx is None:
        # 选 PV 峰值中等的样本
        pv_max = true_ref[:, :, 1].max(axis=1)
        sample_idx = np.argsort(pv_max)[len(pv_max) // 2]

    T = true_ref.shape[1]
    time_steps = np.arange(T)

    fig, ax = plt.subplots(figsize=(12, 5))

    # 先画 Ground Truth，设为最粗的浅灰色背景参照底线
    ax.plot(time_steps, true_ref[sample_idx, :, channel_idx],
            color='#999999', lw=3.5, label='Ground Truth', zorder=1, alpha=0.6)

    # 按 MSE 排序画各模型，仅保留代表性模型以便观察
    selected_plot_models = {'PhysFormer', 'Informer', 'DLinear'}
    sorted_models = sorted([m for m in avail_models if m in selected_plot_models],
                           key=lambda m: results.get(m, {}).get('MSE', 999),
                           reverse=True)

    for model in sorted_models:
        p = pred_data[model][sample_idx, :, channel_idx]
        mse_val = results.get(model, {}).get('MSE', float('nan'))
        style, lw = MODEL_STYLES.get(model, ('-', 1.0))
        color = MODEL_COLORS.get(model, 'gray')
        
        # 核心：PhysFormer画在最上层，Informer等次之，其他在下面，适当设置透明度
        is_hero = (model == 'PhysFormer')
        is_subhero = (model in ['Informer', 'PatchTST'])
        
        lw_use = lw
        z = 10 if is_hero else (5 if is_subhero else 3)
        a = 1.0 if is_hero else (0.85 if is_subhero else 0.5)
        
        ax.plot(time_steps, p, color=color, lw=lw_use, linestyle=style,
                label=f'{model} (MSE={mse_val:.4f})', zorder=z, alpha=a)

    ax.set_xlabel('Time Steps (15 min)', fontweight='bold')
    ax.set_ylabel(CHANNEL_UNITS, fontweight='bold')
    ax.set_title(f'Multi-Model Forecasting Comparison — {channel_name} Channel',
                 fontweight='bold')

    # 放宽图例，避免挤到一起
    ax.legend(loc='upper right', framealpha=0.95, ncol=2, fontsize=10, 
              edgecolor='black', fancybox=False)
    ax.grid(True, which='major', ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_{channel_name}.pdf', bbox_inches='tight')
    plt.savefig(f'{save_prefix}_{channel_name}.png', bbox_inches='tight')
    plt.close()
    print(f"已生成: {save_prefix}_{channel_name}.pdf")


# 绘制三个通道的多模型对比图
if pred_data:
    # 确定统一使用的 sample_idx（用 PhysFormer 的 PV 中等峰值样本）
    ref = 'PhysFormer' if 'PhysFormer' in pred_data else list(pred_data.keys())[0]
    pv_max = true_data[ref][:, :, 1].max(axis=1)
    shared_sample = int(np.argsort(pv_max)[len(pv_max) // 2])

    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        plot_multi_model_comparison(
            pred_data, true_data, results,
            channel_idx=ch_idx,
            channel_name=ch_name,
            sample_idx=shared_sample,
        )


# ==========================================
# 7. Fig 9: 误差分布箱线图
#    每个模型的逐步预测误差分布，展示稳定性
# ==========================================
def plot_error_boxplot(pred_data, true_data, save_prefix='IEEE_Fig9_ErrorBoxplot'):
    """
    计算每个模型的 MAE 分布（按预测步长），画箱线图对比稳定性
    """
    avail_models = [m for m in pred_data if m in true_data]
    if not avail_models:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ch_idx, (ax, ch_name) in enumerate(zip(axes, CHANNEL_NAMES)):
        plot_data = []
        model_labels = []

        # 按 MSE 从大到小排序
        sorted_models = sorted(avail_models,
                               key=lambda m: results.get(m, {}).get('MSE', 999),
                               reverse=True)

        for model in sorted_models:
            p = pred_data[model][:, :, ch_idx]   # [N, T]
            t = true_data[model][:, :, ch_idx]
            mae_per_step = np.mean(np.abs(p - t), axis=0)   # [T]，每步的平均误差
            plot_data.append(mae_per_step)
            model_labels.append(model)

        bp = ax.boxplot(plot_data, labels=model_labels, patch_artist=True,
                        showfliers=False, medianprops=dict(color='red', lw=1.5))

        # 着色
        for patch, model in zip(bp['boxes'], sorted_models):
            patch.set_facecolor(MODEL_COLORS.get(model, 'lightblue'))
            patch.set_alpha(0.7)
            if model == 'PhysFormer':
                patch.set_edgecolor('black')
                patch.set_linewidth(2.0)

        ax.set_title(f'{ch_name} — Step-wise MAE Distribution', fontweight='bold')
        ax.set_ylabel('MAE (MW)' if ch_idx == 0 else '')
        ax.set_xticklabels(model_labels, rotation=30, ha='right', fontsize=8)
        ax.grid(True, axis='y', ls='--', alpha=0.3)

    fig.suptitle('Forecast Error Distribution Across Models (96-step Horizon)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}.pdf', bbox_inches='tight')
    plt.savefig(f'{save_prefix}.png', bbox_inches='tight')
    plt.close()
    print(f"已生成: {save_prefix}.pdf")


if pred_data:
    plot_error_boxplot(pred_data, true_data)


# ==========================================
# 8. Fig 10: 预测误差随时间步长的变化曲线
#    展示各模型长期预测退化情况
# ==========================================
def plot_mae_vs_horizon(pred_data, true_data, save_prefix='IEEE_Fig10_MAEvsHorizon'):
    """
    对所有测试样本，计算每个预测步的平均 MAE，展示误差随预测距离的增长趋势
    """
    avail_models = [m for m in pred_data if m in true_data]
    if not avail_models:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    sorted_models = sorted(avail_models,
                           key=lambda m: results.get(m, {}).get('MSE', 999),
                           reverse=True)

    for ch_idx, (ax, ch_name) in enumerate(zip(axes, CHANNEL_NAMES)):
        for model in sorted_models:
            p = pred_data[model][:, :, ch_idx]   # [N, T]
            t = true_data[model][:, :, ch_idx]
            mae_per_step = np.mean(np.abs(p - t), axis=0)   # [T]

            style, lw = MODEL_STYLES.get(model, ('-', 1.0))
            color = MODEL_COLORS.get(model, 'gray')
            lw_use = lw * 1.5 if model == 'PhysFormer' else lw
            ax.plot(np.arange(1, len(mae_per_step)+1), mae_per_step,
                    color=color, lw=lw_use, linestyle=style, label=model,
                    alpha=1.0 if model == 'PhysFormer' else 0.7,
                    zorder=5 if model == 'PhysFormer' else 3)

        ax.set_title(f'{ch_name}', fontweight='bold')
        ax.set_xlabel('Prediction Horizon (steps × 15min)')
        ax.set_ylabel('MAE (MW)' if ch_idx == 0 else '')
        ax.grid(True, ls='--', alpha=0.3)
        if ch_idx == 2:
            ax.legend(loc='upper left', fontsize=8, ncol=1)

    fig.suptitle('MAE vs. Prediction Horizon — All Models',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}.pdf', bbox_inches='tight')
    plt.savefig(f'{save_prefix}.png', bbox_inches='tight')
    plt.close()
    print(f"已生成: {save_prefix}.pdf")


if pred_data:
    plot_mae_vs_horizon(pred_data, true_data)

print("\n✅ 所有图表生成完毕！")