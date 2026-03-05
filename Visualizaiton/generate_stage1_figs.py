import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import torch
import warnings
warnings.filterwarnings('ignore')

# 保证能导入上级目录的模型
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.src.models.PhysFormer.physical_layer import ExplicitPhysicalMapping

def set_ieee_style():
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'lines.linewidth': 1.5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 1.0,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

COLORS = {
    'PhysFormer': '#D7191C', # Red
    'PINN': '#FDAE61',       # Orange
    'Transformer': '#2C7BB6',# Blue
    'RNN': '#ABD9E9',        # Light Blue
    'Ablation': '#7B3294'    # Purple
}

base_dir = './Visualizaiton_output'
os.makedirs(base_dir, exist_ok=True)

# =====================================================================
# Fig 2: Explicit Physical Mapping Functions (Fixed)
# =====================================================================
def plot_fig2_physical_mapping(checkpoint_path):
    print("Generating Figure 2: Physical Mapping Functions...")
    # Load model weights
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    # 论文 Table III 写了最终收敛值
    p_temp = 0.0032 # 0.0032/C
    p_ci = 3.5      # 3.5 m/s
    p_rated = 12.0  # 12.0 m/s
    p_co = 25.0     # 25.0 m/s
    p_base = 2.83   # 2.83 MW
    p_comf = 20.0   # 20.0 C

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    
    # 1. PV Curve
    # P_pv = G * η * clamp(1 - β(T-25), 0, 1.5)
    ax = axes[0]
    temps = np.linspace(-10, 45, 100)
    G_levels = [0.2, 0.6, 1.0] # Normalized Irradiance levels
    for G in G_levels:
        # Assuming nominal efficiency η corresponds to max output 1.0 mapping
        P_pv = G * np.clip(1 - p_temp * (temps - 25), 0, 1.5)
        ax.plot(temps, P_pv, label=f'G = {G} (norm)', lw=2)
    ax.set_title('(a) PV Temperature Degradation')
    ax.set_xlabel('Cell Temperature ($^\circ$C)')
    ax.set_ylabel('Normalized Power Output')
    ax.legend(frameon=False)
    ax.grid(True)

    # 2. Wind Curve
    # P_wind = scale * sigmoid(5 * (v_norm - 0.5)) * I_running
    # v_norm = (v - ci) / (rated - ci) 
    ax = axes[1]
    speeds = np.linspace(0, 30, 200)
    P_wind = np.zeros_like(speeds)
    for i, v in enumerate(speeds):
        if p_ci <= v < p_co:
            # 只有在运行区间才有输出
            v_norm = (v - p_ci) / (p_rated - p_ci + 1e-6) 
            # Sigmoid response smoothed curve 
            val = 1.0 / (1 + np.exp(-5 * (v_norm - 0.5)))
            
            # Constrain by max capacity at rated
            if v >= p_rated:
                val = 1.0 # 额定到切出之间满发
            P_wind[i] = val
        else:
            P_wind[i] = 0.0
            
    ax.plot(speeds, P_wind, color='#1A9641', lw=2)
    # 添加阴影区域强调运行区间
    ax.axvspan(p_ci, p_co, color='#1A9641', alpha=0.1)
    
    ax.axvline(p_ci, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(p_rated, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(p_co, color='gray', linestyle='--', alpha=0.7)
    
    ax.text(p_ci, 0.8, '$v_{ci}$', ha='right', va='center', color='gray')
    ax.text(p_rated, 0.2, '$v_{r}$', ha='right', va='center', color='gray')
    ax.text(p_co, 0.5, '$v_{co}$', ha='left', va='center', color='gray')
    
    ax.set_title('(b) Wind Turbine Power Curve')
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Power Generation Coefficient')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True)

    # 3. Load Curve
    # P_load = P_base + γ * (T - Tc)^2
    # 这里的 gamma 为了画图明显表现凹函数趋势，我们可以取真实数据学到的或合理的刻度
    ax = axes[2]
    temps = np.linspace(-5, 40, 100)
    gamma = 0.005 
    P_load = p_base + gamma * (temps - p_comf)**2
    
    ax.plot(temps, P_load, color='#2C7BB6', lw=2)
    ax.axvline(p_comf, color='red', linestyle='--', alpha=0.5)
    
    # 画一个最低点标示
    ax.scatter([p_comf], [p_base], color='red', zorder=5)
    ax.text(p_comf, p_base + 0.5, f'$T_c={p_comf}^\circ$C\nBase: {p_base} MW', color='red', ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax.set_title('(c) Load Thermal Comfort Demand')
    ax.set_xlabel('Ambient Temperature ($^\circ$C)')
    ax.set_ylabel('Power Demand (MW)')
    ax.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20, wspace=0.25)
    plt.savefig(os.path.join(base_dir, 'IEEE_Fig2_PhysicalMapping.pdf'))
    plt.close()

# =====================================================================
# Fig 5: 2D Pareto Frontier (MSE vs BVR%, bubble=MVS)
# =====================================================================
def plot_fig5_pareto():
    print("Generating Figure 5: 2D Pareto Frontier...")
    # Data from Table I
    data = {
        'Model': ['LSTM', 'GRU', 'PINN', 'Informer', 'Autoformer', 'DLinear', 'PatchTST', 'PhysFormer'],
        'MSE': [0.0167, 0.0163, 0.0157, 0.0121, 0.1253, 0.1014, 0.0230, 0.0128],
        'BVR': [11.25, 11.15, 10.94, 19.53, 12.67, 7.03, 12.63, 0.00],
        'MVS': [0.0093, 0.0103, 0.0097, 0.0036, 0.0737, 0.0295, 0.0207, 0.0000],
        'Category': ['RNN', 'RNN', 'PINN', 'Transformer', 'Transformer', 'Transformer', 'Transformer', 'PhysFormer']
    }
    df = pd.DataFrame(data)
    
    # 为在图中好看，过滤掉 Autoformer和DLinear的超大MSE以便聚焦高性能区域
    df_focus = df[df['MSE'] < 0.03].copy()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # 乘一个放大系数使气泡显示清晰
    bubble_scale = 30000 
    
    for i, row in df_focus.iterrows():
        color = COLORS.get(row['Category'], '#333333')
        # 计算气泡大小，确保 MVS=0 时依然可见 (代表极致合规)
        size = max(row['MVS'] * bubble_scale, 80) 
        
        # 对于 PhysFormer 使用特殊的视觉强调
        if row['Model'] == 'PhysFormer':
            ax.scatter(row['MSE'], row['BVR'], s=size, color=color, alpha=0.9, edgecolors='black', linewidth=1.5, zorder=5)
            # 添加光晕效果
            ax.scatter(row['MSE'], row['BVR'], s=size*2.5, color=color, alpha=0.15, zorder=4)
        else:
            ax.scatter(row['MSE'], row['BVR'], s=size, color=color, alpha=0.6, edgecolors='white', linewidth=0.5, zorder=3)
            
        # 添加模型文字标签
        y_offset = -0.8 if row['Model'] == 'Informer' else 0.8
        x_offset = 0 if row['Model'] in ['PINN', 'GRU', 'LSTM'] else 0.0002
        
        # 针对不同模型精细调整偏移量防止重叠
        if row['Model'] == 'PhysFormer': 
            y_offset = 1.0; x_offset = 0.0002
        elif row['Model'] == 'PINN': 
            y_offset = -1.5; x_offset = 0.0
        elif row['Model'] == 'GRU': 
            y_offset = 0.8; x_offset = -0.0005 # 往左上方偏
        elif row['Model'] == 'LSTM': 
            y_offset = 1.5; x_offset = 0.0005  # 往右上方偏，避开 GRU
        elif row['Model'] == 'PatchTST':
            y_offset = 1.5; x_offset = 0.0
        
        ax.text(row['MSE'] + x_offset, row['BVR'] + y_offset, row['Model'], 
                fontsize=9, ha='center', va='bottom', fontweight='bold' if row['Model'] == 'PhysFormer' else 'normal')

    # 绘制理想的 Pareto 前沿线（虚线）
    pareto_x = [0.0121, 0.0128]
    pareto_y = [19.53, 0.00]
    ax.plot(pareto_x, pareto_y, linestyle='--', color='gray', alpha=0.5, zorder=1)

    ax.set_xlabel('Mean Squared Error (MSE) $\\downarrow$')
    ax.set_ylabel('Boundary Violation Rate (BVR \%) $\\downarrow$')
    ax.set_title('Fig. 5: Accuracy-Compliance Pareto Frontier (Bubble Size = MVS)')
    
    # Legend for Categories
    legend_elements = [
        Patch(facecolor=COLORS['PhysFormer'], label='PhysFormer (Ours)'),
        Patch(facecolor=COLORS['PINN'], label='Physics-Informed (PINN)'),
        Patch(facecolor=COLORS['Transformer'], label='Transformer Variants'),
        Patch(facecolor=COLORS['RNN'], label='RNN Variants')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False)
    ax.grid(True)
    
    # 增加一个注释说明气泡大小
    ax.text(0.05, 0.05, 'Bubble Size $\\propto$ Mean Violation Severity (MVS)', 
            transform=ax.transAxes, fontsize=9, style='italic', color='dimgray')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(os.path.join(base_dir, 'IEEE_Fig5_Pareto_2D.pdf'))
    plt.close()

# =====================================================================
# Fig 9: Ablation Dual-Axis Bar Chart
# =====================================================================
def plot_fig9_ablation():
    print("Generating Figure 9: Ablation Study...")
    # Data from Table IV
    labels = ['w/o Curriculum', 'w/o Future GLU', 'w/o PGCC', 'w/o Phys Stream', 'PhysFormer']
    mse_vals = [0.0139, 0.0130, 0.0128, 0.0134, 0.0128]
    # base MSE 选用没有课程学习的 0.0139
    base_mse = 0.0139
    # 相对 MSE 增长 (这里我们画相对变化，或者直接画绝对值也可以，为了体现递进，用条形图画绝对值并标出差值)
    # y2 画 gate_r。 对于 N/A 我们填 0，并在图上标 N/A
    gate_r_vals = [0.7866, 0.8024, 0.0, 0.0, 0.8306]
    gate_r_labels = ['0.7866', '0.8024', 'N/A', 'N/A', '0.8306']

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # Axis 1: MSE
    color1 = '#3182bd'
    bars1 = ax1.bar(x - width/2, mse_vals, width, color=color1, alpha=0.8, edgecolor='black', label='MSE (Lower is better)')
    ax1.set_ylabel('Mean Squared Error (MSE)', color=color1, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    # 设置一个合理的ylim截断，以突现微小差距
    ax1.set_ylim(0.0120, 0.0142)

    # 标上具体的 MSE 值
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.00005, f'{yval:.4f}', ha='center', va='bottom', fontsize=9)

    # Axis 2: Gate-R
    ax2 = ax1.twinx()
    color2 = '#e6550d'
    bars2 = ax2.bar(x + width/2, gate_r_vals, width, color=color2, alpha=0.8, edgecolor='black', label='Gate-Irradiance Pearson $r$ (Higher is better)')
    ax2.set_ylabel('Causal Correlation $r$', color=color2, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1.0)

    # 标上 N/A 或实际值
    for i, bar in enumerate(bars2):
        yval = bar.get_height()
        label = gate_r_labels[i]
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (0.02 if yval > 0 else 0.05), label, ha='center', va='bottom', fontsize=9, color='darkred' if label=='N/A' else 'black', fontweight='bold' if label=='N/A' else 'normal')

    # Titles and Legends
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right')
    ax1.set_title('Fig. 9: Ablation Analysis on Predictive Accuracy vs. Causal Interpretability')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'IEEE_Fig9_Ablation.pdf'))
    plt.close()

if __name__ == '__main__':
    set_ieee_style()
    ckpt_path = os.path.join('e:/Py_program/Soft-phys-CFC-Informer', 'exp_results', 'PhysFormer/checkpoints/PhysFormer_full_seed2024', 'checkpoint.pth')
    plot_fig2_physical_mapping(ckpt_path)
    plot_fig5_pareto()
    plot_fig9_ablation()
    print("Stage 1 plots generated successfully in Visualizaiton_output/")
