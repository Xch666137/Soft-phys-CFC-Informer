import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

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
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

base_dir = './Visualizaiton_output'
os.makedirs(base_dir, exist_ok=True)
ckpt_dir = 'e:/Py_program/Soft-phys-CFC-Informer/exp_results/PhysFormer/checkpoints/PhysFormer_full_seed2024'

COLORS = {
    'PhysFormer': '#D7191C', # Red
    'PINN': '#FDAE61',       # Orange
    'Transformer': '#2C7BB6',# Blue
    'RNN': '#ABD9E9',        # Light Blue
}

# =====================================================================
# Fig 8: Robustness under 99th-Percentile Volatile Weather
# =====================================================================
def plot_fig8_extreme_weather():
    print("Generating Figure 8: Extreme Weather Robustness...")
    
    # Load Table II CSV
    csv_path = 'e:/Py_program/Soft-phys-CFC-Informer/IEEE_Extreme_Weather_Table.csv'
    df = pd.read_csv(csv_path)
    
    # Rename columns to avoid spacing issues
    df.columns = ['Model', 'MSE', 'BVR', 'MVS', 'NET_MAE', 'Strict RVR', 'DSA']
    
    # Sort models logically
    models = ['LSTM', 'GRU', 'PINN', 'Informer', 'Autoformer', 'DLinear', 'PatchTST', 'iTransformer', 'PhysFormer']
    # Reindex dataframe based on list
    df = df.set_index('Model').reindex(models).reset_index()
    
    # We will draw a combination grouped Bar chart emphasizing MSE vs BVR under strict condition
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Axis 1: Extreme BVR% (Highlighting the catastrophic failure of others)
    color1 = '#e41a1c'
    bars1 = ax1.bar(x - width/2, df['BVR'], width, color=color1, alpha=0.8, edgecolor='black', label='Extreme BVR (%) ↓')
    ax1.set_ylabel('Boundary Violation Rate (BVR %)', color=color1, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Annotate PhysFormer zero-violation
    for bar, model in zip(bars1, df['Model']):
        if model == 'PhysFormer':
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, 
                     '0.00%', ha='center', va='bottom', color=color1, fontweight='bold')
    
    # Axis 2: MSE 
    ax2 = ax1.twinx()
    color2 = '#377eb8'
    bars2 = ax2.bar(x + width/2, df['MSE'], width, color=color2, alpha=0.8, edgecolor='black', label='Extreme MSE ↓')
    ax2.set_ylabel('Mean Squared Error (MSE)', color=color2, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Cut off y-limit for MSE to make 0.01xx differences visible, but Autoformer is 0.2165 (huge),
    # DLinear is 0.1687 (huge). Let's use log scale or limit to show the gap gracefully
    ax2.set_ylim(0, max(df['MSE'])*1.15)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=20, ha='right')
    # Vertical distinguishing line before PhysFormer
    ax1.axvline(x[-1]-1 + width*2, color='gray', linestyle='--')
    
    # Combine legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'IEEE_Fig8_ExtremeWeather.pdf'))
    plt.close()

if __name__ == '__main__':
    set_ieee_style()
    plot_fig8_extreme_weather()
    print("Stage 4 plots generated successfully in Visualizaiton_output/")

