import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def plot_fig3_new():
    # Load data
    gate = np.load('fig3_gate_data.npy')
    irr = np.load('fig3_irr_data.npy')
    
    # Subsample for visualization (2000 points is enough for a clear scatter)
    indices = np.random.choice(len(gate), 2000, replace=False)
    gate_sub = gate[indices]
    irr_sub = irr[indices]
    
    # Global r (from full data)
    r_val = 0.8306
    
    plt.figure(figsize=(8, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    
    # Main Scatter
    plt.scatter(irr_sub, gate_sub, alpha=0.4, color='#3498db', s=15, label='Temporal Gate Samples')
    
    # Regression line (on subsample for trend)
    sns.regplot(x=irr_sub, y=gate_sub, scatter=False, color='#e74c3c', 
                line_kws={'linewidth': 2.5, 'label': f'Monotonic Trend (Global r={r_val:.4f})'})
    
    plt.xlabel('Normalized Solar Irradiance ($G_{norm}$)', fontsize=12)
    plt.ylabel('PV Temporal Gate Activation ($\Gamma_{PV}$)', fontsize=12)
    plt.title('Figure 3: Imprinted Gate-Irradiance Correlation in PGCC', fontsize=14, fontweight='bold')
    
    plt.legend(loc='lower right', frameon=True)
    
    # Add a note about the range
    plt.text(0.05, 0.95, f'N={len(gate)} points\nGlobal Pearson r={r_val:.4f}', 
             transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig('IEEE_Fig3_CausalGate.png')
    plt.savefig('IEEE_Fig3_CausalGate.pdf')
    print("New Figure 3 generated: IEEE_Fig3_CausalGate.png/pdf")

if __name__ == "__main__":
    plot_fig3_new()
