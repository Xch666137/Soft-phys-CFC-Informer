import torch
import numpy as np
import os
import sys

# Ensure root dir is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.src.models import PhysFormer

def diagnose_init():
    print("--- PGCC Initial Alignment Check (Zero-Shot) ---")
    
    # Initialize model with random weights
    # We need dummy data specs
    model = PhysFormer(
        enc_in=3, seq_len=672, pred_len=96,
        d_model=512, n_heads=8, e_layers=3,
        weather_mean=torch.zeros(3), weather_std=torch.ones(3),
        target_mean=torch.zeros(3), target_std=torch.ones(3)
    )
    model.eval()
    
    # Create synthetic weather data (varying irradiance)
    B, S = 1, 672
    x_weather_hist = torch.randn(B, S, 3) 
    # Let's make irradiance (index 1) a sine wave to see correlation clearly
    t = torch.linspace(0, 4*np.pi, S)
    x_weather_hist[0, :, 1] = torch.sin(t) 
    
    x_stat = torch.randn(B, S, 3)
    x_weather_future = torch.randn(B, 96, 3)
    x_mark_enc = torch.zeros(B, S, 4)
    
    with torch.no_grad():
        output, reg_loss, gates_info = model(
            x_stat, x_weather_hist, x_weather_future, x_mark_enc, 
            alpha=0.0, # Fully trust model (alpha=0)
            return_gates=True
        )
    
    # pv_seq_batch is the gate activation for PV [B, S]
    gate_pv = gates_info['pv_seq_batch'][0]
    irr = x_weather_hist[0, :, 1].numpy()
    
    # Calculate Correlation at Init
    corr = np.corrcoef(gate_pv, irr)[0, 1]
    
    print(f"Correlation (Gate vs Irradiance) at Initialization: {corr:.4f}")
    
    if abs(corr) > 0.3:
        print("Conclusion: Significant architectural bias detected at initialization.")
    else:
        print("Conclusion: No significant architectural bias at initialization. Causal alignment likely comes from optimization.")

if __name__ == "__main__":
    diagnose_init()
