import os
import sys
import torch
import numpy as np
from scripts.run_benchmark import get_args
from experiments.exp_PhysFormer import Exp_PhysFormer
from scipy.stats import pearsonr

def analyze_gate():
    args = get_args()
    args.model = 'PhysFormer'
    args.root_path = './'
    args.data_path = 'data/vpp_dataset_3years.csv'
    args.features = 'M'
    args.target = None
    args.freq = '15min'
    args.num_workers = 0
    args.use_amp = True
    args.patience = 3
    args.learning_rate = 1e-3
    args.batch_size = 32
    args.train_epochs = 1
    args.seq_len = 672
    args.pred_len = 96
    args.label_len = 0
    args.checkpoint_name = 'PhysFormer_full_seed2024'
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 3
    args.use_gpu = torch.cuda.is_available()
    args.gpu = 0
    args.batch_size = 32
    
    # PhysFormer Checkpoint
    args.checkpoints = './exp_results/PhysFormer/checkpoints'
    setting = 'PhysFormer_full_seed2024'
    
    exp = Exp_PhysFormer(args)
    model = exp.model
    device = exp.device
    
    # Load model
    load_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
    if not os.path.exists(load_path):
        print(f"Error: Checkpoint not found at {load_path}")
        return
        
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()
    
    test_data, test_loader = exp._get_data(flag='test')
    
    all_gates = []
    all_irradiance = []
    
    print("Running PhysFormer inference to extract gates...")
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            # batch_data contains stat, weather_hist, weather_future, y, x_mark, y_mark
            batch_stat, batch_weather_hist, batch_weather_future, batch_y_vals, batch_x_mark, batch_y_mark = batch_data
            
            batch_stat = batch_stat.float().to(device)
            batch_weather_hist = batch_weather_hist.float().to(device)
            batch_weather_future = batch_weather_future.float().to(device)
            batch_y_vals = batch_y_mark.float().to(device) # Just to match counts
            
            # Forward pass to get gates
            # PhysFormer returns (outputs, gates)
            # gates is a dict with 'gate_pv', 'gate_load', etc.
            _, gates = model(batch_stat, batch_weather_hist, batch_weather_future, None, alpha=0.0)
            
            pv_gate = gates['gate_pv'].cpu().numpy() # [B, 96, 1]
            # Use future irradiance for correlation (as defined in paper)
            # Irradiance is index 1 in weather_future [B, 96, 3]
            irradiance = batch_weather_future[:, :, 1].cpu().numpy() # [B, 96]
            
            all_gates.append(pv_gate.flatten())
            all_irradiance.append(irradiance.flatten())
            
    all_gates = np.concatenate(all_gates)
    all_irradiance = np.concatenate(all_irradiance)
    
    # Analyze Subsets
    # Full Set
    r_full, _ = pearsonr(all_gates, all_irradiance)
    
    # Day-time Subset (Irradiance > 0.1)
    day_mask = all_irradiance > 0.1
    gates_day = all_gates[day_mask]
    irr_day = all_irradiance[day_mask]
    r_day, p_day = pearsonr(gates_day, irr_day)
    
    # Night-time Subset (Irradiance < 0.01)
    night_mask = all_irradiance < 0.01
    gates_night = all_gates[night_mask]
    irr_night = all_irradiance[night_mask]
    r_night, _ = pearsonr(gates_night, irr_night)
    
    # Dynamic Range
    peak_to_peak = np.max(all_gates) - np.min(all_gates)
    
    print("\n" + "="*40)
    print("GATE CORR ANALYSIS (PhysFormer)")
    print("="*40)
    print(f"Full Set Pearson r:     {r_full:.4f}")
    print(f"Day-time Sub Pearson r: {r_day:.4f} (p={p_day:.4e})")
    print(f"Night-time Sub Pearson r: {r_night:.4f}")
    print(f"Absolute Peak-to-Peak:  {peak_to_peak:.6f}")
    print(f"Gate Mean (Full):       {np.mean(all_gates):.6f}")
    print(f"Gate std (Full):        {np.std(all_gates):.6f}")
    print("="*40)

if __name__ == '__main__':
    analyze_gate()
