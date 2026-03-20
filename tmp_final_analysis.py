import numpy as np
import os
from scipy.stats import pearsonr

def calculate_metrics():
    # 1. Informer RVM Calculation
    informer_dir = r'e:\Py_program\Soft-phys-CFC-Informer\exp_results\Informer_vpp_dataset_3years_sl672_pl96_vpp'
    inf_pred = np.load(os.path.join(informer_dir, 'pred.npy'))
    inf_true = np.load(os.path.join(informer_dir, 'true.npy'))
    
    print(f"Informer Pred: Mean={np.mean(inf_pred):.4f}, Max={np.max(inf_pred):.4f}, Min={np.min(inf_pred):.4f}")
    print(f"Informer True: Mean={np.mean(inf_true):.4f}, Max={np.max(inf_true):.4f}, Min={np.min(inf_true):.4f}")
    
    inf_pred_post = np.maximum(inf_pred, 0)
    
    # Calculate RVR (Ramp Violation Rate)
    # We need the ramp limits. Let's try to load them from phys_metrics.npy or use defaults.
    # PhysFormer limits: env says [1.5, 0.5, 0.8] approx for [Load, PV, Wind]
    limits = [1.5, 0.5, 0.8]
    
    def calculate_rvr(p, l):
        # p: [N, T, C]
        diff = np.abs(p[:, 1:, :] - p[:, :-1, :])
        violations = 0
        for i in range(3):
            violations += np.sum(diff[:, :, i] > l[i])
        return (violations / (diff.shape[0] * diff.shape[1] * 3)) * 100

    rvr_inf = calculate_rvr(inf_pred, limits)
    rvr_post = calculate_rvr(inf_pred_post, limits)
    
    # 2. PhysFormer Gate Analysis
    phys_dir = r'e:\Py_program\Soft-phys-CFC-Informer\exp_results\PhysFormer\checkpoints\PhysFormer_full_seed2024'
    gate_pv = np.load(os.path.join(phys_dir, 'vis_gate_pv.npy'))
    irr = np.load(os.path.join(phys_dir, 'vis_irr.npy'))
    
    # Correlation analysis
    r_full, _ = pearsonr(gate_pv.flatten(), irr.flatten())
    
    # Daytime subset (Irr > 0.1)
    day_mask = irr > 0.1
    r_day, p_day = pearsonr(gate_pv[day_mask], irr[day_mask])
    
    print("\n" + "="*40)
    print("ANALYSIS RESULTS FOR REVISION")
    print("="*40)
    print(f"Informer RVR (Original): {rvr_inf:.4f}%")
    print(f"Informer-Post RVR (Clipped): {rvr_post:.4f}%")
    print(f"Gate PV Max: {np.max(gate_pv):.4f}, Mean: {np.mean(gate_pv):.4f}")
    
    # Let's check the wind gate too
    gate_wind = np.load(os.path.join(phys_dir, 'vis_gate_wind.npy'))
    print(f"Gate Wind Max: {np.max(gate_wind):.4f}")

    # Load checkpoint to get converged params
    import torch
    ckpt = torch.load(os.path.join(phys_dir, 'checkpoint.pth'), map_location='cpu')
    # Filter for phys_layer
    phys_params = {k: v for k, v in ckpt.items() if 'phys_layer' in k}
    for k, v in phys_params.items():
        if 'weight' not in k and 'bias' not in k:
            # Apply activations as in the model
            val = v.detach().cpu().numpy()
            if 'pv_efficiency' in k:
                # model uses softplus
                real_val = np.log(1 + np.exp(val)) 
                print(f"Converged {k}: {real_val}")
            elif 'wind_scale' in k:
                real_val = np.log(1 + np.exp(val))
                print(f"Converged {k}: {real_val}")
            else:
                print(f"Converged {k}: {val}")

    print("="*40)

if __name__ == '__main__':
    calculate_metrics()
