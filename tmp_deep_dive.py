import numpy as np
import os

def analyze_all():
    inf_dir = r'e:\Py_program\Soft-phys-CFC-Informer\exp_results\Informer_vpp_dataset_3years_sl672_pl96_vpp'
    phys_dir = r'e:\Py_program\Soft-phys-CFC-Informer\exp_results\PhysFormer\checkpoints\PhysFormer_full_seed2024'
    
    inf_pred = np.load(os.path.join(inf_dir, 'pred.npy'))
    inf_true = np.load(os.path.join(inf_dir, 'true.npy'))
    phys_pred = np.load(os.path.join(phys_dir, 'pred.npy'))
    phys_true = np.load(os.path.join(phys_dir, 'true.npy'))

    # Anti-norm
    # Based on plot_rvm.py, let's assume scaler means/stds
    # (Actually we can just work in normalized space or use approx MW)
    # We only care about PV (index 1) and Wind (index 2)
    
    def get_metrics(p, t, label):
        # Focus on PV/Wind channels [:, :, 1:3]
        p_sub = p[:, :, 1:3]
        t_sub = t[:, :, 1:3]
        
        diff_p = np.abs(np.diff(p_sub, axis=1))
        diff_t = np.abs(np.diff(t_sub, axis=1))
        
        # 1. Mean Ramp Error (MRE)
        mre = np.mean(np.abs(diff_p - diff_t))
        
        # 2. Max Ramp
        max_ramp = np.max(diff_p)
        
        # 3. Jitter (2nd diff magnitude)
        jitter = np.mean(np.abs(p_sub[:, 2:, :] - 2*p_sub[:, 1:-1, :] + p_sub[:, :-2, :]))
        
        # 4. RVR with tight threshold (e.g. 95th percentile of true ramps)
        thresh = np.percentile(np.abs(np.diff(t_sub, axis=1)), 99)
        rvr = np.mean(diff_p > thresh) * 100
        
        print(f"[{label}] MRE: {mre:.6f} | MaxRamp: {max_ramp:.4f} | Jitter: {jitter:.6f} | RVR(99%): {rvr:.4f}%")

    print("Global Test Set Analysis:")
    get_metrics(inf_pred, inf_true, "Informer")
    inf_post = np.maximum(inf_pred, -0.4) # Approx zero in normalized space for PV (needs exact zero)
    # Let's get exact zero from data
    zero_pv = (0 - 0.15) / 0.18 # Example based on logs
    get_metrics(np.maximum(inf_pred, -0.8), inf_true, "Informer-Post (Approx Clip)")
    get_metrics(phys_pred, phys_true, "PhysFormer")

if __name__ == '__main__':
    analyze_all()
