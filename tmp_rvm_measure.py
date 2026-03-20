import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def calculate_rvm():
    # 1. Load Train Data to get limits
    df_raw = pd.read_csv(r'e:\Py_program\Soft-phys-CFC-Informer\data\vpp_dataset_3years.csv')
    num_train = int(len(df_raw) * 0.7)
    train_data = df_raw.iloc[:num_train, 1:4].values # Load, PV, Wind
    
    diff_train = np.abs(train_data[1:] - train_data[:-1])
    limits = np.percentile(diff_train, 99.9, axis=0) * 1.5
    print(f"Ramp Limits (MW/step): {limits}")

    # 2. Load Predictions
    inf_dir = r'e:\Py_program\Soft-phys-CFC-Informer\exp_results\Informer_vpp_dataset_3years_sl672_pl96_vpp'
    phys_dir = r'e:\Py_program\Soft-phys-CFC-Informer\exp_results\PhysFormer\checkpoints\PhysFormer_full_seed2024'
    
    inf_pred = np.load(os.path.join(inf_dir, 'pred.npy')) 
    inf_true = np.load(os.path.join(inf_dir, 'true.npy'))
    phys_pred = np.load(os.path.join(phys_dir, 'pred.npy'))
    phys_true = np.load(os.path.join(phys_dir, 'true.npy'))
    
    # 3. Informer-Post (Clipped)
    inf_pred_post = np.maximum(inf_pred, 0)
    
    def get_rvm_jitter(p, l):
        # We want a metric that captures "jaggedness" beyond the limit.
        diff = np.abs(p[:, 1:, :] - p[:, :-1, :])
        # RVM in paper text refers to high-frequency artifacts.
        # Let's try Mean Violation Magnitude: mean(relu(diff - limit))
        violations = np.maximum(diff - l, 0)
        return np.mean(violations)

    def get_rvr(p, l):
        diff = np.abs(p[:, 1:, :] - p[:, :-1, :])
        violations = diff > l
        return np.mean(violations) * 100

    rvm_inf = get_rvm_jitter(inf_pred, limits)
    rvm_post = get_rvm_jitter(inf_pred_post, limits)
    
    rvr_inf = get_rvr(inf_pred, limits)
    rvr_post = get_rvr(inf_pred_post, limits)

    # Average Ramp Magnitude
    avg_ramp_inf = np.mean(np.abs(np.diff(inf_pred, axis=1)))
    avg_ramp_post = np.mean(np.abs(np.diff(inf_pred_post, axis=1)))
    
    # Jitter (2nd difference)
    jitter_inf = np.mean(np.abs(inf_pred[:, 2:, :] - 2*inf_pred[:, 1:-1, :] + inf_pred[:, :-2, :]))
    jitter_post = np.mean(np.abs(inf_pred_post[:, 2:, :] - 2*inf_pred_post[:, 1:-1, :] + inf_pred_post[:, :-2, :]))

    print("\n" + "="*40)
    print("RECALCULATED RVM (Jitter Magnitude)")
    print("="*40)
    print(f"Informer RVM (MVM):    {rvm_inf:.6f}")
    print(f"Informer-Post RVM (MVM): {rvm_post:.6f}")
    print(f"Informer RVR:          {rvr_inf:.6f}%")
    print(f"Informer-Post RVR:     {rvr_post:.6f}%")
    print("-" * 40)
    print(f"Avg Ramp (1st diff): Inf={avg_ramp_inf:.6f}, Post={avg_ramp_post:.6f}")
    print(f"Jitter (2nd diff):   Inf={jitter_inf:.6f},   Post={jitter_post:.6f}")
    def get_dsa(p, t):
        return np.mean(np.abs(np.diff(p, axis=1) - np.diff(t, axis=1)))

    dsa_inf = get_dsa(inf_pred, inf_true)
    dsa_post = get_dsa(inf_pred_post, inf_true)
    dsa_phys = get_dsa(phys_pred, phys_true)

    print("\n" + "="*40)
    print("DERIVATIVE SIMILARITY ALIGNMENT (DSA)")
    print("="*40)
    print(f"Informer DSA:      {dsa_inf:.6f}")
    print(f"Informer-Post DSA: {dsa_post:.6f}")
    print(f"PhysFormer DSA:    {dsa_phys:.6f}")
    print(f"DSA Penalty due to clipping: {(dsa_post/dsa_inf - 1)*100:.4f}%")
    print("="*40)

if __name__ == '__main__':
    calculate_rvm()
