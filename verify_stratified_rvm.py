import numpy as np
import pandas as pd
import os

def calculate_stratified_rvm():
    # 1. Load Train Data to get limits and std
    print("Loading data...")
    df_raw = pd.read_csv(r'e:\Py_program\Soft-phys-CFC-Informer\data\vpp_dataset_3years.csv')
    num_train = int(len(df_raw) * 0.7)
    train_data = df_raw.iloc[:num_train, 1:4].values # Load, PV, Wind
    
    # Calculate limits for RVM definition
    diff_train = np.abs(train_data[1:] - train_data[:-1])
    limits = np.percentile(diff_train, 99.9, axis=0) * 1.5
    print(f"Ramp Limits (MW/step): {limits}")
    
    # Calculate std for each channel to define boundary region
    std_x = np.std(train_data, axis=0)
    print(f"Standard Deviations (Load, PV, Wind): {std_x}")
    boundary_threshold = 0.05 * std_x
    print(f"Boundary thresholds (0.05 * std): {boundary_threshold}")

    # 2. Load Predictions
    inf_dir = r'e:\Py_program\Soft-phys-CFC-Informer\exp_results\Informer_vpp_dataset_3years_sl672_pl96_vpp'
    # Use actual checkpoint path for PhysFormer, you may need to adjust if name differs
    phys_dir = r'e:\Py_program\Soft-phys-CFC-Informer\exp_results\PhysFormer_vpp_dataset_3years_sl672_pl96_vpp'
    if not os.path.exists(phys_dir):
        # Fallback to the one in tmp_rvm_measure.py
        phys_dir = r'e:\Py_program\Soft-phys-CFC-Informer\exp_results\PhysFormer\checkpoints\PhysFormer_full_seed2024'
    
    inf_pred = np.load(os.path.join(inf_dir, 'pred.npy')) 
    inf_true = np.load(os.path.join(inf_dir, 'true.npy'))
    
    try:
        phys_pred = np.load(os.path.join(phys_dir, 'pred.npy'))
    except Exception as e:
        print(f"Could not load physformer pred.npy: {e}")
        phys_pred = None
    
    # 3. Informer-Post (Clipped)
    # Clipping the whole prediction at 0
    inf_pred_post = np.maximum(inf_pred, 0)
    
    def get_stratified_rvm(p, l, thresholds):
        # p shape: (B, L, C)
        diff = np.abs(p[:, 1:, :] - p[:, :-1, :]) # Shape: (B, L-1, C)
        
        # Calculate RVM values (diff > limit)
        violations = np.maximum(diff - l, 0)
        
        # Determine masks for boundary vs non-boundary
        # We consider a transition boundary if either current or next step is within threshold
        # Using abs(p) since true bounds are 0.
        is_boundary_start = np.abs(p[:, :-1, :]) < thresholds
        is_boundary_end = np.abs(p[:, 1:, :]) < thresholds
        
        mask_boundary = is_boundary_start | is_boundary_end
        mask_non_boundary = ~mask_boundary
        
        rvm_all = np.mean(violations) if violations.size > 0 else 0
        rvm_boundary = np.sum(violations * mask_boundary) / np.maximum(np.sum(mask_boundary), 1)
        rvm_non_boundary = np.sum(violations * mask_non_boundary) / np.maximum(np.sum(mask_non_boundary), 1)
        
        return rvm_all, rvm_boundary, rvm_non_boundary, np.mean(mask_boundary)
        
    print("\n--- Informer ---")
    inf_all, inf_b, inf_nb, inf_mask_ratio = get_stratified_rvm(inf_pred, limits, boundary_threshold)
    print(f"Overall RVM: {inf_all:.6f}")
    print(f"Boundary RVM: {inf_b:.6f} (Coverage: {inf_mask_ratio:.2%})")
    print(f"Non-Boundary RVM: {inf_nb:.6f}")

    print("\n--- Informer-Post (Clipped) ---")
    post_all, post_b, post_nb, post_mask_ratio = get_stratified_rvm(inf_pred_post, limits, boundary_threshold)
    print(f"Overall RVM: {post_all:.6f}")
    print(f"Boundary RVM: {post_b:.6f} (Coverage: {post_mask_ratio:.2%})")
    print(f"Non-Boundary RVM: {post_nb:.6f}")
    
    if phys_pred is not None:
        print("\n--- PhysFormer ---")
        phys_all, phys_b, phys_nb, phys_mask_ratio = get_stratified_rvm(phys_pred, limits, boundary_threshold)
        print(f"Overall RVM: {phys_all:.6f}")
        print(f"Boundary RVM: {phys_b:.6f} (Coverage: {phys_mask_ratio:.2%})")
        print(f"Non-Boundary RVM: {phys_nb:.6f}")

if __name__ == '__main__':
    calculate_stratified_rvm()
