import numpy as np
import os
import pandas as pd

def calculate_metrics():
    base_path = 'e:/Py_program/Soft-phys-CFC-Informer/exp_results/iTransformer_vpp_dataset_3years_sl672_pl96_vpp'
    data_path = 'e:/Py_program/Soft-phys-CFC-Informer/data/vpp_dataset_3years.csv'
    
    if not os.path.exists(base_path):
        print(f"Error: Path {base_path} does not exist.")
        return

    metrics_path = os.path.join(base_path, 'metrics.npy')
    pred_path = os.path.join(base_path, 'pred.npy')
    true_path = os.path.join(base_path, 'true.npy')

    if os.path.exists(metrics_path):
        metrics = np.load(metrics_path, allow_pickle=True)
        # Order: [mae, mse, rmse, mape, mspe]
        print("--- Loaded Metrics (Original) ---")
        print(f"MSE: {metrics[1]:.6f}")
        print(f"MAE: {metrics[0]:.6f}")
    
    if os.path.exists(pred_path) and os.path.exists(true_path):
        pred = np.load(pred_path)
        true = np.load(true_path)
        
        # BVR Calculation
        # PV and Wind are at indices 1 and 2
        violations_pv = np.sum(pred[:, :, 1] < 0)
        violations_wind = np.sum(pred[:, :, 2] < 0)
        total_points_channel = pred[:, :, 0].size
        bvr = (violations_pv + violations_wind) / (2 * total_points_channel) * 100
        print(f"\nCalculated BVR: {bvr:.4f}%")
        
        # Calculate Refined RVM and MVS (Magnitude)
        # Load Train Data to get limits for RVM
        df_raw = pd.read_csv(data_path)
        num_train = int(len(df_raw) * 0.7)
        train_data = df_raw.iloc[:num_train, 1:4].values # Load, PV, Wind
        
        diff_train = np.abs(train_data[1:] - train_data[:-1])
        # 99.9th percentile * 1.5
        limits = np.percentile(diff_train, 99.9, axis=0) * 1.5
        
        # RVM: Mean Violation magnitude
        diff_pred = np.abs(pred[:, 1:, :] - pred[:, :-1, :])
        rvm_violations = np.maximum(diff_pred - limits, 0)
        rvm_mag = np.mean(rvm_violations)
        
        # MVS: Mean Violation Size for BVR (Static)
        # Only PV and Wind (indices 1 and 2) have 0-floor constraint
        mvs_mag = np.mean(np.maximum(-pred[:, :, 1:3], 0))
        
        print("\n--- Refined Compliance Analysis ---")
        print(f"RVM (Magnitude): {rvm_mag:.8f}")
        print(f"MVS (Magnitude): {mvs_mag:.8f}")
        
        # Specifically check for non-zero BVR/RVM
        bvr_count = np.sum(pred[:, :, 1:3] < 0)
        total_points = pred[:, :, 1:3].size
        print(f"Exact BVR: {bvr_count / total_points * 100:.4f}% ({bvr_count}/{total_points})")

if __name__ == "__main__":
    calculate_metrics()
