import pandas as pd
import numpy as np
import os

def analyze_degenerate_variance():
    data_path = 'e:/Py_program/Soft-phys-CFC-Informer/data/vpp_dataset_3years.csv'
    df = pd.read_csv(data_path)
    
    target_cols = ['load_mw', 'pv_mw', 'wind_mw']
    
    # Global training stats (for the BPAR floor)
    num_train = int(len(df) * 0.7)
    df_train = df.iloc[:num_train]
    global_mu = df_train[target_cols].mean()
    global_std = df_train[target_cols].std()
    
    # We want to find windows of length seq_len=672 in the TEST set
    # where the local variance is extremely low.
    test_start = num_train
    df_test = df.iloc[test_start:]
    
    seq_len = 672
    threshold = 1e-3
    
    print(f"--- Degenerate Variance Analysis (Threshold={threshold}) ---")
    
    degenerate_windows = {col: 0 for col in target_cols}
    total_windows = len(df_test) - seq_len + 1
    
    for i in range(total_windows):
        window = df_test.iloc[i : i+seq_len][target_cols]
        window_std = window.std()
        
        for col in target_cols:
            if window_std[col] < threshold:
                degenerate_windows[col] += 1
                
    print(f"Total test windows checked: {total_windows}")
    for col in target_cols:
        perc = (degenerate_windows[col] / total_windows) * 100
        print(f"{col}: {degenerate_windows[col]} degenerate windows ({perc:.4f}%)")
        
    print("\n--- Physical Interpretation ---")
    print("If a window has degenerate variance (sigma < 1e-3), it means the power is essentially constant.")
    print("For PV, this happens at night (0 MW, zero variance).")
    print("BPAR handles this via the global floor -mu/std. Even if local sigma is 0, the global BPAR boundary")
    print("still forces the output to be >= 0 MW in the original scale.")

if __name__ == "__main__":
    analyze_degenerate_variance()
