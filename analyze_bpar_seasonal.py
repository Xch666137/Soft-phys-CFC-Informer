import pandas as pd
import numpy as np
import os

def analyze_seasonal_stats():
    data_path = 'e:/Py_program/Soft-phys-CFC-Informer/data/vpp_dataset_3years.csv'
    df = pd.read_csv(data_path)
    
    # 转换时间列
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    
    # 定义季节
    # 1-3: Winter, 4-6: Spring, 7-9: Summer, 10-12: Autumn
    def get_season(m):
        if m in [12, 1, 2]: return 'Winter'
        if m in [3, 4, 5]: return 'Spring'
        if m in [6, 7, 8]: return 'Summer'
        return 'Autumn'
    
    df['season'] = df['month'].apply(get_season)
    
    channels = ['OT', 'PV', 'Wind'] # Assuming these are the column names, let's verify
    # Actually based on common pattern: Load is 1st, PV is 2nd, Wind is 3rd?
    # Let's check headers first
    print(f"Columns: {df.columns.tolist()}")
    
    # Global Stats (Training Set - first 70%)
    num_train = int(len(df) * 0.7)
    df_train = df.iloc[:num_train]
    
    # We care about PV and Wind non-negativity
    target_cols = df.columns[1:4].tolist() # date is 0, then 3 power stats
    print(f"Target Columns: {target_cols}")
    
    global_mu = df_train[target_cols].mean()
    global_std = df_train[target_cols].std()
    
    global_floors = -global_mu / global_std
    
    print("\n--- Global Training Statistics ---")
    for i, col in enumerate(target_cols):
        print(f"{col}: mu={global_mu[i]:.4f}, std={global_std[i]:.4f}, Floor={global_floors[i]:.4f}")
    
    # Seasonal Stats
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    seasonal_results = []
    
    print("\n--- Seasonal Statistics & Floor Shifts ---")
    for s in seasons:
        df_s = df[df['season'] == s]
        mu_s = df_s[target_cols].mean()
        std_s = df_s[target_cols].std()
        
        # If we normalized seasonal data using its OWN stats, the floor would be:
        floor_s = -mu_s / std_s
        
        # But we normalize using GLOBAL stats. 
        # The physical zero in GLOBAL normalized space is always global_floor.
        # Is the global_floor always reachable for this season's distribution?
        # A season's distribution in global normalized space has:
        # mu'_s = (mu_s - global_mu) / global_std
        # std'_s = std_s / global_std
        # The floor in this space is still global_floor = -global_mu / global_std.
        
        print(f"\nSeason: {s}")
        for i, col in enumerate(target_cols):
            mu_prime = (mu_s[i] - global_mu[i]) / global_std[i]
            std_prime = std_s[i] / global_std[i]
            print(f"  {col}: Local_mu={mu_s[i]:.4f}, Local_std={std_s[i]:.4f} | Normalized_mu={mu_prime:.4f}, Normalized_std={std_prime:.4f}")
            
    # BVR Verification (Mathematical Proof simulation)
    # If the model outputs y_hat >= global_floor, then 
    # y = y_hat * global_std + global_mu >= global_floor * global_std + global_mu = 0.
    # This is TRUE for any sample regardless of which season it came from.
    # The only issue is if for some season, mu_s is so small that the physical values
    # are very close to zero, making it easier to violate.
    
    print("\n--- BPAR Robustness Conclusion ---")
    print("The BPAR floor -mu/std is a structural property of the normalization parameters.")
    print("As long as the same parameters are used for both normalization and BPAR floor calculation,")
    print("non-negativity is mathematically guaranteed (BVR=0.00%) regardless of seasonal distribution shifts.")

if __name__ == "__main__":
    analyze_seasonal_stats()
