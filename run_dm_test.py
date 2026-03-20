import numpy as np
import scipy.stats as stats
import os

def dm_test(actual, pred1, pred2, h=1):
    e1 = np.mean((actual - pred1)**2, axis=(1, 2))
    e2 = np.mean((actual - pred2)**2, axis=(1, 2))
    
    d = e1 - e2
    mean_d = np.mean(d)
    
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
            autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov

    gamma = []
    for lag in range(0, h):
        gamma.append(autocovariance(d, len(d), lag, mean_d))
    
    V_d = gamma[0] + 2 * sum(gamma[1:])
    if V_d == 0:
        return 0, 1.0, 1.0
        
    DM_stat = mean_d / np.sqrt(V_d / len(d))
    p_value_two_sided = 2 * stats.norm.cdf(-abs(DM_stat))
    p_value_one_sided = stats.norm.cdf(DM_stat) if DM_stat < 0 else 1.0 - stats.norm.cdf(DM_stat)
    
    return DM_stat, p_value_two_sided, p_value_one_sided

if __name__ == "__main__":
    base_dir = "exp_results/"
    
    gt_path = os.path.join(base_dir, "PhysFormer/checkpoints/PhysFormer_full_seed2024/true.npy")
    phys_path = os.path.join(base_dir, "PhysFormer/checkpoints/PhysFormer_full_seed2024/pred.npy")
    inf_path = os.path.join(base_dir, "Informer_vpp_dataset_3years_sl672_pl96_vpp/pred.npy")
    itrans_path = os.path.join(base_dir, "iTransformer_vpp_dataset_3years_sl672_pl96_vpp/pred.npy")
    
    try:
        y_true = np.load(gt_path)
        y_phys = np.load(phys_path)
        y_inf = np.load(inf_path)
        y_itrans = np.load(itrans_path)
        
        # PhysFormer vs Informer
        dm_stat_1, p_val_two_1, p_val_one_1 = dm_test(y_true, y_phys, y_inf)
        print(f"\n--- PhysFormer vs Informer ---")
        print(f"DM Statistic: {dm_stat_1:.6f}")
        print(f"P-value (two-sided): {p_val_two_1:.6e}")
        
        # PhysFormer vs iTransformer
        dm_stat_2, p_val_two_2, p_val_one_2 = dm_test(y_true, y_phys, y_itrans)
        print(f"\n--- PhysFormer vs iTransformer ---")
        print(f"DM Statistic: {dm_stat_2:.6f}")
        print(f"P-value (two-sided): {p_val_two_2:.6e}")
        
    except Exception as e:
        print(f"Error running DM test: {e}")
