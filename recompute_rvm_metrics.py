import os
import numpy as np
import pandas as pd
from scripts.run_benchmark import get_args
from experiments.exp_baseline import Exp_Baselines
from models.src.utils.metrics import metric

def get_ramp_limits(train_data, percentile=99.9, multiplier=1.5):
    raw_train = train_data.inverse_transform(train_data.data_x)[:, :3]
    diff = np.abs(raw_train[1:] - raw_train[:-1])
    limits = np.percentile(diff, percentile, axis=0) * multiplier
    return limits

def calculate_rvm_magnitude(pred, limits):
    # RVM Magnitude = Mean(Relu(|diff| - limit))
    if pred.ndim == 3:
        diff = np.abs(pred[:, 1:, :] - pred[:, :-1, :])
    else:
        diff = np.abs(pred[1:, :] - pred[:-1, :])
    
    violation_mag = np.maximum(diff - limits.reshape(1, 1, -1), 0)
    return np.mean(violation_mag)

def run_analysis():
    # Setup args
    args = get_args()
    args.model = 'Informer'
    args.checkpoint_name = 'benchmark'
    args.root_path = 'e:/Py_program/Soft-phys-CFC-Informer'
    args.data_path = 'data/vpp_dataset_3years.csv'
    args.features = 'M'
    args.target = None
    args.freq = '15min'
    args.seq_len = 672
    args.pred_len = 96
    args.label_len = 48
    args.d_model = 512
    args.use_gpu = False # Use CPU for simplicity
    
    exp = Exp_Baselines(args)
    train_data, _ = exp._get_data(flag='train')
    scaler = train_data.scaler
    
    # 1. Thresholds
    limits_loose = get_ramp_limits(train_data, 99.9, 1.5)
    limits_strict = get_ramp_limits(train_data, 99.5, 1.0)
    
    print(f"Limits (Loose: 99.9th*1.5): {limits_loose}")
    print(f"Limits (Strict: 99.5th*1.0): {limits_strict}")
    
    models = [
        ('PhysFormer', 'e:/Py_program/Soft-phys-CFC-Informer/exp_results/PhysFormer/checkpoints/PhysFormer_full_seed2024'),
        ('Informer', 'e:/Py_program/Soft-phys-CFC-Informer/exp_results/Informer_vpp_dataset_3years_sl672_pl96_vpp'),
        ('iTransformer', 'e:/Py_program/Soft-phys-CFC-Informer/exp_results/iTransformer_vpp_dataset_3years_sl672_pl96_vpp'),
    ]
    
    all_results = []
    
    for name, path in models:
        print(f"\nModel: {name}")
        p_path = os.path.join(path, 'pred.npy')
        t_path = os.path.join(path, 'true.npy')
        
        if not os.path.exists(p_path):
            print(f"Missing {p_path}")
            continue
            
        pred_norm = np.load(p_path)
        true_norm = np.load(t_path)
        
        # Inverse Scaling
        N, L, C = pred_norm.shape
        # Use dummy for extra 3 columns
        dummy = np.zeros((N*L, 3))
        
        p_flat = pred_norm.reshape(-1, C)
        p_phys = scaler.inverse_transform(np.concatenate([p_flat, dummy], axis=1))[:, :3].reshape(N, L, C)
        
        t_flat = true_norm.reshape(-1, C)
        t_phys = scaler.inverse_transform(np.concatenate([t_flat, dummy], axis=1))[:, :3].reshape(N, L, C)
        
        # Metrics for Raw Model
        _, _, _, _, rvr_loose = metric(p_phys, t_phys, ramp_limits=limits_loose)
        _, _, _, _, rvr_strict = metric(p_phys, t_phys, ramp_limits=limits_strict)
        rvm_mag_strict = calculate_rvm_magnitude(p_phys, limits_strict)
        
        all_results.append({
            'Model': name,
            'RVR_Loose': rvr_loose,
            'RVR_Strict': rvr_strict,
            'RVM_Mag_Strict': rvm_mag_strict
        })
        
        # For Informer, also calculate Informer-Post
        if name == 'Informer':
            print("  + Calculating Informer-Post...")
            mean_val = scaler.mean_[:3]
            std_val = scaler.scale_[:3]
            zero_vals = -mean_val / (std_val + 1e-8)
            
            p_norm_post = pred_norm.copy()
            for c in [1, 2]: # PV and Wind
                p_norm_post[:, :, c] = np.where(pred_norm[:, :, c] < zero_vals[c], zero_vals[c], pred_norm[:, :, c])
            
            p_flat_post = p_norm_post.reshape(-1, C)
            p_phys_post = scaler.inverse_transform(np.concatenate([p_flat_post, dummy], axis=1))[:, :3].reshape(N, L, C)
            
            _, _, _, _, rvr_loose_p = metric(p_phys_post, t_phys, ramp_limits=limits_loose)
            _, _, _, _, rvr_strict_p = metric(p_phys_post, t_phys, ramp_limits=limits_strict)
            rvm_mag_strict_p = calculate_rvm_magnitude(p_phys_post, limits_strict)
            
            all_results.append({
                'Model': 'Informer-Post',
                'RVR_Loose': rvr_loose_p,
                'RVR_Strict': rvr_strict_p,
                'RVM_Mag_Strict': rvm_mag_strict_p
            })

    df = pd.DataFrame(all_results)
    print("\nFinal Results:")
    print(df.to_string(index=False))
    df.to_csv("final_rvm_results.csv", index=False)

if __name__ == '__main__':
    run_analysis()
