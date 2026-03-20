import os
import torch
import numpy as np
import pandas as pd
from scripts.run_benchmark import get_args
from experiments.exp_baseline import Exp_Baselines
from experiments.exp_PhysFormer import Exp_PhysFormer
from models.src.utils.metrics import metric

def get_ramp_limits(train_data, percentile=99.9, multiplier=1.5):
    raw_train = train_data.inverse_transform(train_data.data_x)[:, :3]
    diff = np.abs(raw_train[1:] - raw_train[:-1])
    limits = np.percentile(diff, percentile, axis=0) * multiplier
    return limits

def load_results(exp, setting):
    folder_path = os.path.join(exp.args.checkpoints, setting)
    preds = np.load(os.path.join(folder_path, 'real_prediction.npy'))
    trues = np.load(os.path.join(folder_path, 'true_prediction.npy'))
    return preds, trues

def analyze_sensitivity():
    args = get_args()
    args.use_gpu = torch.cuda.is_available()
    args.gpu = 0
    args.batch_size = 32
    
    # Standard settings
    args.model = 'Informer' # Placeholder
    args.seq_len = 672
    args.pred_len = 96
    args.label_len = 48
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 3
    args.d_ff = 2048
    args.dropout = 0.05
    args.activation = 'gelu'
    args.output_attention = False
    args.do_predict = False
    args.checkpoint_name = 'benchmark'
    args.root_path = './'
    args.data_path = 'data/vpp_dataset_3years.csv'
    args.features = 'M'
    args.target = None
    args.freq = '15min'
    args.num_workers = 0
    args.use_amp = True
    args.patience = 3
    args.learning_rate = 1e-3
    args.train_epochs = 1
    
    # 1. Get Train Data for limits
    exp_dummy = Exp_Baselines(args)
    train_data, _ = exp_dummy._get_data(flag='train')
    scaler = train_data.scaler
    
    # Threshold 1: Original (99.9th * 1.5)
    limits_orig = get_ramp_limits(train_data, 99.9, 1.5)
    # Threshold 2: Stricter (99.5th * 1.0)
    limits_strict = get_ramp_limits(train_data, 99.5, 1.0)
    
    print(f"Original Thresholds (99.9th * 1.5): {limits_orig}")
    print(f"Stricter Thresholds (99.5th * 1.0): {limits_strict}")
    
    models = [
        ('PhysFormer', 'e:/Py_program/Soft-phys-CFC-Informer/exp_results/PhysFormer/checkpoints/PhysFormer_full_seed2024'),
        ('Informer-Post', 'e:/Py_program/Soft-phys-CFC-Informer/exp_results/informer/checkpoints/Informer_full_seed2024'),
        ('iTransformer', 'e:/Py_program/Soft-phys-CFC-Informer/exp_results/iTransformer_vpp_dataset_3years_sl672_pl96_vpp'),
    ]
    
    results = []
    
    for name, folder_path in models:
        print(f"\nProcessing {name}...")
        
        try:
            # Load normalized preds if they exist
            if name == 'iTransformer':
                norm_preds = np.load(os.path.join(folder_path, 'pred.npy'))
                norm_trues = np.load(os.path.join(folder_path, 'true.npy'))
            else:
                norm_preds = np.load(os.path.join(folder_path, 'pred.npy'))
                norm_trues = np.load(os.path.join(folder_path, 'true.npy'))

            # For ALL models, let's inverse transform from 'pred.npy' to be consistent
            N, L, C = norm_preds.shape
            
            # Inverse Transform preds
            flat_preds = norm_preds.reshape(-1, C)
            dummy = np.zeros((flat_preds.shape[0], 3))
            preds = scaler.inverse_transform(np.concatenate([flat_preds, dummy], axis=1))[:, :3]
            preds = preds.reshape(N, L, C)
            
            # Inverse Transform trues
            flat_trues = norm_trues.reshape(-1, C)
            trues = scaler.inverse_transform(np.concatenate([flat_trues, dummy], axis=1))[:, :3]
            trues = trues.reshape(N, L, C)

            if name == 'Informer-Post':
                # Apply Hard-clipping to Informer
                mean_val = scaler.mean_[:3]
                std_val = scaler.scale_[:3]
                zero_vals = -mean_val / (std_val + 1e-8)
                
                norm_preds_post = norm_preds.copy()
                for c in [1, 2]: # PV and Wind
                    norm_preds_post[:, :, c] = np.where(norm_preds[:, :, c] < zero_vals[c], zero_vals[c], norm_preds[:, :, c])
                
                # Inverse Transform post-clipped
                flat_post = norm_preds_post.reshape(-1, C)
                preds = scaler.inverse_transform(np.concatenate([flat_post, dummy], axis=1))[:, :3]
                preds = preds.reshape(N, L, C)

            # Evaluate with both limits
            _, _, _, _, rvm_orig = metric(preds, trues, ramp_limits=limits_orig)
            _, _, _, _, rvm_strict = metric(preds, trues, ramp_limits=limits_strict)
            
            results.append({
                'Model': name,
                'RVM (Orig)': rvm_orig,
                'RVM (Strict)': rvm_strict
            })
            print(f"  RVM (Orig):   {rvm_orig:.8f}")
            print(f"  RVM (Strict): {rvm_strict:.8f}")
            
        except FileNotFoundError as e:
            print(f"  Error: Results for {name} not found. {e}")

    df = pd.DataFrame(results)
    print("\nSummary Table:")
    print(df.to_string(index=False))

if __name__ == '__main__':
    analyze_sensitivity()
