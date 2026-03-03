import os
import numpy as np

def compute_net_mae(pred, true):
    # Calculate Net Load: Load (idx 0) - PV (idx 1) - Wind (idx 2)
    net_pred = pred[:, :, 0] - pred[:, :, 1] - pred[:, :, 2]
    net_true = true[:, :, 0] - true[:, :, 1] - true[:, :, 2]
    return np.mean(np.abs(net_pred - net_true))

def get_gate_r(folder_path):
    gate_pv_file = os.path.join(folder_path, 'vis_gate_pv.npy')
    irr_file = os.path.join(folder_path, 'vis_irr.npy')
    
    if not os.path.exists(gate_pv_file) or not os.path.exists(irr_file):
        return '-'
        
    gate_pv = np.load(gate_pv_file, allow_pickle=True)
    irr = np.load(irr_file, allow_pickle=True)
    
    if len(gate_pv) == 0 or len(irr) == 0:
        return '-'
        
    gate_pv = np.array(gate_pv)
    irr = np.array(irr)
    
    try:
        corr_all = np.corrcoef(gate_pv.flatten(), irr.flatten())[0, 1]
        
        # Check for nan
        if np.isnan(corr_all):
            return '-'
            
        return f"{corr_all:.4f}"
    except Exception:
        return '-'

def collect_results():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir = os.path.join(project_root, 'exp_results', 'PhysFormer', 'checkpoints')
    
    experiments = [
        ("Full PhysFormer", "PhysFormer_full_seed2024"),
        ("w/o Physics Stream", "PhysFormer_ablation_V1_no_phys"),
        ("w/o PGCC", "PhysFormer_ablation_V2_no_pgcc"),
        ("w/o Future GLU", "PhysFormer_ablation_V3_no_future_glu"),
        ("w/o Curriculum", "PhysFormer_ablation_V4_no_curriculum"),
    ]
    
    results = []
    
    for variant, ckpt_name in experiments:
        folder_path = os.path.join(base_dir, ckpt_name)
        
        if not os.path.exists(folder_path):
            results.append({
                'Variant': variant,
                'MSE': 'N/A',
                'BVR%': 'N/A',
                'NET_MAE': 'N/A',
                'gate_r': 'N/A'
            })
            continue
            
        metrics_file = os.path.join(folder_path, 'metrics.npy')
        pred_file = os.path.join(folder_path, 'pred.npy')
        true_file = os.path.join(folder_path, 'true.npy')
        
        mse = '-'
        bvr = '-'
        net_mae = '-'
        gate_r = '-'
        
        # Load standard metrics
        if os.path.exists(metrics_file):
            metrics = np.load(metrics_file, allow_pickle=True)
            # metric() returns -> mae, mse, rmse, mape, mspe, bvr, rvr
            mse = f"{metrics[1]:.4f}"
            bvr = f"{metrics[5]:.2f}"
            
        # Compute NET_MAE
        if os.path.exists(pred_file) and os.path.exists(true_file):
            pred = np.load(pred_file)
            true = np.load(true_file)
            val_net_mae = compute_net_mae(pred, true)
            net_mae = f"{val_net_mae:.4f}"
            
        # 对于不包含 PGCC 的变体，gate_r 不适用
        if variant in ("w/o PGCC", "w/o Physics Stream"):
            gate_r = 'N/A'
        else:
            gate_r = get_gate_r(folder_path)
        
        results.append({
            'Variant': variant,
            'MSE': mse,
            'BVR%': bvr,
            'NET_MAE': net_mae,
            'gate_r': gate_r
        })
        
    # Format and print the table clearly
    print("\n" + "="*80)
    print(f"  {'Variant':<25} {'MSE':<10} {'BVR%':<10} {'NET_MAE':<10} {'gate_r':<10}")
    print("-" * 80)
    for row in results:
        print(f"  {row['Variant']:<25} {row['MSE']:<10} {row['BVR%']:<10} {row['NET_MAE']:<10} {row['gate_r']:<10}")
    print("="*80 + "\n")

if __name__ == '__main__':
    collect_results()
