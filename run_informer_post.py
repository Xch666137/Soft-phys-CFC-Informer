import os
import numpy as np

# Load true and pred from Informer
base_dir = './exp_results'
informer_dir = f'{base_dir}/Informer_vpp_dataset_3years_sl672_pl96_vpp'
out_dir = f'{base_dir}/Informer-Post'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

true_path = os.path.join(informer_dir, 'true.npy')
pred_path = os.path.join(informer_dir, 'pred.npy')

if os.path.exists(pred_path) and os.path.exists(true_path):
    true_data = np.load(true_path)
    pred_data = np.load(pred_path)
    
    # Post-processing: Apply Hard Zero-Clipping
    post_pred = pred_data.copy()
    
    # Clip PV and Wind completely to 0 or positive
    post_pred[:, :, 1] = np.maximum(post_pred[:, :, 1], 0)
    post_pred[:, :, 2] = np.maximum(post_pred[:, :, 2], 0)
    
    # Calculate MSE, MAE, RMSE
    mae = np.mean(np.abs(post_pred - true_data))
    mse = np.mean((post_pred - true_data)**2)
    rmse = np.sqrt(mse)
    
    # BVR
    violations = post_pred[:, :, 1:3][post_pred[:, :, 1:3] < 0]
    bvr = len(violations) / post_pred[:, :, 1:3].size * 100
    
    print(f"Informer-Post MSE: {mse:.4f}")
    print(f"Informer-Post MAE: {mae:.4f}")
    print(f"Informer-Post RMSE: {rmse:.4f}")
    print(f"Informer-Post BVR: {bvr:.4f}%")
    
    np.save(os.path.join(out_dir, 'pred.npy'), post_pred)
    np.save(os.path.join(out_dir, 'true.npy'), true_data)
    
    # Update metrics in metrics.npy
    old_metrics_path = os.path.join(informer_dir, 'metrics.npy')
    if os.path.exists(old_metrics_path):
        old_metrics = np.load(old_metrics_path, allow_pickle=True)
        new_metrics = old_metrics.copy()
        new_metrics[0] = mae
        new_metrics[1] = mse
        new_metrics[2] = rmse
        new_metrics[5] = bvr
        np.save(os.path.join(out_dir, 'metrics.npy'), new_metrics)
        print("Data saved to Informer-Post/")
    else:
        print("metrics.npy not found for Informer!")
else:
    print("Informer's pred.npy or true.npy missing!")
