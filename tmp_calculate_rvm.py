import os
import sys
import torch
import numpy as np
from scripts.run_benchmark import get_args
from experiments.exp_baseline import Exp_Baselines
from models.src.utils.metrics import metric

def calculate_informer_post():
    args = get_args()
    args.model = 'Informer'
    args.root_path = './'
    args.data_path = 'data/vpp_dataset_3years.csv'
    args.features = 'M'
    args.target = None
    args.freq = '15min'
    args.num_workers = 0
    args.use_amp = True
    args.patience = 3
    args.learning_rate = 1e-3
    args.batch_size = 32
    args.train_epochs = 1
    args.seq_len = 672
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 3
    args.pred_len = 96
    args.label_len = 48 # Needed for Informer
    args.checkpoint_name = 'Informer_full_seed2024'
    args.use_gpu = torch.cuda.is_available()
    args.gpu = 0
    args.batch_size = 32
    
    # Informer Checkpoint
    args.checkpoints = './exp_results/Informer/checkpoints'
    setting = 'Informer_full_seed2024'
    
    exp = Exp_Baselines(args)
    model = exp.model
    device = exp.device
    
    # Load model
    load_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
    if not os.path.exists(load_path):
        print(f"Error: Checkpoint not found at {load_path}")
        return
        
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()
    
    test_data, test_loader = exp._get_data(flag='test')
    scaler = test_data.scaler
    
    # Scaling parameters for post-processing
    mean_val = scaler.mean_[:3]
    std_val = scaler.scale_[:3]
    zero_vals = -mean_val / (std_val + 1e-8)
    zero_tensor = torch.tensor(zero_vals, dtype=torch.float32).to(device).view(1, 1, 3)
    
    preds_post = []
    trues = []
    
    print("Running Informer-Post inference on test set...")
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            # Decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            # Debug Prints
            if i == 0:
                print(f"DEBUG: batch_x type: {type(batch_x)}, shape: {batch_x.shape if batch_x is not None else 'None'}")
                print(f"DEBUG: batch_x_mark type: {type(batch_x_mark)}, shape: {batch_x_mark.shape if batch_x_mark is not None else 'None'}")
                print(f"DEBUG: dec_inp type: {type(dec_inp)}, shape: {dec_inp.shape if dec_inp is not None else 'None'}")
                print(f"DEBUG: batch_y_mark type: {type(batch_y_mark)}, shape: {batch_y_mark.shape if batch_y_mark is not None else 'None'}")

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -args.pred_len:, :3] # [B, 96, 3]
            
            # Apply Hard-clipping (Informer-Post)
            # Only clip PV(1) and Wind(2)
            mask = torch.ones_like(outputs)
            mask[:, :, 0] = 0.0 # Don't clip Load
            outputs_post = torch.where((outputs < zero_tensor) & (mask > 0.5), zero_tensor, outputs)
            
            preds_post.append(outputs_post.cpu().numpy())
            trues.append(batch_y[:, -args.pred_len:, :3].cpu().numpy())
            
    preds_post = np.concatenate(preds_post, axis=0) # [N, 96, 3]
    trues = np.concatenate(trues, axis=0)
    
    # Inverse Transform
    N, L, C = preds_post.shape
    preds_flat = preds_post.reshape(-1, C)
    trues_flat = trues.reshape(-1, C)
    
    dummy = np.zeros((preds_flat.shape[0], 3))
    preds_phys = scaler.inverse_transform(np.concatenate([preds_flat, dummy], axis=1))[:, :3]
    trues_phys = scaler.inverse_transform(np.concatenate([trues_flat, dummy], axis=1))[:, :3]
    
    preds_phys = preds_phys.reshape(N, L, C)
    trues_phys = trues_phys.reshape(N, L, C)
    
    # Calculate Ramp Limits from Train Data (as specified in paper)
    train_data, _ = exp._get_data(flag='train')
    raw_train = train_data.inverse_transform(train_data.data_x)[:, :3]
    diff = np.abs(raw_train[1:] - raw_train[:-1])
    ramp_limits = np.percentile(diff, 99.9, axis=0) * 1.5
    
    # Calculate Metrics
    mae, mse, rmse, bvr, rvr = metric(preds_phys, trues_phys, ramp_limits=ramp_limits)
    
    print("\n" + "="*40)
    print(f"RESULTS FOR INFORMER-POST")
    print("="*40)
    print(f"MSE: {mse:.6f}")
    print(f"BVR: {bvr:.4f}%")
    print(f"RVM (RVR): {rvr:.6f}")
    print("="*40)

if __name__ == '__main__':
    calculate_informer_post()
