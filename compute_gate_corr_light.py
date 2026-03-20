import os
import sys
import torch
import numpy as np
from scipy import stats

from scripts.run_benchmark import get_args
from experiments.exp_PhysFormer import Exp_PhysFormer

def compute_from_model_light():
    base_dir = './exp_results'
    checkpoint_dir = f'{base_dir}/PhysFormer/checkpoints/PhysFormer_full_seed2024'
    ckpt_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    
    sys.argv = ['compute_gate_corr.py']
    args = get_args()
    args.model = 'PhysFormer'
    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0
    args.batch_size = 32 # Smaller batch for CPU
    args.num_workers = 0
    args.checkpoints = f'{base_dir}/PhysFormer/checkpoints'
    args.checkpoint_name = 'PhysFormer_full_seed2024'
    args.root_path = './'
    args.data_path = 'data/vpp_dataset_3years.csv'
    args.features = 'M'
    args.seq_len = 672
    args.pred_len = 96
    
    args.enc_in = 6
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 3
    args.d_ff = 2048
    args.factor = 5
    args.dropout = 0.10
    args.attn = 'full'
    args.embed = 'custom'
    args.activation = 'gelu'
    args.output_attention = False
    args.distil = False
    args.mix = True
    args.use_rope = True
    args.rope_base = 10000
    args.freq = 'h'
    args.ablation_no_phys_stream = False
    args.ablation_no_pgcc = False
    args.ablation_no_future_glu = False
    args.ablation_no_curriculum = False
    args.ablation_fixed_phys = False

    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        exp = Exp_PhysFormer(args)
    finally:
        sys.stdout = old_stdout

    model = exp.model
    device = exp.device
    state_dict = torch.load(ckpt_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    test_data, test_loader = exp._get_data(flag='test')
    all_gate_pv = []
    all_irr = []

    print(">>> Extracting Gate_PV (Light mode)...")
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            if i >= 20: break # Only 20 batches
            batch_stat, batch_weather_hist, batch_weather_future, batch_y, batch_x_mark, batch_y_mark = batch_data
            batch_stat = batch_stat.float().to(device)
            batch_weather_hist = batch_weather_hist.float().to(device)
            batch_weather_future = batch_weather_future.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)

            outputs, reg_loss, gates_info = model(
                x_stat=batch_stat,
                x_weather_hist=batch_weather_hist,
                x_weather_future=batch_weather_future,
                x_mark_enc=batch_x_mark,
                alpha=0.1, 
                return_gates=True
            )
            if 'pv_seq_batch' in gates_info and 'irr_seq_batch' in gates_info:
                all_gate_pv.append(gates_info['pv_seq_batch'])
                all_irr.append(gates_info['irr_seq_batch'])
    
    gate_pv_array = np.concatenate(all_gate_pv, axis=0)
    irr_array = np.concatenate(all_irr, axis=0)

    gate_flat = gate_pv_array.flatten()
    irr_flat = irr_array.flatten()
    r_global, p_global = stats.pearsonr(gate_flat, irr_flat)

    print(f"Sample points: {len(gate_flat)}")
    print(f"Global Pearson r = {r_global:.4f}")
    
    # Save for plotting
    np.save('fig3_gate_data.npy', gate_flat)
    np.save('fig3_irr_data.npy', irr_flat)

if __name__ == '__main__':
    compute_from_model_light()
