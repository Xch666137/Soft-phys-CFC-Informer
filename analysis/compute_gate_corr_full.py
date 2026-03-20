import os
import sys
import torch
import numpy as np
from scipy import stats

from scripts.run_benchmark import get_args
from experiments.exp_PhysFormer import Exp_PhysFormer

def compute_from_model():
    """
    通过加载实验设置、初始化 PhysFormer 模型并过一遍测试集，提取每个样本的
    gate_pv 和对应的 irradiance 序列，然后计算它们的 Pearson 相关系数。
    """
    base_dir = './exp_results'
    checkpoint_dir = f'{base_dir}/PhysFormer/checkpoints/PhysFormer_full_seed2024'
    ckpt_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    
    if not os.path.exists(ckpt_path):
        print(f"Error: 找不到模型权重文件 -> {ckpt_path}")
        return

    print(">>> 正在加载测试集与 PhysFormer 权重...")
    sys.argv = ['compute_gate_corr.py']
    args = get_args()
    args.model = 'PhysFormer'
    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0
    args.batch_size = 128
    args.num_workers = 4
    args.checkpoints = f'{base_dir}/PhysFormer/checkpoints'
    args.checkpoint_name = 'PhysFormer_full_seed2024'

    # 数据和模型主体超参数 (参考 run_single_train.sh)
    args.root_path = './'
    args.data_path = 'data/vpp_dataset_3years.csv'
    args.features = 'M'
    args.target = None
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
    
    # 物理模型对应的 Ablation Flags
    args.ablation_no_phys_stream = False
    args.ablation_no_pgcc = False
    args.ablation_no_future_glu = False
    args.ablation_no_curriculum = False
    args.ablation_fixed_phys = False

    # 初始化实验类以便拿到测试集的数据加载器和设定好的模型
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        exp = Exp_PhysFormer(args)
    finally:
        sys.stdout = old_stdout

    model = exp.model
    device = exp.device

    # 加载权重
    state_dict = torch.load(ckpt_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 获取测试集的 DataLoader
    test_data, test_loader = exp._get_data(flag='test')

    all_gate_pv = []
    all_irr = []

    print(">>> 开始在测试集上推理提取 Gate_PV...")
    with torch.no_grad():
        # 遍历测试集的前 N 个 batches (全部测完更准确)
        for i, batch_data in enumerate(test_loader):
            # 取出输入特征 (按照 Exp_PhysFormer 的拆分逻辑)
            batch_stat, batch_weather_hist, batch_weather_future, batch_y, batch_x_mark, batch_y_mark = batch_data
            batch_stat = batch_stat.float().to(device)
            batch_weather_hist = batch_weather_hist.float().to(device)
            batch_weather_future = batch_weather_future.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)

            # 调用模型前向传播并返回 gates_info (即包含 gate_pv 和 irr_seq)
            outputs, reg_loss, gates_info = model(
                x_stat=batch_stat,
                x_weather_hist=batch_weather_hist,
                x_weather_future=batch_weather_future,
                x_mark_enc=batch_x_mark,
                alpha=0.1,  # 评估时使用的混合 alpha 
                return_gates=True
            )
            # 在 gates_info 里，'pv_seq_batch' 是 [B, S]，'irr_seq_batch' 也是 [B, S]
            if 'pv_seq_batch' in gates_info and 'irr_seq_batch' in gates_info:
                all_gate_pv.append(gates_info['pv_seq_batch']) # 这是 numpy array
                all_irr.append(gates_info['irr_seq_batch'])   # numpy array
    
    if not all_gate_pv:
        print("Error: 提取的 gate_pv 或 irr_seq 为空，请检查 model.py / Causal_coupling.py 返回的 gates_info.")
        return

    # 拼接所有的 batch
    gate_pv_array = np.concatenate(all_gate_pv, axis=0)
    irr_array = np.concatenate(all_irr, axis=0)

    print(f"\n收集到测试集样本数: {gate_pv_array.shape[0]}，序列长度 {gate_pv_array.shape[1]}")
    
    # 将这两者完全压平，计算基于全部时间步观测点的全局皮尔逊相关系数
    gate_flat = gate_pv_array.flatten()
    irr_flat = irr_array.flatten()

    r_global, p_global = stats.pearsonr(gate_flat, irr_flat)

    print("\n============================================================")
    print("                      Pearson Correlation                     ")
    print("============================================================")
    print(f"[方法A] 全部数据点全局 Pearson r = {r_global:.4f} (p = {p_global:.2e})")
    
    # 基于每个独立样本求 r 再平均
    r_per_sample = []
    for i in range(gate_pv_array.shape[0]):
        # 防止存在全为常量的情况导致除0报警或NaN
        if np.std(gate_pv_array[i]) > 1e-6 and np.std(irr_array[i]) > 1e-6:
            r_i, _ = stats.pearsonr(gate_pv_array[i], irr_array[i])
            r_per_sample.append(r_i)

    if r_per_sample:
        r_mean = np.mean(r_per_sample)
        r_std = np.std(r_per_sample)
        print(f"[方法B] 逐样本独立计算的均值 r = {r_mean:.4f} ± {r_std:.4f}")

    print("\n>>> 论文中替换 Table III 中 r=0.809 的推荐值: ")
    print(f">>> {r_global:.4f}")
    print("============================================================\n")


if __name__ == '__main__':
    compute_from_model()
