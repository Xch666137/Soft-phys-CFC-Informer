import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scripts.run_benchmark import get_args
from experiments.exp_PhysFormer import Exp_PhysFormer

def setup_args(model_name):
    args = get_args()
    args.model = model_name
    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0
    args.batch_size = 1
    args.num_workers = 0
    
    args.root_path = './'
    args.data_path = 'data/vpp_dataset_3years.csv'
    args.features = 'M'
    args.target = None
    args.seq_len = 672
    args.pred_len = 96
    
    args.enc_in = 6
    if model_name == 'PhysFormer':
        args.d_model = 512
        args.n_heads = 8
        args.e_layers = 3
        args.checkpoints = r'e:\Py_program\Soft-phys-CFC-Informer\exp_results\PhysFormer\checkpoints'
        args.checkpoint_name = 'PhysFormer_full_seed2024'
    else: # Informer
        args.d_model = 512
        args.n_heads = 8
        args.e_layers = 3
        args.factor = 5
        args.checkpoints = r'e:\Py_program\Soft-phys-CFC-Informer\exp_results'
        args.checkpoint_name = 'Informer_vpp_dataset_3years_sl672_pl96_vpp'

    args.ablation_no_phys_stream = False
    args.ablation_no_pgcc = False
    args.ablation_no_future_glu = False
    args.ablation_no_curriculum = False
    args.ablation_fixed_phys = False
    
    return args

def get_predictions(model_name, args):
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        if model_name == 'PhysFormer':
            exp = Exp_PhysFormer(args)
        else:
            from experiments.exp_baseline import Exp_Baselines
            exp = Exp_Baselines(args)
    finally:
        sys.stdout = old_stdout

    model = exp.model
    device = exp.device

    ckpt_path = os.path.join(args.checkpoints, args.checkpoint_name, 'checkpoint.pth')
    if not os.path.exists(ckpt_path):
        print(f"[{model_name}] Error: Checkpoint not found -> {ckpt_path}")
        return None, None, None

    state_dict = torch.load(ckpt_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    test_data, test_loader = exp._get_data(flag='test')
    
    # 我们只取一天的数据进行精细可视化 (批次0)
    # 取一个白天光伏有强烈波动的好例子
    preds = []
    trues = []
    
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            if i > 100: # 寻找一个合适的索引
                batch_stat = batch_data[0].float().to(device)
                batch_weather_hist = batch_data[1].float().to(device)
                batch_weather_future = batch_data[2].float().to(device)
                batch_y = batch_data[3].float().to(device)
                batch_x_mark = batch_data[4].float().to(device)
                batch_y_mark = batch_data[5].float().to(device)
                
                # 寻找白天数据 (通过未来天气的光照判断)
                if batch_weather_future[0, 48, 1].item() > 0.5: # 预测窗口中间有较强光照
                    
                    if model_name == 'PhysFormer':
                        outputs, _ = model(batch_stat, batch_weather_hist, batch_weather_future, batch_x_mark, alpha=0.1)
                    else:
                        # 兼容 Informer 的 decoder 输入构建 (简单起见)
                        dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float().to(device)
                        dec_inp = torch.cat([batch_stat[:, -args.label_len:, :], dec_inp], dim=1).float().to(device)
                        # 这里简写，你需要根据实际 Inforemr forward 调整
                        outputs = model(batch_stat, batch_x_mark, dec_inp, batch_y_mark)
                        
                        # Apply Hard-clipping for Informer-Post
                        mean_val = exp.scaler.mean_[:3]
                        std_val = exp.scaler.scale_[:3]
                        zero_vals = -mean_val / (std_val + 1e-4) # 物理零点归一化值
                        zero_tensor = torch.tensor(zero_vals).float().to(device).view(1, 1, 3)
                        
                        # 仅对 PV(1) 和 Wind(2) 剪切，Load(0)不变
                        mask = torch.ones_like(outputs)
                        mask[:, :, 0] = 0.0 # 不截断Load
                        
                        outputs_clipped = torch.where((outputs < zero_tensor) & (mask > 0.5), zero_tensor, outputs)
                        outputs = outputs_clipped # 使用 Informer-Post
                    
                    pred = outputs.detach().cpu().numpy()
                    true = batch_y.detach().cpu().numpy()
                    
                    return pred[0], true[0], test_data.scaler
                    
    return None, None, None

def plot_rvm_comparison():
    print(">>> 正在生成 RVM 爬坡率违规对比图...")
    
    args_phys = setup_args('PhysFormer')
    pred_phys, true, scaler = get_predictions('PhysFormer', args_phys)
    
    if pred_phys is None:
        print("PhysFormer 推理失败。请检查检查点或由于计算资源太少未找到合适的白天片段")
        return
        
    print("PhysFormer 推理成功")
    # 为了简化脚本并且100%能够跑通，如果 Informer 代码调用复杂，
    # 我们可以基于真实值添加一定的高频噪音后进行硬裁剪来“模拟”经典的黑盒模型结果以画出概念图
    # 在真实论文中再用完整的模型去替换。
    # 我们这里尝试提取真实的 Informer 结果，如果在当前环境下无法快速载入 Informer 模型
    # (有些代码框架下同时实例化两个不同架构容易由于全局变量混淆)，我们直接画 PhysFormer 与真实值
    
    # 真实的反归一化
    def inverse_transform(data, scaler):
        # 取出 Load, PV, Wind
        # 假设 data.shape = [96, 3]
        mean = scaler.mean_[:3]
        std = scaler.scale_[:3]
        return data * std + mean
        
    y_phys = inverse_transform(pred_phys, scaler)
    y_true = inverse_transform(true, scaler)
    
    # 生成时间轴 (假设 15 min 一个点)
    time_axis = np.arange(96) * 15 / 60.0 # 转换为小时
    
    # Real Informer-Post (instead of simulated noise)
    # We already have pred_phys, let's load Informer too.
    args_inf = setup_args('Informer')
    pred_inf, _, _ = get_predictions('Informer', args_inf)
    
    if pred_inf is not None:
        y_inf = inverse_transform(pred_inf, scaler)
        y_inf_post = np.maximum(y_inf, 0)[:, 1] # Click PV channel
        inf_label = 'Informer-Post (Non-smooth clipping)'
    else:
        # 如果模型加载失败，诚实地对真实值进行硬截断作为概念基线，绝不能加噪声
        y_inf_post = np.maximum(y_true[:, 1], 0)
        inf_label = 'Heuristic Hard-Clipping (Zero-gradient Dead Zone)'

    plt.figure(figsize=(10, 6))
    
    plt.plot(time_axis, y_true[:, 1], 'k--', linewidth=2, label='Ground Truth (PV)')
    plt.plot(time_axis, y_inf_post, 'r-', alpha=0.9, linewidth=1.5, label=inf_label) # 修改了图例
    plt.plot(time_axis, y_phys[:, 1], 'b-', linewidth=2.5, label='PhysFormer (Continuous Differentiable Compliance)')
    
    plt.axhline(0, color='gray', linestyle=':', label='Physical Zero Boundary')
    
    plt.title("Non-differentiable Dead Zone vs. Structural Physical Manifold (PV Generation)", fontsize=14)
    plt.xlabel("Prediction Horizon (Hours)", fontsize=12)
    plt.ylabel("PV Generation (MW)", fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('RVM_concept_visualization.pdf', bbox_inches='tight')
    plt.close()
    print(">>> 成功保存真实 RVM 概念图: RVM_concept_visualization.pdf")

if __name__ == '__main__':
    plot_rvm_comparison()
