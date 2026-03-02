import os
import time
import torch
import numpy as np
import sys
import gc

# 确保能 import 到完整的工程模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from experiments.exp_baseline import Exp_Baselines
from experiments.exp_PhysFormer import Exp_PhysFormer
from scripts.run_benchmark import get_args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_inference_latency(model, name, exp, num_runs=100):
    model.eval()
    
    # 拿到真实的 Test 数据加载器，以获取1个真实 Batch
    test_data, test_loader = exp._get_data(flag='test')
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
    
    device = exp.device
    
    # 按照各自的 Exp 类的 _process_one_batch 预处理格式
    with torch.no_grad():
        if name == 'PhysFormer':
            # PhysFormer 数据预处理
            batch_stat = batch_x[:, :, :3].float().to(device)
            batch_weather_hist = batch_x[:, :, 3:].float().to(device)
            batch_weather_future = batch_y[:, :, 3:].float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            
            # 热身阶段
            for _ in range(10):
                model(
                    x_stat=batch_stat,
                    x_weather_hist=batch_weather_hist,
                    x_weather_future=batch_weather_future,
                    x_mark_enc=batch_x_mark,
                    alpha=1.0,
                    return_gates=False
                )
        else:
            # Baseline 数据预处理
            batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = \
                exp._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
                
            # 热身阶段
            for _ in range(10):
                if name in ['Informer', 'Autoformer']:
                    _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start_time = time.time()
        
        valid_runs = 0
        for _ in range(num_runs):
            if name == 'PhysFormer':
                model(
                    x_stat=batch_stat,
                    x_weather_hist=batch_weather_hist,
                    x_weather_future=batch_weather_future,
                    x_mark_enc=batch_x_mark,
                    alpha=1.0,
                    return_gates=False
                )
            else:
                if name in ['Informer', 'Autoformer']:
                    _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            valid_runs += 1
                
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.time()
        
    total_time_ms = (end_time - start_time) * 1000
    avg_latency_ms = total_time_ms / valid_runs / exp.args.batch_size
    return avg_latency_ms

# ==========================================
# 评测主流程
# ==========================================
models_to_run = ['PhysFormer', 'Informer', 'Autoformer', 'PatchTST', 'DLinear', 'LSTM', 'GRU', 'PINN']
results = {}

print("============================================================")
print("   Strict Model Benchmark via Built-in Experiment Classes   ")
print("============================================================")
print(f"{'Model':<15} | {'Params (M)':<12} | {'Latency (ms/sample)'}")
print("-" * 60)

for m_name in models_to_run:
    # 让每一个模型拥有干净独立的 args 副本
    sys.argv = ['run_exp_benchmark.py']
    args = get_args()
    args.model = m_name
    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0
    
    # 保证推断时的一致性
    args.batch_size = 32
    args.num_workers = 0  # Windows系统最好关掉多进程，防止卡进程
    
    base_dir = './exp_results'
    data_name = 'vpp_dataset_3years'
    
    # 拼装与训练完全一样的 checkpoint_name
    if m_name == 'PhysFormer':
        args.checkpoints = f'{base_dir}/PhysFormer/checkpoints'
        args.checkpoint_name = 'PhysFormer_full_seed2024'
        args.enc_in = 3
        args.d_model = 256
        args.n_heads = 8
        args.e_layers = 3
    else:
        setting = f'{m_name}_{data_name}_sl{args.seq_len}_pl{args.pred_len}_vpp'
        args.checkpoints = f'{base_dir}/Baselines/checkpoints'
        args.checkpoint_name = setting
        
    try:
        # 屏蔽多余的 stdout 日志创建输出
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        # 使用官方控制器初始化
        if m_name == 'PhysFormer':
            exp = Exp_PhysFormer(args)
        else:
            exp = Exp_Baselines(args)
            
        sys.stdout = old_stdout
            
        # 尝试加载真实权重
        model = exp.model 
        ckpt_path = os.path.join(args.checkpoints, args.checkpoint_name, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=exp.device)
            # 处理 DDP 前缀
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            status = ""
        else:
            status = "(Random weights for testing)"
            
        params_m = count_parameters(model) / 1e6
        
        # 开始在真实的测试数据上循环测算
        latency_ms = evaluate_inference_latency(model, m_name, exp, num_runs=50)
        
        print(f"{m_name:<15} | {params_m:<12.2f} | {latency_ms:.2f} {status}")
        
    except Exception as e:
        sys.stdout = old_stdout
        print(f"{m_name:<15} | Failed evaluation: {str(e)}")
        
    # 主动清理显存和内存避免 OOM
    gc.collect()
    torch.cuda.empty_cache()

print("============================================================\n")
