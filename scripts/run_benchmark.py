import os
import argparse
import torch
import numpy as np
import pandas as pd
import random

from physformer.exp.exp_baseline import Exp_Baselines


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser(description='VPP Baseline Benchmark Runner')

    # --- 核心控制参数 ---
    parser.add_argument('--models_to_run', type=str, nargs='+',
                        default=['LSTM', 'GRU', 'PINN', 'Informer', 'Autoformer', 'DLinear', 'PatchTST'],
                        help='List of models to benchmark')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus')

    # --- 数据参数 (适配你的 VPP 数据) ---
    parser.add_argument('--root_path', type=str, default='./', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='data/vpp_dataset_3years.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='t', help='freq for time features encoding')

    # --- 维度参数 (必须与 exp_baseline.py 中的强制修正一致) ---
    parser.add_argument('--seq_len', type=int, default=672, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=96, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # 这里的默认值虽然会被 exp_baseline 覆盖，但保留着是个好习惯
    parser.add_argument('--enc_in', type=int, default=6, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=6, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=3, help='output size')

    # --- 训练参数 ---
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer weight decay')
    parser.add_argument('--use_amp', action='store_true', default=False, help='use automatic mixed precision training')

    # --- 模型特定参数 (Transformer类) ---
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='attn factor')
    parser.add_argument('--distil', action='store_false',default=False, help='whether to use distilling in encoder')
    parser.add_argument('--attn', type=str, default='full', help='attention used in encoder')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='custom', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--mix', default=True, action='store_false', help='use mix attention')
    parser.add_argument('--use_rope', default=True, action='store_true', help='use rotary position encoding')
    parser.add_argument('--rope_base', default=10000, type=int, help='rope base freq')

    # --- 路径与系统 ---
    parser.add_argument('--checkpoints', type=str, default='./exp_results/Baselines/checkpoints/',
                        help='location of model checkpoints')
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')

    return parser.parse_args()


def main():
    args = get_args()

    # 设备配置
    args.use_gpu = True if torch.cuda.is_available() and args.use_multi_gpu == False else False
    args.use_multi_gpu = False  # Baseline 建议单卡跑，比较稳
    args.device_ids = [args.gpu]
    device = torch.device('cuda:{}'.format(args.gpu)) if args.use_gpu else torch.device('cpu')

    print(f"=======================================================")
    print(f"  Starting VPP Benchmark on Device: {device}")
    print(f"  Models: {args.models_to_run}")
    print(f"  Seq Len: {args.seq_len} -> Pred Len: {args.pred_len}")
    print(f"=======================================================\n")

    # 用于存储最终结果的字典
    benchmark_results = {}

    for model_name in args.models_to_run:
        # 1. 基础设置
        print(f"\n>>>>>>> Running Baseline: {model_name} >>>>>>>")
        set_seed(2026)  # 重置种子，公平竞争

        args.model = model_name
        data_name = os.path.basename(args.data_path)
        if data_name.endswith('.csv'):
            data_name = data_name[:-4]
        # 为每个模型创建独立的文件夹，防止覆盖
        setting = '{}_{}_sl{}_pl{}_vpp'.format(
            model_name,
            data_name,  # 去掉.csv后缀
            args.seq_len,
            args.pred_len
        )
        args.checkpoint_name = setting

        # 2. 运行实验
        try:
            exp = Exp_Baselines(args)  # 实例化我们修改过的 Baseline 控制器

            # 训练
            # print(f"-> Training {model_name}...")
            # exp.train(setting)

            # 测试
            print(f"-> Testing {model_name}...")
            exp.test(setting, test=1)

            # 3. 读取并记录结果
            # 结果保存在 ./exp_results/{setting}/metrics.npy
            # 顺序: [mae, mse, rmse, bvr, rvr] (共5个)
            res_path = f'./exp_results/{setting}/metrics.npy'
            if os.path.exists(res_path):
                metrics = np.load(res_path)
                benchmark_results[model_name] = {
                    'MSE': metrics[1],
                    'MAE': metrics[0],
                    'RMSE': metrics[2],
                    'BVR (%)': metrics[3],  # 物理边界违规率
                    'RVR (%)': metrics[4]  # 爬坡违规率
                }

            # 4. 清理资源
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"!! Error running {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # --- 生成汇总报告 ---
    print("\n\n")
    print("=" * 60)
    print("      VPP FORECASTING BENCHMARK REPORT (SCI-Q2)")
    print("=" * 60)

    if len(benchmark_results) > 0:
        # 转为 Pandas DataFrame 展示
        df = pd.DataFrame(benchmark_results).T

        # 调整列顺序
        cols = ['MSE', 'MAE', 'RMSE', 'BVR (%)', 'RVR (%)']
        df = df[cols]

        print(df)

        # 保存为 CSV
        df.to_csv("benchmark_summary_report.csv")
        print(f"\nReport saved to 'benchmark_summary_report.csv'")

        print("-" * 60)
        print("Note: Compare these metrics with PhysFormer.")
        print("PhysFormer should have similar/lower MSE but MUCH lower BVR/RVR.")
        print("=" * 60)
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()