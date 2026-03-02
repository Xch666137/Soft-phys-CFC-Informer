import sys
import os

# 获取项目根目录，确保能 import 到 modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import argparse
import torch
from experiments.exp_PhysFormer import Exp_PhysFormer


def main():
    parser = argparse.ArgumentParser(description='PhysFormer Experiment')

    # 命名与保存
    parser.add_argument('--model', type=str, default='PhysFormer', help='model name')
    parser.add_argument('--checkpoint_name', type=str, default='PhysFormer_ensemble_seed2024', help='experiment name')
    parser.add_argument('--checkpoints', type=str, default='exp_results/PhysFormer/checkpoints/',
                        help='location of model checkpoints')
    parser.add_argument('--save_gate_details', action='store_true', help='save detailed gate values for visualization')

    # 数据相关
    parser.add_argument('--root_path', type=str, default='./', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='data/vpp_dataset_3years.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default=None, help='target feature in S or MS task')

    # 序列长度
    parser.add_argument('--seq_len', type=int, default=672, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # 模型结构参数
    parser.add_argument('--enc_in', type=int, default=6, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--dropout', type=float, default=0.10, help='dropout')
    parser.add_argument('--attn', type=str, default='full', help='attention used in encoder')
    parser.add_argument('--embed', type=str, default='custom', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')

    # 优化组件
    parser.add_argument('--distil', action='store_true', default=False, help='whether to use distillation')
    parser.add_argument('--mix', action='store_false', default=True, help='use mix attention')
    parser.add_argument('--use_rope', action='store_true', default=True, help='use rotary position encoding')
    parser.add_argument('--rope_base', default=10000, type=int, help='rope base freq')
    parser.add_argument('--freq', default='h', type=str, help='freq for time features encoding')
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')
    parser.add_argument('--physics_prior_weight', type=float, default=0.1, help='physics parameter prior regularization weight')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping max norm')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')

    # --- ABLATION: Global Flags ---
    parser.add_argument('--ablation_no_phys_stream', action='store_true', default=False,
                        help='Ablation: Remove physics stream entirely (V1)')
    parser.add_argument('--ablation_no_pgcc', action='store_true', default=False,
                        help='Ablation: Replace PGCC with naive concatenation fusion (V2)')
    parser.add_argument('--ablation_no_future_glu', action='store_true', default=False,
                        help='Ablation: Remove future weather GLU injection (V3)')
    parser.add_argument('--ablation_no_curriculum', action='store_true', default=False,
                        help='Ablation: Disable curriculum learning, soft gates from scratch (V4)')
    parser.add_argument('--ablation_fixed_phys', action='store_true', default=False,
                        help='Ablation: Freeze physical parameters, no data-driven fine-tuning (V5)')

    # 硬件参数
    parser.add_argument('--use_amp', type=int, default=1, help='use AMP (0/1)')
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu (0/1)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')

    parser.add_argument('--is_training', type=int, default=1, help='status')

    args = parser.parse_args()

    # 显式转换为 bool
    args.use_amp = bool(args.use_amp)
    args.use_gpu = bool(args.use_gpu)

    print(f'Args in PhysFormer experiment:\n{args}')

    Exp = Exp_PhysFormer(args)
    print(">>> Using PhysFormer Experiment (Physics-Guided Loss) <<<")

    if args.is_training:
        print('>>>>>>>start PhysFormer training : >>>>>>>>>>>>>>>>>>>>>>>>>>')
        Exp.train()

    print('>>>>>>>start PhysFormer test : >>>>>>>>>>>>>>>>>>>>>>>>>>')
    Exp.test(setting=args.checkpoint_name)


if __name__ == "__main__":
    main()