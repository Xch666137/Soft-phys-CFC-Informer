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

    # 存储与命名参数
    parser.add_argument('--model', type=str, default='PhysFormer', help='model name')
    parser.add_argument('--checkpoint_name', type=str, default='PhysFormer_experiment_v1.0', help='experiment name')
    parser.add_argument('--checkpoints', type=str, default='exp_results/PhysFormer/checkpoints/',
                        help='location of model checkpoints')
    parser.add_argument('--save_gate_details', action='store_true',
                        help='save detailed gate values for visualization')

    # 数据相关
    parser.add_argument('--root_path', type=str, default='./', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='data/vpp_dataset_3years.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default=None, help='target feature in S or MS task')

    # 序列长度
    parser.add_argument('--seq_len', type=int, default=672, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=96, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # 模型结构参数
    parser.add_argument('--enc_in', type=int, default=6, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=6, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=3, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_phys', type=int, default=64, help='dimension of physical states in CFC')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--attn', type=str, default='full', help='attention used in encoder')
    parser.add_argument('--embed', type=str, default='custom', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')


    # 优化组件参数
    parser.add_argument('--distil', default=False, action='store_true', help='whether to use distillation')
    parser.add_argument('--mix', default=True, action='store_false', help='use mix attention')
    parser.add_argument('--use_rope', default=True, action='store_true', help='use rotary position encoding')
    parser.add_argument('--rope_base', default=10000, type=int, help='rope base freq')
    parser.add_argument('--freq', default='t', type=str, help='freq for time features encoding')
    parser.add_argument('--stride', type=int, default=2, help='stride for CFC')

    # --- 物理正则化参数 ---
    parser.add_argument('--w_inertia', type=float, default=1e-4, help='CFC 参数惯性正则化权重')

    # 训练配置
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='optimizer weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping max norm')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')


    # 硬件
    parser.add_argument('--use_amp', type=bool, default=True, help='use AMP')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')

    args = parser.parse_args()

    print(f'Args in PhysFormer experiment:\n{args}')

    Exp = Exp_PhysFormer(args)
    print(">>> Using PhysFormer Experiment (Physics-Guided Loss) <<<")

    # print('>>>>>>>start PhysFormer training : >>>>>>>>>>>>>>>>>>>>>>>>>>')
    Exp.train()

    print('>>>>>>>start PhysFormer test : >>>>>>>>>>>>>>>>>>>>>>>>>>')
    Exp.test(setting=args.checkpoint_name)


if __name__ == "__main__":
    main()