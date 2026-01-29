import sys
import os

# --- 新增代码开始 ---
# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (scripts 的上一级)
project_root = os.path.dirname(current_dir)
# 将项目根目录加入 python 搜索路径
sys.path.append(project_root)
# --- 新增代码结束 ---


import argparse
import torch
from experiments.exp_baseline import Exp_Informer


def main():
    parser = argparse.ArgumentParser(description='VPP Informer Experiment')

    # 基础配置
    parser.add_argument('--model', type=str, default='informer', help='model name')
    parser.add_argument('--root_path', type=str, default='./', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='data/vpp_dataset_3years.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default=None, help='target feature in S or MS task')

    parser.add_argument('--checkpoint_name', type=str, default='vpp_experiment', help='experiment name to label the checkpoint folder')
    parser.add_argument('--checkpoints', type=str, default='exp_results/informer/checkpoints/', help='location of model checkpoints')

    # 序列长度配置 (关键参数)
    # 15分钟数据：96点/天。
    # seq_len=672 (观察过去7天), label_len=48 (重叠半天), pred_len=96 (预测未来24小时)
    parser.add_argument('--seq_len', type=int, default=672, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # 模型结构参数
    parser.add_argument('--enc_in', type=int, default=10, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=6, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=3, help='output size')  # M模式下输出所有特征
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--dropout', type=float, default=0.15, help='dropout')
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder')
    parser.add_argument('--embed', type=str, default='custom', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # 优化的新组件参数
    parser.add_argument('--distil', default=True, action='store_false', help='whether to use distillation')
    parser.add_argument('--mix', default=True, action='store_false', help='use mix attention')
    parser.add_argument('--use_rope', default=True, action='store_true', help='use rotary position encoding')
    parser.add_argument('--rope_base', default=10000, type=int, help='rope base freq')
    parser.add_argument('--freq', default='t', type=str, help='freq for time features encoding')

    # 训练配置
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer weight decay')
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')

    # 硬件
    parser.add_argument('--use_amp', type=bool, default=True, help='use AMP')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--num_workers', type=int, default=16, help='data loader num workers')

    args = parser.parse_args()

    # 运行实验
    print(f'Args in experiment:\n{args}')
    exp = Exp_Informer(args)

    print('>>>>>>>start training : >>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train()


if __name__ == "__main__":
    main()