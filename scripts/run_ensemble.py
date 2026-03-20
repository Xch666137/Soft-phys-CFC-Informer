import argparse
import torch
import numpy as np

from physformer.exp.exp_physformer import Exp_PhysFormer
from physformer.utils.metrics import metric

def train_single_model(seed, args):
    """
    使用指定随机种子训练单个模型
    """
    print(f"\n{'='*60}")
    print(f"Training model with seed {seed}")
    print(f"{'='*60}")

    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # 复制参数并修改实验名称
    model_args = argparse.Namespace(**vars(args))
    model_args.checkpoint_name = f"{args.checkpoint_name}_seed{seed}"

    # 创建实验实例
    exp = Exp_PhysFormer(model_args)

    # 训练
    # exp.train()

    # 测试并返回预测结果
    print(f'>>>>>>>start PhysFormer test for seed {seed}: >>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.test(setting=model_args.checkpoint_name)

    return {
        'seed': seed,
        'checkpoint_name': model_args.checkpoint_name,
    }

def main():
    parser = argparse.ArgumentParser(description='PhysFormer Ensemble Experiment')

    # 集成相关参数
    parser.add_argument('--seeds', type=int, nargs='+', default=[2024, 2025, 2026],
                        help='random seeds for ensemble members')

    # 添加所有参数
    add_common_args(parser)

    args = parser.parse_args()

    # 显式转换为 bool
    args.use_amp = bool(args.use_amp)
    args.use_gpu = bool(args.use_gpu)

    print(f'Args in PhysFormer ensemble experiment:\n{args}')

    # 训练所有模型
    models_info = []
    for seed in args.seeds:
        model_info = train_single_model(seed, args)
        models_info.append(model_info)

def add_common_args(parser):
    """
    从run_PhysFormer.py复制常见参数定义
    为了避免重复代码，直接从原文件导入参数
    """
    # 这里复制run_PhysFormer.py中的参数定义
    # 由于技术限制，我们直接复制关键参数
    # 命名与保存
    parser.add_argument('--model', type=str, default='PhysFormer', help='model name')
    parser.add_argument('--checkpoint_name', type=str, default='PhysFormer_ensemble', help='experiment name')
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
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
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
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')
    parser.add_argument('--physics_prior_weight', type=float, default=0.1, help='physics parameter prior regularization weight')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping max norm')
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')

    # 硬件参数
    parser.add_argument('--use_amp', type=int, default=1, help='use AMP (0/1)')
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu (0/1)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')

if __name__ == "__main__":
    main()