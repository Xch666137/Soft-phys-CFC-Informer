"""
PhysFormer unified entry point.

Usage:
    python run.py --config configs/physformer_default.yaml
    python run.py --config configs/physformer_default.yaml --lr 1e-4 --epochs 100
    python run.py --config configs/baselines/informer.yaml
    python run.py --config configs/physformer_default.yaml --test_only
    python run.py --config configs/physformer_default.yaml --print_config
"""

import argparse
import yaml
import torch
import sys
import os


def deep_update(base, override):
    """Recursively merge override dict into base dict."""
    for k, v in override.items():
        if k == '_base_':
            continue
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path):
    """Load YAML config with _base_ inheritance support."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    # Handle _base_ inheritance
    if '_base_' in cfg:
        base_path = os.path.join(os.path.dirname(config_path), cfg['_base_'])
        base_cfg = load_config(base_path)
        cfg = deep_update(base_cfg, cfg)

    return cfg


def flatten_config(cfg, prefix=''):
    """Flatten nested config dict to dot-separated keys."""
    items = {}
    for k, v in cfg.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            items.update(flatten_config(v, key))
        else:
            items[key] = v
    return items


def config_to_args(cfg):
    """Convert nested config dict to flat argparse-like namespace."""
    flat = {}

    # Model params
    m = cfg.get('model', {})
    flat['model'] = m.get('name', 'PhysFormer')
    for k in ['enc_in', 'dec_in', 'c_out', 'd_model', 'n_heads', 'e_layers',
              'd_layers', 'd_ff', 'factor', 'dropout', 'attn', 'embed',
              'activation', 'use_rope', 'rope_base', 'distil', 'output_attention']:
        if k in m:
            flat[k] = m[k]

    # Data params
    d = cfg.get('data', {})
    for k in ['root_path', 'data_path', 'features', 'target', 'seq_len',
              'pred_len', 'label_len', 'freq']:
        if k in d:
            flat[k] = d[k]

    # Training params
    t = cfg.get('training', {})
    flat['batch_size'] = t.get('batch_size', 128)
    flat['train_epochs'] = t.get('train_epochs', 100)
    flat['learning_rate'] = t.get('learning_rate', 3e-4)
    flat['weight_decay'] = t.get('weight_decay', 1e-5)
    flat['physics_prior_weight'] = t.get('physics_prior_weight', 0.1)
    flat['grad_clip'] = t.get('grad_clip', 1.0)
    flat['patience'] = t.get('patience', 10)
    flat['use_amp'] = t.get('use_amp', True)

    # Hardware params
    h = cfg.get('hardware', {})
    flat['use_gpu'] = h.get('use_gpu', True)
    flat['gpu'] = h.get('gpu', 0)
    flat['num_workers'] = h.get('num_workers', 8)

    # Checkpoint params
    c = cfg.get('checkpoint', {})
    flat['checkpoint_name'] = c.get('name', 'PhysFormer_full_seed2024')
    flat['checkpoints'] = c.get('path', 'checkpoints/PhysFormer/')

    # Ablation params
    a = cfg.get('ablation', {})
    flat['ablation_no_phys_stream'] = a.get('no_phys_stream', False)
    flat['ablation_no_pgcc'] = a.get('no_pgcc', False)
    flat['ablation_no_future_glu'] = a.get('no_future_glu', False)
    flat['ablation_no_curriculum'] = a.get('no_curriculum', False)
    flat['ablation_fixed_phys'] = a.get('fixed_phys', False)

    return argparse.Namespace(**flat)


def build_cli_parser():
    """Build argparse parser for CLI overrides."""
    parser = argparse.ArgumentParser(description='PhysFormer Unified Runner')

    # Config file
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')

    # Flow control
    parser.add_argument('--test_only', action='store_true',
                        help='Skip training, only run test')
    parser.add_argument('--print_config', action='store_true',
                        help='Print merged config and exit')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')

    # CLI overrides (override config values)
    parser.add_argument('--model', type=str, help='Model name override')
    parser.add_argument('--epochs', type=int, dest='train_epochs',
                        help='Training epochs override')
    parser.add_argument('--lr', type=float, dest='learning_rate',
                        help='Learning rate override')
    parser.add_argument('--batch_size', type=int, help='Batch size override')
    parser.add_argument('--gpu', type=int, help='GPU id override')
    parser.add_argument('--patience', type=int, help='Early stopping patience')
    parser.add_argument('--checkpoint_name', type=str, help='Checkpoint name')

    # Debug
    parser.add_argument('--debug_nan', action='store_true',
                        help='Enable anomaly detection')
    parser.add_argument('--save_gate_details', action='store_true',
                        help='Save detailed gate values')

    return parser


def main():
    parser = build_cli_parser()
    cli_args, unknown = parser.parse_known_args()

    # Load config from YAML
    cfg = load_config(cli_args.config)
    args = config_to_args(cfg)

    # Apply CLI overrides
    cli_dict = {k: v for k, v in vars(cli_args).items()
                if v is not None and k != 'config'}
    for k, v in cli_dict.items():
        setattr(args, k, v)

    # Set defaults for optional fields
    if not hasattr(args, 'is_training'):
        args.is_training = not cli_args.test_only
    if not hasattr(args, 'resume'):
        args.resume = cli_args.resume
    if not hasattr(args, 'debug_nan'):
        args.debug_nan = cli_args.debug_nan
    if not hasattr(args, 'save_gate_details'):
        args.save_gate_details = cli_args.save_gate_details
    if not hasattr(args, 'mix'):
        args.mix = True
    if not hasattr(args, 'output_attention'):
        args.output_attention = False

    # Ensure bool types
    args.use_amp = bool(args.use_amp)
    args.use_gpu = bool(args.use_gpu) and torch.cuda.is_available()

    # Print config mode
    if cli_args.print_config:
        print("=== Merged Configuration ===")
        for k, v in sorted(vars(args).items()):
            print(f"  {k}: {v}")
        return

    print(f'Args: {args}')

    # Route to appropriate experiment
    model_name = args.model

    if model_name == 'PhysFormer':
        from physformer.exp.exp_physformer import Exp_PhysFormer
        exp = Exp_PhysFormer(args)
        print(">>> Using PhysFormer Experiment (Physics-Guided Loss) <<<")

        if args.is_training:
            print('>>>>>>>start PhysFormer training : >>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train()

        print('>>>>>>>start PhysFormer test : >>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.test(setting=args.checkpoint_name)

    else:
        from physformer.exp.exp_baseline import Exp_Baselines

        # Baseline-specific defaults
        if not hasattr(args, 'label_len'):
            args.label_len = 96
        if not hasattr(args, 'dec_in'):
            args.dec_in = 6
        if not hasattr(args, 'c_out'):
            args.c_out = 3
        if not hasattr(args, 'use_multi_gpu'):
            args.use_multi_gpu = False
        if not hasattr(args, 'device_ids'):
            args.device_ids = [args.gpu]
        if not hasattr(args, 'd_layers'):
            args.d_layers = 2

        exp = Exp_Baselines(args)
        print(f">>> Using Baseline Experiment: {model_name} <<<")

        if args.is_training:
            data_name = os.path.basename(args.data_path)
            if data_name.endswith('.csv'):
                data_name = data_name[:-4]
            setting = f'{model_name}_{data_name}_sl{args.seq_len}_pl{args.pred_len}_vpp'
            args.checkpoint_name = setting

            print(f'>>>>>>>start {model_name} training : >>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

        print(f'>>>>>>>start {model_name} test : >>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.test(args.checkpoint_name)


if __name__ == "__main__":
    main()
