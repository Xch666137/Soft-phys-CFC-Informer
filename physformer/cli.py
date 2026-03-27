"""
Config-driven CLI entrypoint for PhysFormer experiments.
"""

import argparse
import os
from copy import deepcopy

import yaml


COMPAT_OVERRIDE_MAP = {
    'model': 'model.name',
    'train_epochs': 'training.train_epochs',
    'learning_rate': 'training.learning_rate',
    'batch_size': 'training.batch_size',
    'gpu': 'hardware.gpu',
    'patience': 'training.patience',
    'checkpoint_name': 'checkpoint.name',
}

ABLATION_MAP = {
    'no_phys_stream': 'ablation_no_phys_stream',
    'no_pgcc': 'ablation_no_pgcc',
    'no_future_glu': 'ablation_no_future_glu',
    'no_curriculum': 'ablation_no_curriculum',
    'fixed_phys': 'ablation_fixed_phys',
    'no_gate_reg': 'ablation_no_gate_reg',
}


def deep_update(base, override):
    for key, value in override.items():
        if key == '_base_':
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    if '_base_' in cfg:
        base_path = os.path.join(os.path.dirname(config_path), cfg['_base_'])
        base_cfg = load_config(base_path)
        cfg = deep_update(base_cfg, cfg)

    return cfg


def parse_override_value(raw_value):
    try:
        return yaml.safe_load(raw_value)
    except yaml.YAMLError:
        return raw_value


def set_nested(config, dotted_key, value):
    parts = dotted_key.split('.')
    cursor = config
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def apply_cli_overrides(cfg, cli_args):
    merged = deepcopy(cfg)

    for raw_item in cli_args.override or []:
        if '=' not in raw_item:
            raise ValueError(f"Invalid override '{raw_item}'. Expected format section.key=value")
        dotted_key, raw_value = raw_item.split('=', 1)
        set_nested(merged, dotted_key.strip(), parse_override_value(raw_value))

    for cli_key, dotted_key in COMPAT_OVERRIDE_MAP.items():
        value = getattr(cli_args, cli_key, None)
        if value is not None:
            set_nested(merged, dotted_key, value)

    if cli_args.resume:
        set_nested(merged, 'runtime.resume', True)
    if cli_args.debug_nan:
        set_nested(merged, 'runtime.debug_nan', True)
    if cli_args.save_gate_details:
        set_nested(merged, 'output.save_gate_details', True)
    if cli_args.test_only:
        set_nested(merged, 'runtime.is_training', False)

    return merged


def ensure_section(cfg, section_name):
    section = cfg.get(section_name)
    if not isinstance(section, dict):
        section = {}
        cfg[section_name] = section
    return section


def normalize_config(cfg):
    cfg = deepcopy(cfg)

    model = ensure_section(cfg, 'model')
    data = ensure_section(cfg, 'data')
    training = ensure_section(cfg, 'training')
    hardware = ensure_section(cfg, 'hardware')
    checkpoint = ensure_section(cfg, 'checkpoint')
    runtime = ensure_section(cfg, 'runtime')
    output = ensure_section(cfg, 'output')
    evaluation = ensure_section(cfg, 'evaluation')
    ablation = ensure_section(cfg, 'ablation')

    model_name = model.get('name', 'PhysFormer')
    trainer_family = 'physformer' if model_name == 'PhysFormer' else 'baseline'
    output_root = output.get('root', 'exp_results')
    family_dir = 'PhysFormer' if trainer_family == 'physformer' else 'Baselines'

    checkpoint.setdefault('path', os.path.join(output_root, family_dir, 'checkpoints'))
    checkpoint.setdefault('name', None if trainer_family == 'baseline' else 'PhysFormer_full_seed2024')

    legacy_checkpoint_paths = {
        'checkpoints/PhysFormer/',
        'checkpoints/PhysFormer',
        'checkpoints/Baselines/',
        'checkpoints/Baselines',
    }
    if checkpoint.get('path') in legacy_checkpoint_paths:
        checkpoint['path'] = os.path.join(output_root, family_dir, 'checkpoints')

    data.setdefault('root_path', './')
    data.setdefault('data_path', 'data/vpp_dataset_3years.csv')
    data.setdefault('features', 'M')
    data.setdefault('target', None)
    data.setdefault('seq_len', 672)
    data.setdefault('pred_len', 96)
    data.setdefault('label_len', 0 if trainer_family == 'physformer' else 96)
    data.setdefault('freq', 't')
    data.setdefault('train_noise_level', 0.01 if trainer_family == 'physformer' else 0.03)
    data.setdefault('val_batch_size', training.get('batch_size', 128))
    data.setdefault('test_batch_size', 1 if trainer_family == 'baseline' else training.get('batch_size', 128))

    training.setdefault('batch_size', 128 if trainer_family == 'physformer' else 32)
    training.setdefault('train_epochs', 100 if trainer_family == 'physformer' else 50)
    training.setdefault('learning_rate', 3e-4 if trainer_family == 'physformer' else 1e-4)
    training.setdefault('weight_decay', 1e-5 if trainer_family == 'physformer' else 1e-4)
    training.setdefault('patience', 10 if trainer_family == 'physformer' else 5)
    training.setdefault('use_amp', trainer_family == 'physformer')
    training.setdefault('grad_clip', 1.0)
    training.setdefault('physics_prior_weight', 0.1)
    training.setdefault('plot_val_every', 1)

    hardware.setdefault('use_gpu', True)
    hardware.setdefault('gpu', 0)
    hardware.setdefault('num_workers', 8)
    hardware.setdefault('pin_memory', hardware.get('use_gpu', True))
    hardware.setdefault('persistent_workers', hardware.get('num_workers', 8) > 0)
    hardware.setdefault('prefetch_factor', 2 if hardware.get('num_workers', 8) > 0 else None)
    hardware.setdefault('use_multi_gpu', False)
    hardware.setdefault('device_ids', [hardware.get('gpu', 0)])

    runtime.setdefault('seed', 2024)
    runtime.setdefault('deterministic', False)
    runtime.setdefault('benchmark', not runtime['deterministic'])
    runtime.setdefault('resume', False)
    runtime.setdefault('debug_nan', False)
    runtime.setdefault('is_training', True)

    output.setdefault('root', output_root)
    output.setdefault('save_gate_details', False)

    evaluation.setdefault('extreme_scenario_test', False)

    for ablation_key in ABLATION_MAP:
        ablation.setdefault(ablation_key, False)

    return cfg


def default_baseline_setting(args):
    data_name = os.path.basename(args.data_path)
    if data_name.endswith('.csv'):
        data_name = data_name[:-4]
    return f'{args.model}_{data_name}_sl{args.seq_len}_pl{args.pred_len}_vpp'


def config_to_args(cfg):
    normalized = normalize_config(cfg)
    flat = {}

    model_cfg = normalized['model']
    for key, value in model_cfg.items():
        if key == 'name':
            flat['model'] = value
        else:
            flat[key] = value

    for section_name in ['data', 'training', 'hardware', 'runtime', 'evaluation']:
        flat.update(normalized.get(section_name, {}))

    checkpoint_cfg = normalized['checkpoint']
    flat['checkpoint_name'] = checkpoint_cfg.get('name')
    flat['checkpoints'] = checkpoint_cfg.get('path')

    output_cfg = normalized['output']
    flat['output_root'] = output_cfg.get('root')
    for key, value in output_cfg.items():
        if key != 'root':
            flat[key] = value

    flat['trainer_family'] = 'physformer' if flat['model'] == 'PhysFormer' else 'baseline'

    ablation_cfg = normalized.get('ablation', {})
    for source_key, target_key in ABLATION_MAP.items():
        flat[target_key] = ablation_cfg.get(source_key, False)

    if flat['trainer_family'] == 'physformer':
        flat['label_len'] = 0
    elif flat.get('checkpoint_name') is None:
        probe_args = argparse.Namespace(**flat)
        flat['checkpoint_name'] = default_baseline_setting(probe_args)

    if 'mix' not in flat:
        flat['mix'] = True
    if 'output_attention' not in flat:
        flat['output_attention'] = False

    flat['resolved_config'] = normalized
    return argparse.Namespace(**flat)


def build_cli_parser():
    parser = argparse.ArgumentParser(description='PhysFormer Unified Runner')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--override', action='append', default=[],
                        help='Dotted config override, e.g. training.train_epochs=1')
    parser.add_argument('--test_only', action='store_true',
                        help='Skip training, only run test')
    parser.add_argument('--print_config', action='store_true',
                        help='Print merged config and exit')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--model', type=str, help='Model name override')
    parser.add_argument('--epochs', type=int, dest='train_epochs',
                        help='Training epochs override')
    parser.add_argument('--lr', type=float, dest='learning_rate',
                        help='Learning rate override')
    parser.add_argument('--batch_size', type=int, help='Batch size override')
    parser.add_argument('--gpu', type=int, help='GPU id override')
    parser.add_argument('--patience', type=int, help='Early stopping patience')
    parser.add_argument('--checkpoint_name', type=str, help='Checkpoint name override')
    parser.add_argument('--debug_nan', action='store_true',
                        help='Enable anomaly detection')
    parser.add_argument('--save_gate_details', action='store_true',
                        help='Save detailed gate values')
    return parser


def _import_torch():
    import torch
    return torch


def configure_runtime(args):
    torch = _import_torch()
    from physformer.utils.tools import set_seed

    seed = getattr(args, 'seed', None)
    if seed is not None:
        set_seed(seed)

    deterministic = bool(getattr(args, 'deterministic', False))
    benchmark = bool(getattr(args, 'benchmark', not deterministic))

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark and not deterministic

    args.use_amp = bool(args.use_amp)
    args.use_gpu = bool(args.use_gpu) and torch.cuda.is_available()

    if not args.use_gpu:
        args.device_ids = []
    elif getattr(args, 'use_multi_gpu', False):
        args.device_ids = list(getattr(args, 'device_ids', [args.gpu]))
    else:
        args.device_ids = [args.gpu]


def print_merged_config(cfg):
    print("=== Merged Configuration ===")
    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


def main():
    parser = build_cli_parser()
    cli_args = parser.parse_args()

    cfg = load_config(cli_args.config)
    cfg = apply_cli_overrides(cfg, cli_args)
    args = config_to_args(cfg)

    if cli_args.print_config:
        print_merged_config(args.resolved_config)
        return

    configure_runtime(args)
    print(f'Args: {args}')

    if args.model == 'PhysFormer':
        from physformer.exp.exp_physformer import Exp_PhysFormer

        exp = Exp_PhysFormer(args)
        exp_label = 'PhysFormer Experiment (Physics-Guided Loss)'
    else:
        from physformer.exp.exp_baseline import Exp_Baselines

        if not getattr(args, 'checkpoint_name', None):
            args.checkpoint_name = default_baseline_setting(args)

        exp = Exp_Baselines(args)
        exp_label = f'Baseline Experiment: {args.model}'
    try:
        print(f">>> Using {exp_label} <<<")

        if args.is_training:
            print(f'>>>>>>>start {args.model} training : >>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting=args.checkpoint_name)

        print(f'>>>>>>>start {args.model} test : >>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.test(
            setting=args.checkpoint_name,
            load=not args.is_training or args.resume,
            extreme_scenario_test=getattr(args, 'extreme_scenario_test', False),
        )
    finally:
        exp.close()
