"""
Unified thesis experiment entrypoint.

Examples:
    python run.py train --config configs/baselines/tide_net_injection.yaml --print-config
    python run.py test --config configs/baselines/tide_net_injection.yaml --run-name tide_net_injection
    python run.py benchmark --config configs/drivers/benchmark_net_injection.yaml
    python run.py ablation --config configs/drivers/physformer_ablation.yaml
    python run.py build-dataset --nextgen-dir data_raw/nextgen --act-weather-csv data_raw/era5/act_canberra_hourly.csv --rye-generation-csv data_raw/rye/rye_generation_and_load.csv --rye-weather-csv data_raw/era5/rye_template_hourly.csv --output-dir data_processed/multi_portfolio
    python run.py pipeline --config configs/baselines/tide_net_injection.yaml --mapping-csv templates/network_mapping.csv --nextgen-dir data_raw/nextgen --act-weather-csv data_raw/era5/act_canberra_hourly.csv --rye-generation-csv data_raw/rye/rye_generation_and_load.csv --rye-weather-csv data_raw/era5/rye_template_hourly.csv --output-dir data_processed/multi_portfolio
"""

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import yaml

from physformer.data.data_factory import data_provider
from physformer.exp.exp_baseline import BaselineExperiment
from physformer.pipelines import (
    build_multi_portfolio_dataset,
    export_portfolio_forecasts,
    summarize_runs,
    validate_portfolio_forecasts,
)
from physformer.utils.tools import set_seed


def deep_update(base, override):
    for k, v in override.items():
        if k == '_base_':
            continue
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path):
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    if '_base_' in cfg:
        base_path = config_path.parent / cfg['_base_']
        base_cfg = load_config(base_path)
        cfg = deep_update(base_cfg, cfg)

    return cfg


def config_to_args(cfg):
    flat = {}

    model_cfg = cfg.get('model', {})
    flat['model'] = model_cfg.get('name', 'PhysFormer')
    for key in [
        'enc_in', 'dec_in', 'c_out', 'd_model', 'n_heads', 'e_layers', 'd_layers',
        'd_ff', 'factor', 'dropout', 'attn', 'embed', 'activation', 'use_rope',
        'rope_base', 'distil', 'output_attention'
    ]:
        if key in model_cfg:
            flat[key] = model_cfg[key]

    data_cfg = cfg.get('data', {})
    for key in [
        'root_path', 'data_path', 'features', 'target', 'seq_len', 'pred_len', 'label_len',
        'freq', 'time_col', 'id_col', 'region_col', 'split_col', 'split_strategy',
        'task_mode', 'target_cols', 'covariate_cols', 'known_future_covariate_cols',
        'history_state_cols', 'aux_target_cols'
    ]:
        if key in data_cfg:
            flat[key] = data_cfg[key]

    target_cols = data_cfg.get('target_cols')
    covariate_cols = data_cfg.get('covariate_cols')
    known_future_covariate_cols = data_cfg.get('known_future_covariate_cols')
    history_state_cols = data_cfg.get('history_state_cols')
    aux_target_cols = data_cfg.get('aux_target_cols')
    if target_cols is not None:
        flat['target_cols'] = list(target_cols)
    if covariate_cols is not None:
        flat['covariate_cols'] = list(covariate_cols)
    if known_future_covariate_cols is not None:
        flat['known_future_covariate_cols'] = list(known_future_covariate_cols)
    if history_state_cols is not None:
        flat['history_state_cols'] = list(history_state_cols)
    if aux_target_cols is not None:
        flat['aux_target_cols'] = list(aux_target_cols)

    training_cfg = cfg.get('training', {})
    flat['batch_size'] = training_cfg.get('batch_size', 128)
    flat['train_epochs'] = training_cfg.get('train_epochs', 100)
    flat['learning_rate'] = training_cfg.get('learning_rate', 3e-4)
    flat['weight_decay'] = training_cfg.get('weight_decay', 1e-5)
    flat['physics_prior_weight'] = training_cfg.get('physics_prior_weight', 0.1)
    flat['grad_clip'] = training_cfg.get('grad_clip', 1.0)
    flat['patience'] = training_cfg.get('patience', 10)
    flat['use_amp'] = training_cfg.get('use_amp', True)
    flat['seed'] = training_cfg.get('seed', 2024)
    flat['log_interval'] = training_cfg.get('log_interval', 50)
    flat['warmup_epochs'] = training_cfg.get('warmup_epochs', 0)
    flat['warmup_start_factor'] = training_cfg.get('warmup_start_factor', 0.2)
    flat['early_stop_metric'] = training_cfg.get('early_stop_metric', 'loss')
    flat['early_stop_start_epoch'] = training_cfg.get('early_stop_start_epoch', 1)

    hardware_cfg = cfg.get('hardware', {})
    flat['use_gpu'] = hardware_cfg.get('use_gpu', True)
    flat['gpu'] = hardware_cfg.get('gpu', 0)
    flat['num_workers'] = hardware_cfg.get('num_workers', 8)
    flat['use_multi_gpu'] = hardware_cfg.get('use_multi_gpu', False)
    flat['device_ids'] = hardware_cfg.get('device_ids', [flat['gpu']])
    flat['pin_memory'] = hardware_cfg.get('pin_memory', True)
    flat['persistent_workers'] = hardware_cfg.get('persistent_workers', True)
    flat['prefetch_factor'] = hardware_cfg.get('prefetch_factor', 4)

    checkpoint_cfg = cfg.get('checkpoint', {})
    flat['checkpoint_name'] = checkpoint_cfg.get('name')
    flat['checkpoints'] = checkpoint_cfg.get('path', 'runs/')

    ablation_cfg = cfg.get('ablation', {})
    flat['ablation_no_phys_stream'] = ablation_cfg.get('no_phys_stream', False)
    flat['ablation_no_battery_branch'] = ablation_cfg.get('no_battery_branch', False)
    flat['ablation_no_aux_supervision'] = ablation_cfg.get('no_aux_supervision', False)
    flat['ablation_no_soc_consistency'] = ablation_cfg.get('no_soc_consistency', False)
    flat['ablation_no_future_weather'] = ablation_cfg.get('no_future_weather', False)
    flat['ablation_shared_query_only'] = ablation_cfg.get('shared_query_only', False)

    feature_target_cols = list(target_cols or [])
    feature_known_future_cols = list(known_future_covariate_cols or covariate_cols or [])
    feature_history_state_cols = list(history_state_cols or [])

    if feature_target_cols:
        flat['c_out'] = len(target_cols)
        feature_count = len(feature_target_cols) + len(feature_known_future_cols) + len(feature_history_state_cols)
        flat['enc_in'] = feature_count
        flat['dec_in'] = feature_count

    if 'label_len' not in flat:
        flat['label_len'] = 0 if flat.get('model') == 'PhysFormer' else 96
    if 'features' not in flat:
        flat['features'] = 'M'

    return argparse.Namespace(**flat)


def apply_cli_overrides(args, cli_args):
    if cli_args.epochs is not None:
        args.train_epochs = cli_args.epochs
    if cli_args.lr is not None:
        args.learning_rate = cli_args.lr
    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size
    if cli_args.gpu is not None:
        args.gpu = cli_args.gpu
        args.device_ids = [cli_args.gpu]
    if cli_args.patience is not None:
        args.patience = cli_args.patience
    if getattr(cli_args, 'seed', None) is not None:
        args.seed = cli_args.seed
    if cli_args.run_name is not None:
        args.run_name = cli_args.run_name
    if cli_args.run_dir is not None:
        args.run_dir = cli_args.run_dir
    if getattr(cli_args, 'resume', False):
        args.resume = True
    if getattr(cli_args, 'debug_nan', False):
        args.debug_nan = True
    if getattr(cli_args, 'save_gate_details', False):
        args.save_gate_details = True
    return args


def infer_dataset_name(data_path: str):
    name = Path(data_path).stem
    return name or 'dataset'


def default_run_name(args):
    if getattr(args, 'checkpoint_name', None):
        return args.checkpoint_name
    task_mode = getattr(args, 'task_mode', 'task')
    dataset = infer_dataset_name(getattr(args, 'data_path', 'data'))
    return f"{args.model}_{task_mode}_{dataset}_sl{args.seq_len}_pl{args.pred_len}"


def finalize_args(args, cfg, cli_args):
    args = apply_cli_overrides(args, cli_args)
    args.use_amp = bool(getattr(args, 'use_amp', False))
    args.use_gpu = bool(getattr(args, 'use_gpu', False)) and torch.cuda.is_available()
    args.resume = bool(getattr(args, 'resume', False))
    args.debug_nan = bool(getattr(args, 'debug_nan', False))
    args.save_gate_details = bool(getattr(args, 'save_gate_details', False))
    args.mix = getattr(args, 'mix', True)
    args.output_attention = getattr(args, 'output_attention', False)
    args.seed = int(getattr(args, 'seed', 2024))

    run_name = getattr(args, 'run_name', None) or default_run_name(args)
    args.run_name = run_name
    run_root = Path('runs')
    run_dir = Path(getattr(args, 'run_dir', run_root / run_name))
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    args.run_dir = str(run_dir)
    args.checkpoint_name = run_name
    args.checkpoints = str(run_dir)
    return args, cfg


def create_experiment(args):
    if args.model == 'PhysFormer':
        from physformer.exp.exp_physformer import Exp_PhysFormer
        return Exp_PhysFormer(args)
    return BaselineExperiment(args)


def print_config(args):
    print("=== Effective Arguments ===")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")


def run_train(args, cfg):
    set_seed(getattr(args, 'seed', 2024))
    exp = create_experiment(args)
    exp.save_config(cfg)
    exp.train()
    return exp


def run_test(args, cfg):
    set_seed(getattr(args, 'seed', 2024))
    exp = create_experiment(args)
    exp.save_config(cfg)
    exp.test(load=True)
    return exp


def run_build_dataset(cli_args):
    outputs = build_multi_portfolio_dataset(
        nextgen_dir=cli_args.nextgen_dir,
        act_weather_csv=cli_args.act_weather_csv,
        rye_generation_csv=cli_args.rye_generation_csv,
        rye_weather_csv=cli_args.rye_weather_csv,
        output_dir=cli_args.output_dir,
        portfolio_size=cli_args.portfolio_size,
        target_penetration=cli_args.wind_penetration_target,
        audit_year=cli_args.audit_year,
        source_timezone=cli_args.source_timezone,
        region_id=cli_args.region_id,
        min_feature_availability=cli_args.min_feature_availability,
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    return outputs


def run_export_forecast(args, config_path):
    run_dir = Path(args.run_dir)
    output_csv = run_dir / 'exports' / 'portfolio_forecasts.csv'
    output_path = export_portfolio_forecasts(load_config, config_to_args, data_provider, config_path, str(run_dir), str(output_csv))
    print(f"Saved forecast export to: {output_path}")
    return output_path


def run_validate_powerflow(args, mapping_csv):
    run_dir = Path(args.run_dir)
    forecast_csv = run_dir / 'exports' / 'portfolio_forecasts.csv'
    summary = validate_portfolio_forecasts(str(forecast_csv), mapping_csv, str(run_dir / 'powerflow'))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def clone_args(args):
    return argparse.Namespace(**vars(copy.deepcopy(args)))


def apply_job_overrides(args, cfg, job):
    if 'batch_size' in job:
        args.batch_size = int(job['batch_size'])
        cfg.setdefault('training', {})['batch_size'] = int(job['batch_size'])
    if 'epochs' in job:
        args.train_epochs = int(job['epochs'])
        cfg.setdefault('training', {})['train_epochs'] = int(job['epochs'])
    if 'lr' in job:
        args.learning_rate = float(job['lr'])
        cfg.setdefault('training', {})['learning_rate'] = float(job['lr'])
    if 'patience' in job:
        args.patience = int(job['patience'])
        cfg.setdefault('training', {})['patience'] = int(job['patience'])
    if 'num_workers' in job:
        args.num_workers = int(job['num_workers'])
        cfg.setdefault('hardware', {})['num_workers'] = int(job['num_workers'])
    if 'gpu' in job:
        args.gpu = int(job['gpu'])
        args.device_ids = [int(job['gpu'])]
        cfg.setdefault('hardware', {})['gpu'] = int(job['gpu'])
    if 'pin_memory' in job:
        args.pin_memory = bool(job['pin_memory'])
        cfg.setdefault('hardware', {})['pin_memory'] = bool(job['pin_memory'])
    if 'persistent_workers' in job:
        args.persistent_workers = bool(job['persistent_workers'])
        cfg.setdefault('hardware', {})['persistent_workers'] = bool(job['persistent_workers'])
    if 'prefetch_factor' in job:
        args.prefetch_factor = int(job['prefetch_factor'])
        cfg.setdefault('hardware', {})['prefetch_factor'] = int(job['prefetch_factor'])
    if 'log_interval' in job:
        args.log_interval = int(job['log_interval'])
        cfg.setdefault('training', {})['log_interval'] = int(job['log_interval'])
    if 'warmup_epochs' in job:
        args.warmup_epochs = int(job['warmup_epochs'])
        cfg.setdefault('training', {})['warmup_epochs'] = int(job['warmup_epochs'])
    if 'warmup_start_factor' in job:
        args.warmup_start_factor = float(job['warmup_start_factor'])
        cfg.setdefault('training', {})['warmup_start_factor'] = float(job['warmup_start_factor'])
    if 'early_stop_metric' in job:
        args.early_stop_metric = str(job['early_stop_metric'])
        cfg.setdefault('training', {})['early_stop_metric'] = str(job['early_stop_metric'])
    if 'early_stop_start_epoch' in job:
        args.early_stop_start_epoch = int(job['early_stop_start_epoch'])
        cfg.setdefault('training', {})['early_stop_start_epoch'] = int(job['early_stop_start_epoch'])
    return args, cfg


def run_driver_jobs(driver_cfg, cli_args, job_kind):
    jobs = driver_cfg.get(job_kind, {}).get('jobs', [])
    if not jobs:
        raise ValueError(f"No jobs defined under '{job_kind}.jobs' in driver config.")

    run_dirs = []
    for job in jobs:
        seeds = job.get('seeds')
        if seeds is None:
            seeds = [getattr(cli_args, 'seed', None)] if getattr(cli_args, 'seed', None) is not None else [None]

        for seed in seeds:
            job_config_path = job['config']
            cfg = load_config(job_config_path)
            args = config_to_args(cfg)
            base_run_name = job.get('run_name') or job.get('name') or getattr(args, 'checkpoint_name', None)
            args.run_name = base_run_name
            args, cfg = finalize_args(args, cfg, cli_args)
            args, cfg = apply_job_overrides(args, cfg, job)
            if seed is not None:
                args.seed = int(seed)
                cfg.setdefault('training', {})['seed'] = int(seed)
                args.run_name = f"{base_run_name}__s{seed}"
                run_root = Path('runs')
                args.run_dir = str((Path.cwd() / (run_root / args.run_name)).resolve())
                args.checkpoint_name = args.run_name
                args.checkpoints = args.run_dir

            exp = run_train(args, cfg)
            exp.test(load=True)
            run_dirs.append(args.run_dir)

    driver_stem = Path(cli_args.config).stem
    summary_path = Path('runs') / 'reports' / f'{driver_stem}_summary_raw.csv'
    summary = summarize_runs(run_dirs, str(summary_path), job_kind)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def run_pipeline(args, cfg, cli_args):
    run_dir = Path(args.run_dir)
    outputs = build_multi_portfolio_dataset(
        nextgen_dir=cli_args.nextgen_dir,
        act_weather_csv=cli_args.act_weather_csv,
        rye_generation_csv=cli_args.rye_generation_csv,
        rye_weather_csv=cli_args.rye_weather_csv,
        output_dir=cli_args.output_dir,
        portfolio_size=cli_args.portfolio_size,
        target_penetration=cli_args.wind_penetration_target,
        audit_year=cli_args.audit_year,
        source_timezone=cli_args.source_timezone,
        region_id=cli_args.region_id,
        min_feature_availability=cli_args.min_feature_availability,
    )
    data_path = outputs['portfolio_dataset_for_training_csv']
    args.data_path = data_path
    cfg.setdefault('data', {})['data_path'] = data_path

    exp = run_train(args, cfg)
    exp.test(load=True)

    export_path = run_export_forecast(args, str(run_dir / 'config_merged.yaml'))
    if cli_args.mapping_csv:
        summary = validate_portfolio_forecasts(str(export_path), cli_args.mapping_csv, str(run_dir / 'powerflow'))
        print(json.dumps(summary, indent=2, ensure_ascii=False))


def add_common_run_args(parser):
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--run-name', help='Explicit run name')
    parser.add_argument('--run-dir', help='Explicit run directory')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint if supported')
    parser.add_argument('--gpu', type=int, help='GPU id override')
    parser.add_argument('--seed', type=int, help='Random seed override')
    parser.add_argument('--print-config', action='store_true', help='Print effective args and exit')
    parser.add_argument('--epochs', type=int, help='Training epochs override')
    parser.add_argument('--lr', type=float, help='Learning rate override')
    parser.add_argument('--batch-size', type=int, help='Batch size override')
    parser.add_argument('--patience', type=int, help='Early stopping patience override')
    parser.add_argument('--debug-nan', action='store_true', help='Enable anomaly detection')
    parser.add_argument('--save-gate-details', action='store_true', help='Save detailed gate values if model supports it')


def build_parser():
    parser = argparse.ArgumentParser(description='Unified thesis runner')
    subparsers = parser.add_subparsers(dest='command', required=True)

    build_dataset_parser = subparsers.add_parser('build-dataset', help='Build the strict multi-portfolio benchmark dataset')
    build_dataset_parser.add_argument('--region-id', default='act_canberra')
    build_dataset_parser.add_argument('--nextgen-dir', default='data_raw/nextgen')
    build_dataset_parser.add_argument('--act-weather-csv', default='data_raw/era5/act_canberra_hourly.csv')
    build_dataset_parser.add_argument('--rye-generation-csv', default='data_raw/rye/rye_generation_and_load.csv')
    build_dataset_parser.add_argument('--rye-weather-csv', default='data_raw/era5/rye_template_hourly.csv')
    build_dataset_parser.add_argument('--output-dir', default='data_processed/multi_portfolio')
    build_dataset_parser.add_argument('--portfolio-size', type=int, default=5)
    build_dataset_parser.add_argument('--wind-penetration-target', type=float, default=0.15)
    build_dataset_parser.add_argument('--audit-year', type=int, default=2018)
    build_dataset_parser.add_argument('--source-timezone', default='Australia/Sydney')
    build_dataset_parser.add_argument('--min-feature-availability', type=float, default=0.99)

    train_parser = subparsers.add_parser('train', help='Train one experiment')
    add_common_run_args(train_parser)

    test_parser = subparsers.add_parser('test', help='Test one experiment')
    add_common_run_args(test_parser)

    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark driver config')
    add_common_run_args(benchmark_parser)

    ablation_parser = subparsers.add_parser('ablation', help='Run ablation driver config')
    add_common_run_args(ablation_parser)

    export_parser = subparsers.add_parser('export-forecast', help='Export forecast CSV from one run')
    add_common_run_args(export_parser)

    validate_parser = subparsers.add_parser('validate-powerflow', help='Validate forecast via pandapower/SimBench')
    add_common_run_args(validate_parser)
    validate_parser.add_argument('--mapping-csv', required=True)

    pipeline_parser = subparsers.add_parser('pipeline', help='Run thesis pipeline')
    add_common_run_args(pipeline_parser)
    pipeline_parser.add_argument('--region-id', default='act_canberra')
    pipeline_parser.add_argument('--nextgen-dir', default='data_raw/nextgen')
    pipeline_parser.add_argument('--act-weather-csv', default='data_raw/era5/act_canberra_hourly.csv')
    pipeline_parser.add_argument('--rye-generation-csv', default='data_raw/rye/rye_generation_and_load.csv')
    pipeline_parser.add_argument('--rye-weather-csv', default='data_raw/era5/rye_template_hourly.csv')
    pipeline_parser.add_argument('--output-dir', default='data_processed/multi_portfolio')
    pipeline_parser.add_argument('--portfolio-size', type=int, default=5)
    pipeline_parser.add_argument('--wind-penetration-target', type=float, default=0.15)
    pipeline_parser.add_argument('--audit-year', type=int, default=2018)
    pipeline_parser.add_argument('--source-timezone', default='Australia/Sydney')
    pipeline_parser.add_argument('--min-feature-availability', type=float, default=0.99)
    pipeline_parser.add_argument('--mapping-csv', required=True)

    return parser


def main():
    parser = build_parser()
    cli_args = parser.parse_args()

    if cli_args.command == 'build-dataset':
        run_build_dataset(cli_args)
        return

    cfg = load_config(cli_args.config)

    if cli_args.command == 'benchmark':
        run_driver_jobs(cfg, cli_args, 'benchmark')
        return

    if cli_args.command == 'ablation':
        run_driver_jobs(cfg, cli_args, 'ablation')
        return

    args = config_to_args(cfg)
    args, cfg = finalize_args(args, cfg, cli_args)

    if cli_args.print_config:
        print_config(args)
        return

    if cli_args.command == 'train':
        run_train(args, cfg)
    elif cli_args.command == 'test':
        run_test(args, cfg)
    elif cli_args.command == 'export-forecast':
        run_export_forecast(args, cli_args.config)
    elif cli_args.command == 'validate-powerflow':
        run_validate_powerflow(args, cli_args.mapping_csv)
    elif cli_args.command == 'pipeline':
        run_pipeline(args, cfg, cli_args)
    else:
        raise ValueError(f"Unsupported command: {cli_args.command}")


if __name__ == '__main__':
    main()
