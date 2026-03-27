import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_TO_CONFIG = {
    'LSTM': 'configs/baselines/lstm.yaml',
    'GRU': 'configs/baselines/gru.yaml',
    'PINN': 'configs/baselines/pinn.yaml',
    'Informer': 'configs/baselines/informer.yaml',
    'Autoformer': 'configs/baselines/autoformer.yaml',
    'DLinear': 'configs/baselines/dlinear.yaml',
    'PatchTST': 'configs/baselines/patchtst.yaml',
    'iTransformer': 'configs/baselines/itransformer.yaml',
}


def default_setting(model_name, data_path, seq_len, pred_len):
    data_name = os.path.basename(data_path)
    if data_name.endswith('.csv'):
        data_name = data_name[:-4]
    return f'{model_name}_{data_name}_sl{seq_len}_pl{pred_len}_vpp'


def main():
    parser = argparse.ArgumentParser(description='Unified baseline benchmark runner')
    parser.add_argument('--models_to_run', type=str, nargs='+',
                        default=['LSTM', 'GRU', 'PINN', 'Informer', 'Autoformer', 'DLinear', 'PatchTST', 'iTransformer'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seq_len', type=int, default=672)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--data_path', type=str, default='data/vpp_dataset_3years.csv')
    parser.add_argument('--test_only', action='store_true', default=True)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    summary_dir = root_dir / 'exp_results' / 'Baselines' / 'benchmarks'
    summary_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for model_name in args.models_to_run:
        config_path = MODEL_TO_CONFIG.get(model_name)
        if config_path is None:
            raise ValueError(f'Unsupported model for benchmark script: {model_name}')

        checkpoint_name = default_setting(model_name, args.data_path, args.seq_len, args.pred_len)
        command = [
            sys.executable,
            str(root_dir / 'run.py'),
            '--config',
            config_path,
            '--checkpoint_name',
            checkpoint_name,
            '--override',
            f'hardware.gpu={args.gpu}',
            '--override',
            f'data.seq_len={args.seq_len}',
            '--override',
            f'data.pred_len={args.pred_len}',
            '--override',
            f'data.data_path={args.data_path}',
        ]
        if args.test_only:
            command.append('--test_only')

        print(f'Running benchmark model {model_name}: {" ".join(command)}')
        subprocess.run(command, check=True, cwd=root_dir)

        metrics_path = root_dir / 'exp_results' / 'Baselines' / 'checkpoints' / checkpoint_name / 'metrics.npy'
        if metrics_path.exists():
            metrics = np.load(metrics_path)
            results[model_name] = {
                'MSE': metrics[1],
                'MAE': metrics[0],
                'RMSE': metrics[2],
                'BVR (%)': metrics[3],
                'RVR (%)': metrics[4],
            }

    if results:
        df = pd.DataFrame(results).T[['MSE', 'MAE', 'RMSE', 'BVR (%)', 'RVR (%)']]
        print(df)
        df.to_csv(summary_dir / 'benchmark_summary_report.csv')
    else:
        print('No results collected.')


if __name__ == "__main__":
    main()
