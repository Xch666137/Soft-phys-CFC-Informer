import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_portfolio_dataset(input_csv: str, output_csv: str, portfolio_id: str, region_id: str):
    df = pd.read_csv(input_csv)
    required = ['date', 'temperature', 'irradiance', 'wind_speed']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if 'p_vpp_mw' in df.columns:
        p_vpp = df['p_vpp_mw']
    elif all(c in df.columns for c in ['load_mw', 'pv_mw', 'wind_mw']):
        p_vpp = df['load_mw'] - df['pv_mw'] - df['wind_mw']
    else:
        raise ValueError(
            "Input must contain either 'p_vpp_mw' or all of 'load_mw', 'pv_mw', 'wind_mw'."
        )

    out = pd.DataFrame({
        'date': pd.to_datetime(df['date']),
        'portfolio_id': portfolio_id,
        'region_id': region_id,
        'p_vpp_mw': p_vpp,
        'temperature': df['temperature'],
        'irradiance': df['irradiance'],
        'wind_speed': df['wind_speed'],
    })

    optional_cols = {
        'load_mw': 'p_load_mw',
        'pv_mw': 'p_pv_mw',
        'wind_mw': 'p_wind_mw',
    }
    for src, dst in optional_cols.items():
        if src in df.columns:
            out[dst] = df[src]

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values('date').reset_index(drop=True).to_csv(output_path, index=False)
    return output_path


def export_portfolio_forecasts(config_loader, config_to_args, data_provider, config_path: str, experiment_dir: str, output_csv: str):
    cfg = config_loader(config_path)
    args = config_to_args(cfg)
    args.model = getattr(args, 'model', cfg.get('model', {}).get('name', 'Informer'))
    dataset, _ = data_provider(args, flag='test')

    if getattr(args, 'task_mode', 'component_multitask') != 'net_injection':
        raise ValueError("This exporter currently supports task_mode='net_injection' only.")
    if len(getattr(args, 'target_cols', [])) != 1:
        raise ValueError("This exporter expects exactly one target column for net injection.")

    run_dir = Path(experiment_dir)
    pred = np.load(run_dir / 'pred.npy')
    true = np.load(run_dir / 'true.npy')
    target_name = args.target_cols[0]

    if pred.shape[0] != len(dataset):
        raise ValueError(f"Prediction sample count mismatch: pred={pred.shape[0]}, dataset={len(dataset)}")

    rows = []
    for sample_idx in range(len(dataset)):
        meta = dataset.get_prediction_metadata(sample_idx)
        timestamps = pd.to_datetime(meta['forecast_timestamps'])
        for horizon_idx, ts in enumerate(timestamps):
            rows.append({
                'date': ts,
                'portfolio_id': meta['portfolio_id'],
                'region_id': meta['region_id'],
                'sample_index': sample_idx,
                'horizon_index': horizon_idx,
                f'pred_{target_name}': float(pred[sample_idx, horizon_idx, 0]),
                f'true_{target_name}': float(true[sample_idx, horizon_idx, 0]),
            })

    out = pd.DataFrame(rows).sort_values(
        ['date', 'portfolio_id', 'sample_index', 'horizon_index']
    ).reset_index(drop=True)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return output_path


def validate_portfolio_forecasts(forecast_csv: str, mapping_csv: str, output_dir: str):
    try:
        import pandapower as pp
        import simbench as sb
    except Exception as exc:
        raise RuntimeError(
            "Network validation requires pandapower and simbench. Install them in the active environment first."
        ) from exc

    forecast_df = pd.read_csv(forecast_csv)
    mapping_df = pd.read_csv(mapping_csv)
    required_mapping = ['portfolio_id', 'simbench_case', 'target_bus_or_asset_group', 'allocation_weight']
    missing_mapping = [c for c in required_mapping if c not in mapping_df.columns]
    if missing_mapping:
        raise ValueError(f"Missing mapping columns: {missing_mapping}")

    pred_col = next((c for c in forecast_df.columns if c.startswith('pred_')), None)
    true_col = next((c for c in forecast_df.columns if c.startswith('true_')), None)
    if pred_col is None or true_col is None:
        raise ValueError("Forecast CSV must contain one pred_* column and one true_* column.")

    simbench_cases = mapping_df['simbench_case'].unique().tolist()
    if len(simbench_cases) != 1:
        raise ValueError("This validator currently supports a single simbench_case per run.")

    def create_virtual_elements(net):
        element_map = {}
        for row in mapping_df.itertuples(index=False):
            bus = int(row.target_bus_or_asset_group)
            key = (row.portfolio_id, bus)
            load_idx = pp.create_load(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"VPP_LOAD_{row.portfolio_id}_{bus}")
            sgen_idx = pp.create_sgen(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"VPP_SGEN_{row.portfolio_id}_{bus}")
            element_map[key] = {'load_idx': load_idx, 'sgen_idx': sgen_idx, 'weight': float(row.allocation_weight)}
        return element_map

    def apply_portfolio_values(net, element_map, ts_frame):
        net.load.loc[[v['load_idx'] for v in element_map.values()], 'p_mw'] = 0.0
        net.sgen.loc[[v['sgen_idx'] for v in element_map.values()], 'p_mw'] = 0.0
        for row in ts_frame.itertuples(index=False):
            for (pid, _bus), meta in element_map.items():
                if pid != row.portfolio_id:
                    continue
                allocated = float(row.p_value) * meta['weight']
                if allocated >= 0:
                    net.load.at[meta['load_idx'], 'p_mw'] = allocated
                    net.sgen.at[meta['sgen_idx'], 'p_mw'] = 0.0
                else:
                    net.load.at[meta['load_idx'], 'p_mw'] = 0.0
                    net.sgen.at[meta['sgen_idx'], 'p_mw'] = -allocated

    def run_single_pf(net):
        try:
            pp.runpp(net)
            vm = net.res_bus.vm_pu.to_numpy(copy=True)
            line_loading = net.res_line.loading_percent.to_numpy(copy=True) if len(net.line) else np.array([])
            trafo_loading = net.res_trafo.loading_percent.to_numpy(copy=True) if len(net.trafo) else np.array([])
            return {
                'ok': True,
                'vm': vm,
                'line_loading': line_loading,
                'trafo_loading': trafo_loading,
                'voltage_violations': int(((vm < 0.95) | (vm > 1.05)).sum()),
                'line_overloads': int((line_loading > 100.0).sum()) if line_loading.size else 0,
                'trafo_overloads': int((trafo_loading > 100.0).sum()) if trafo_loading.size else 0,
                'line_loss_mw': float(net.res_line.pl_mw.sum()) if len(net.line) else 0.0,
            }
        except Exception:
            return {
                'ok': False,
                'vm': np.array([]),
                'line_loading': np.array([]),
                'trafo_loading': np.array([]),
                'voltage_violations': np.nan,
                'line_overloads': np.nan,
                'trafo_overloads': np.nan,
                'line_loss_mw': np.nan,
            }

    simbench_case = simbench_cases[0]
    net_pred = sb.get_simbench_net(simbench_case)
    net_true = sb.get_simbench_net(simbench_case)
    element_map_pred = create_virtual_elements(net_pred)
    element_map_true = create_virtual_elements(net_true)

    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
    forecast_df = forecast_df.sort_values(['date', 'portfolio_id'])

    rows = []
    for ts, ts_df in forecast_df.groupby('date', sort=True):
        pred_frame = ts_df[['portfolio_id', pred_col]].rename(columns={pred_col: 'p_value'})
        true_frame = ts_df[['portfolio_id', true_col]].rename(columns={true_col: 'p_value'})
        apply_portfolio_values(net_pred, element_map_pred, pred_frame)
        apply_portfolio_values(net_true, element_map_true, true_frame)

        pred_result = run_single_pf(net_pred)
        true_result = run_single_pf(net_true)
        if pred_result['ok'] and true_result['ok']:
            vm_mae = float(np.mean(np.abs(pred_result['vm'] - true_result['vm'])))
            line_loading_mae = float(np.mean(np.abs(pred_result['line_loading'] - true_result['line_loading']))) if pred_result['line_loading'].size and true_result['line_loading'].size else 0.0
            trafo_loading_mae = float(np.mean(np.abs(pred_result['trafo_loading'] - true_result['trafo_loading']))) if pred_result['trafo_loading'].size and true_result['trafo_loading'].size else 0.0
        else:
            vm_mae = np.nan
            line_loading_mae = np.nan
            trafo_loading_mae = np.nan

        rows.append({
            'date': ts,
            'pred_feasible': int(pred_result['ok']),
            'true_feasible': int(true_result['ok']),
            'vm_mae': vm_mae,
            'line_loading_mae': line_loading_mae,
            'trafo_loading_mae': trafo_loading_mae,
            'pred_voltage_violations': pred_result['voltage_violations'],
            'true_voltage_violations': true_result['voltage_violations'],
            'pred_line_overloads': pred_result['line_overloads'],
            'true_line_overloads': true_result['line_overloads'],
            'pred_trafo_overloads': pred_result['trafo_overloads'],
            'true_trafo_overloads': true_result['trafo_overloads'],
            'pred_line_loss_mw': pred_result['line_loss_mw'],
            'true_line_loss_mw': true_result['line_loss_mw'],
        })

    ts_out = pd.DataFrame(rows)
    summary = {
        'simbench_case': simbench_case,
        'timesteps': int(len(ts_out)),
        'pred_feasible_rate': float(ts_out['pred_feasible'].mean()),
        'true_feasible_rate': float(ts_out['true_feasible'].mean()),
        'vm_mae_mean': float(ts_out['vm_mae'].mean()),
        'line_loading_mae_mean': float(ts_out['line_loading_mae'].mean()),
        'trafo_loading_mae_mean': float(ts_out['trafo_loading_mae'].mean()),
        'pred_voltage_violations_mean': float(ts_out['pred_voltage_violations'].mean()),
        'true_voltage_violations_mean': float(ts_out['true_voltage_violations'].mean()),
        'pred_line_overloads_mean': float(ts_out['pred_line_overloads'].mean()),
        'true_line_overloads_mean': float(ts_out['true_line_overloads'].mean()),
        'pred_line_loss_mw_mean': float(ts_out['pred_line_loss_mw'].mean()),
        'true_line_loss_mw_mean': float(ts_out['true_line_loss_mw'].mean()),
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_out.to_csv(out_dir / 'powerflow_timeseries_metrics.csv', index=False)
    with open(out_dir / 'powerflow_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def summarize_runs(run_dirs, output_path: str, kind: str):
    rows = []
    for run_dir in [Path(p) for p in run_dirs]:
        metrics_path = run_dir / 'metrics.json'
        if not metrics_path.exists():
            continue
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        row = {'run_name': run_dir.name, 'run_dir': str(run_dir)}
        row.update({k: v for k, v in metrics.items() if not isinstance(v, (dict, list))})
        rows.append(row)

    df = pd.DataFrame(rows).sort_values('run_name').reset_index(drop=True) if rows else pd.DataFrame()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return {'kind': kind, 'rows': int(len(df)), 'output_csv': str(output)}
