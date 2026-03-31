# Network Validation Chain

## Target definition

The single forecasting target for the thesis mainline is the portfolio-level net
load:

- `p_vpp_mw > 0`: net import from the grid
- `p_vpp_mw < 0`: net export to the grid

This is **not** a single bus net load target. The forecasting layer predicts a
portfolio-level quantity first, and the network layer maps it into multiple
network buses using a fixed allocation table.

## End-to-end chain

1. Build a canonical forecasting table

```bash
python tools/build_portfolio_dataset.py ^
  --input_csv E:\\Py_program\\Soft-phys-CFC-Informer\\data\\vpp_dataset_3years.csv ^
  --output_csv data/portfolio_net_injection.csv ^
  --portfolio_id residential_dominant_01 ^
  --region_id region_a
```

2. Train a baseline net-load forecasting model on the remote GPU machine

```bash
conda activate Soft-phys-CFC-Informer
python run.py --config configs/baselines/informer_net_injection.yaml
```

3. Export predictions with timestamps and portfolio IDs

```bash
python analysis/export_portfolio_forecasts.py ^
  --config configs/baselines/informer_net_injection.yaml ^
  --experiment_dir checkpoints/Baselines/Informer_portfolio_net_injection_sl672_pl96_vpp ^
  --output_csv analysis_outputs/portfolio_forecasts.csv
```

4. Run network validation

```bash
python analysis/validate_portfolio_powerflow.py ^
  --forecast_csv analysis_outputs/portfolio_forecasts.csv ^
  --mapping_csv templates/network_mapping.csv ^
  --output_dir analysis_outputs/powerflow_validation
```

## Outputs

- `powerflow_timeseries_metrics.csv`
- `powerflow_summary.json`

These files are the minimal thesis-grade bridge from forecasting outputs to
network-level usefulness metrics.
