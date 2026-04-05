# Network Validation Chain

## Target definition

The thesis forecasting target is the portfolio-level net injection:

- `p_vpp_mw > 0`: net import from the grid
- `p_vpp_mw < 0`: net export to the grid

This is not a single bus target. Forecasting operates at portfolio level first,
then the network layer maps portfolio power into multiple buses using a fixed
allocation table.

## Shared fairness rule

All models in the forecasting benchmark follow the same input rule:

- Historical inputs:
  `p_vpp_mw + weather + battery state`
- Future known inputs:
  `temperature + irradiance + wind_speed + calendar features`
- Forbidden future inputs:
  true future battery trajectories

PhysFormer adds physics structure and auxiliary supervision, but it does not
receive extra future truth unavailable to the baselines.

## End-to-end chain

1. Build the strict multi-portfolio training dataset

```bash
python run.py build-dataset \
  --nextgen-dir data_raw/nextgen \
  --act-weather-csv data_raw/era5/act_canberra_hourly.csv \
  --rye-generation-csv data_raw/rye/rye_generation_and_load.csv \
  --rye-weather-csv data_raw/era5/rye_template_hourly.csv \
  --output-dir data_processed/multi_portfolio
```

2. Train a forecasting model

Baseline example:

```bash
conda activate Soft-phys-CFC-Informer
python run.py train --config configs/baselines/tide_net_injection.yaml
```

PhysFormer example:

```bash
conda activate Soft-phys-CFC-Informer
python run.py train --config configs/physformer_default.yaml
```

3. Export predictions with timestamps and portfolio IDs

```bash
python run.py export-forecast \
  --config configs/physformer_default.yaml \
  --run-name physformer_net_injection__s2024
```

4. Run network validation

```bash
python run.py validate-powerflow \
  --config configs/physformer_default.yaml \
  --run-name physformer_net_injection__s2024 \
  --mapping-csv templates/network_mapping.csv
```

For multi-seed benchmark runs, select a concrete seed run before export and
powerflow validation. The thesis default is to compare representative seed runs
rather than validating every seed.

## Outputs

- `runs/<run_name>/pred.npy`
- `runs/<run_name>/true.npy`
- `runs/<run_name>/metrics.json`
- `runs/<run_name>/exports/portfolio_forecasts.csv`
- `runs/<run_name>/powerflow/powerflow_timeseries_metrics.csv`
- `runs/<run_name>/powerflow/powerflow_summary.json`

For PhysFormer runs, the `extras/` directory also stores:

- `component_preds.npz`
- `component_trues.npz`
- `battery_state_preds.npz`
- `physics_states.npz`

These files are the thesis-grade bridge from forecasting outputs to
network-level usefulness metrics.
