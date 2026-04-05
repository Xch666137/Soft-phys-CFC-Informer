# Unified Thesis Runner

This worktree uses a single thesis runner entry:

```bash
python run.py <subcommand> ...
```

Supported subcommands:

- `build-dataset`
- `train`
- `test`
- `benchmark`
- `ablation`
- `export-forecast`
- `validate-powerflow`
- `pipeline`

All run artifacts are written to:

```text
runs/<run_name>/
```

## Dataset entrypoint

The only supported data build path in this branch is the strict
NextGen-based multi-portfolio benchmark:

```bash
python run.py build-dataset \
  --nextgen-dir data_raw/nextgen \
  --act-weather-csv data_raw/era5/act_canberra_hourly.csv \
  --rye-generation-csv data_raw/rye/rye_generation_and_load.csv \
  --rye-weather-csv data_raw/era5/rye_template_hourly.csv \
  --output-dir data_processed/multi_portfolio
```

This builder produces two training tables:

- `data_processed/multi_portfolio/portfolio_dataset_for_training.csv`
  Composition generalization benchmark with held-out portfolios.
- `data_processed/multi_portfolio/portfolio_dataset_for_time_generalization.csv`
  Time generalization benchmark built from the train portfolios only.

## Main configs

The thesis defaults are now:

- Main PhysFormer benchmark:
  `configs/physformer_default.yaml`
- Time generalization PhysFormer benchmark:
  `configs/physformer_time_generalization.yaml`
- Main baseline benchmark:
  `configs/baselines/dlinear_net_injection.yaml`
  `configs/baselines/tide_net_injection.yaml`
  `configs/baselines/timexer_net_injection.yaml`
  `configs/baselines/tft_net_injection.yaml`
- Time generalization baseline benchmark:
  `configs/baselines/dlinear_net_injection_time_generalization.yaml`
  `configs/baselines/tide_net_injection_time_generalization.yaml`
  `configs/baselines/timexer_net_injection_time_generalization.yaml`
  `configs/baselines/tft_net_injection_time_generalization.yaml`
- Appendix benchmark configs:
  `configs/baselines/itransformer_net_injection.yaml`
  `configs/baselines/informer_net_injection.yaml`
  `configs/baselines/lstm_net_injection.yaml`

All net-injection configs share the same feature contract:

- Historical inputs:
  `p_vpp_mw + temperature + irradiance + wind_speed + p_battery_mw + e_battery_soc_mwh`
- Future known inputs:
  `temperature + irradiance + wind_speed`
- Auxiliary supervision for PhysFormer:
  `p_load_mw + p_pv_mw + p_wind_mw + p_battery_mw + e_battery_soc_mwh`

## Typical Linux usage

Build the benchmark:

```bash
bash scripts/pipeline.sh \
  --config configs/baselines/tide_net_injection.yaml \
  --mapping-csv templates/network_mapping.csv \
  --nextgen-dir data_raw/nextgen \
  --act-weather-csv data_raw/era5/act_canberra_hourly.csv \
  --rye-generation-csv data_raw/rye/rye_generation_and_load.csv \
  --rye-weather-csv data_raw/era5/rye_template_hourly.csv \
  --output-dir data_processed/multi_portfolio
```

Run PhysFormer on the main benchmark:

```bash
bash scripts/train.sh --config configs/physformer_default.yaml --run-name physformer_net_injection
```

Run PhysFormer on the time-generalization benchmark:

```bash
bash scripts/train.sh --config configs/physformer_time_generalization.yaml --run-name physformer_net_injection_time_generalization
```

Run the main benchmark driver:

```bash
bash scripts/benchmark.sh --config configs/drivers/benchmark_net_injection.yaml
```

The main benchmark drivers are configured for `3 seeds`:

- `2024`
- `2025`
- `2026`

Each seed is written as an independent run:

```text
runs/<base_run_name>__s2024/
runs/<base_run_name>__s2025/
runs/<base_run_name>__s2026/
```

The runner writes two benchmark summaries:

- raw run-level summary:
  `runs/reports/benchmark_summary_raw.csv`
- grouped experiment summary with `mean/std` across seeds:
  `runs/reports/benchmark_summary_grouped.csv`

Run the appendix benchmark driver:

```bash
bash scripts/benchmark.sh --config configs/drivers/benchmark_net_injection_appendix.yaml
```

Appendix drivers remain single-seed by default.

Run the PhysFormer ablation driver:

```bash
bash scripts/ablation.sh --config configs/drivers/physformer_ablation.yaml
```

Legacy single-portfolio build flows are not part of the supported workflow in
this thesis branch.
