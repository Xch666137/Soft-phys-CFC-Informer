# Multi-Portfolio Benchmark

This thesis branch uses a strict household-disjoint multi-portfolio benchmark
built from the NextGen-based semi-synthetic dataset.

## What it evaluates

- single-climate portfolio forecasting under shared `ACT / Canberra` weather
- held-out portfolio composition generalization
- later-time evaluation on train portfolios via time generalization

It does not claim cross-region or cross-climate generalization.

## Leakage rules

- `household_id` eligibility is audited before benchmark creation
- `household_id` is split into `train / val / test` before portfolio use
- a household is never reused across portfolios in v1
- the main benchmark uses `split_strategy=portfolio_manifest`

## Builder entry

Canonical builder command:

```bash
python run.py build-dataset \
  --nextgen-dir data_raw/nextgen \
  --act-weather-csv data_raw/era5/act_canberra_hourly.csv \
  --rye-generation-csv data_raw/rye/rye_generation_and_load.csv \
  --rye-weather-csv data_raw/era5/rye_template_hourly.csv \
  --output-dir data_processed/multi_portfolio
```

## Output files

The builder writes:

- `household_eligibility.csv`
- `household_exclusion_report.json`
- `multi_portfolio_timeseries.csv`
- `portfolio_membership.csv`
- `portfolio_summary.csv`
- `portfolio_dataset_for_training.csv`
- `portfolio_dataset_for_time_generalization.csv`
- `multi_portfolio_metadata.json`

## Training schema

Canonical thesis table fields:

- `date`
- `portfolio_id`
- `region_id`
- `split`
- `p_vpp_mw`
- `temperature`
- `irradiance`
- `wind_speed`
- `p_load_mw`
- `p_pv_mw`
- `p_wind_mw`
- `p_battery_mw`
- `e_battery_soc_mwh`

`cloud_cover` remains excluded from the v1 training feature contract.

## Benchmark drivers

Main benchmark:

```bash
python run.py benchmark --config configs/drivers/benchmark_net_injection.yaml
```

Time generalization:

```bash
python run.py benchmark --config configs/drivers/benchmark_net_injection_time_generalization.yaml
```

Main paper benchmark models:

- `PhysFormer v2`
- `DLinear`
- `TiDE`
- `TimeXer`
- `TFT`

Default seeds:

- `2024`
- `2025`
- `2026`

Appendix / legacy benchmark configs now live under:

```text
configs/legacy/
```
