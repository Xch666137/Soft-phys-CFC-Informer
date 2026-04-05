# Multi-Portfolio Benchmark

This thesis branch uses a strict household-disjoint multi-portfolio benchmark
built from the NextGen-based semi-synthetic dataset.

## What it evaluates

- It is a **single-climate** benchmark under shared `ACT / Canberra` weather.
- Its main question is **composition-level generalization** across held-out
  portfolios.
- It does **not** claim cross-region or cross-climate generalization.

## Leakage rules

- `household_id` values are audited for eligibility before benchmark creation.
- `household_id` values are split into `train / val / test` before portfolio use.
- A household is never reused across portfolios in v1.
- The main benchmark uses `split_strategy=portfolio_manifest`, so held-out
  portfolios stay fully unseen during training.

## Builder entry

Unified runner entry:

```bash
python run.py build-dataset \
  --nextgen-dir data_raw/nextgen \
  --act-weather-csv data_raw/era5/act_canberra_hourly.csv \
  --rye-generation-csv data_raw/rye/rye_generation_and_load.csv \
  --rye-weather-csv data_raw/era5/rye_template_hourly.csv \
  --output-dir data_processed/multi_portfolio
```

Direct tool entry:

```bash
python tools/build_multi_portfolio_dataset.py \
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

The canonical thesis training table contains:

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

`cloud_cover` is retained at the raw weather stage but is not enabled as a v1
training covariate.

## Benchmarks

Main benchmark:

- File:
  `portfolio_dataset_for_training.csv`
- Split mode:
  `portfolio_manifest`
- Meaning:
  held-out portfolio composition generalization

Additional benchmark:

- File:
  `portfolio_dataset_for_time_generalization.csv`
- Split mode:
  `portfolio_manifest`
- Meaning:
  later-time evaluation on the train portfolios only

## Typical training commands

Baseline main benchmark:

```bash
python run.py train --config configs/baselines/tide_net_injection.yaml
```

PhysFormer main benchmark:

```bash
python run.py train --config configs/physformer_default.yaml
```

PhysFormer time generalization:

```bash
python run.py train --config configs/physformer_time_generalization.yaml
```

Main paper benchmark driver:

```bash
python run.py benchmark --config configs/drivers/benchmark_net_injection.yaml
```

This driver compares:

- `PhysFormer v2`
- `DLinear`
- `TiDE`
- `TimeXer`
- `TFT`

The main driver runs `3 seeds` per model:

- `2024`
- `2025`
- `2026`

It writes:

- raw per-run summary:
  `runs/reports/benchmark_net_injection_summary_raw.csv`
- grouped per-experiment summary with `mean/std`:
  `runs/reports/benchmark_net_injection_summary_grouped.csv`

Appendix benchmark driver:

```bash
python run.py benchmark --config configs/drivers/benchmark_net_injection_appendix.yaml
```

Appendix drivers stay single-seed unless you explicitly add `seeds` entries to
the driver YAML.
