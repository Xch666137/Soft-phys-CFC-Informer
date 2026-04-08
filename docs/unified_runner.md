# Unified Thesis Runner

This branch uses a single entrypoint:

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

## Config precedence

Resolved runtime config follows one order only:

1. base config
2. derived config via `_base_`
3. driver/job override
4. CLI override

The resolved config written to each run is:

```text
runs/<run_name>/config_merged.yaml
```

This file is now intended to match the actual runtime arguments used for the run.

## Main configs

Primary thesis configs:

- Stage A main benchmark:
  `configs/physformer_default.yaml`
- Stage A time generalization:
  `configs/physformer_time_generalization.yaml`
- Stage B operational fit:
  `configs/physformer_operational_fit.yaml`

Primary benchmark drivers:

- Main benchmark:
  `configs/drivers/benchmark_net_injection.yaml`
- Time generalization:
  `configs/drivers/benchmark_net_injection_time_generalization.yaml`
- PhysFormer ablation:
  `configs/drivers/physformer_ablation.yaml`

Legacy appendix configs are kept under:

```text
configs/legacy/
```

## Typical usage

Build the dataset:

```bash
python run.py build-dataset \
  --nextgen-dir data_raw/nextgen \
  --act-weather-csv data_raw/era5/act_canberra_hourly.csv \
  --rye-generation-csv data_raw/rye/rye_generation_and_load.csv \
  --rye-weather-csv data_raw/era5/rye_template_hourly.csv \
  --output-dir data_processed/multi_portfolio
```

Train one run:

```bash
python run.py train --config configs/physformer_default.yaml --run-name physformer_net_injection__s2024
```

Test one run:

```bash
python run.py test --config configs/physformer_default.yaml --run-name physformer_net_injection__s2024
```

Run the main benchmark driver:

```bash
python run.py benchmark --config configs/drivers/benchmark_net_injection.yaml
```

Run the time-generalization driver:

```bash
python run.py benchmark --config configs/drivers/benchmark_net_injection_time_generalization.yaml
```

Run the PhysFormer ablation driver:

```bash
python run.py ablation --config configs/drivers/physformer_ablation.yaml
```

Benchmark summaries are written as:

- raw:
  `runs/reports/<driver_name>_summary_raw.csv`
- grouped:
  `runs/reports/<driver_name>_summary_grouped.csv`

For AutoDL-specific submission, monitoring, and fetch flow, see:

`docs/autodl_automation.md`
