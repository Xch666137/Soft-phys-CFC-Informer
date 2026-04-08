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

from physformer.runner.cli import main


if __name__ == "__main__":
    main()
