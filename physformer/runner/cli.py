import argparse

from .commands import (
    run_build_dataset,
    run_export_forecast,
    run_pipeline,
    run_test,
    run_train,
    run_validate_powerflow,
)
from .config import config_to_args, finalize_args, load_config, print_config
from .drivers import run_driver_jobs


def add_common_run_args(parser):
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--run-name", help="Explicit run name")
    parser.add_argument("--run-dir", help="Explicit run directory")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint if supported")
    parser.add_argument("--gpu", type=int, help="GPU id override")
    parser.add_argument("--seed", type=int, help="Random seed override")
    parser.add_argument("--print-config", action="store_true", help="Print effective args and exit")
    parser.add_argument("--epochs", type=int, help="Training epochs override")
    parser.add_argument("--lr", type=float, help="Learning rate override")
    parser.add_argument("--batch-size", type=int, help="Batch size override")
    parser.add_argument("--num-workers", type=int, help="Dataloader worker override")
    parser.add_argument("--patience", type=int, help="Early stopping patience override")
    parser.add_argument("--debug-nan", action="store_true", help="Enable anomaly detection")
    parser.add_argument("--save-gate-details", action="store_true", help="Save detailed gate values if model supports it")
    parser.add_argument("--init-from-run", help="Stage A run directory or checkpoint path for operational-fit initialization")


def build_parser():
    parser = argparse.ArgumentParser(description="Unified thesis runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_dataset_parser = subparsers.add_parser("build-dataset", help="Build the strict multi-portfolio benchmark dataset")
    build_dataset_parser.add_argument("--region-id", default="act_canberra")
    build_dataset_parser.add_argument("--nextgen-dir", default="data_raw/nextgen")
    build_dataset_parser.add_argument("--act-weather-csv", default="data_raw/era5/act_canberra_hourly.csv")
    build_dataset_parser.add_argument("--rye-generation-csv", default="data_raw/rye/rye_generation_and_load.csv")
    build_dataset_parser.add_argument("--rye-weather-csv", default="data_raw/era5/rye_template_hourly.csv")
    build_dataset_parser.add_argument("--output-dir", default="data_processed/multi_portfolio")
    build_dataset_parser.add_argument("--portfolio-size", type=int, default=5)
    build_dataset_parser.add_argument("--wind-penetration-target", type=float, default=0.15)
    build_dataset_parser.add_argument("--audit-year", type=int, default=2018)
    build_dataset_parser.add_argument("--source-timezone", default="Australia/Sydney")
    build_dataset_parser.add_argument("--min-feature-availability", type=float, default=0.99)

    train_parser = subparsers.add_parser("train", help="Train one experiment")
    add_common_run_args(train_parser)

    test_parser = subparsers.add_parser("test", help="Test one experiment")
    add_common_run_args(test_parser)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark driver config")
    add_common_run_args(benchmark_parser)

    ablation_parser = subparsers.add_parser("ablation", help="Run ablation driver config")
    add_common_run_args(ablation_parser)

    export_parser = subparsers.add_parser("export-forecast", help="Export forecast CSV from one run")
    add_common_run_args(export_parser)
    export_parser.add_argument("--include-operational-interface", action="store_true")

    validate_parser = subparsers.add_parser("validate-powerflow", help="Validate forecast via pandapower/SimBench")
    add_common_run_args(validate_parser)
    validate_parser.add_argument("--mapping-csv", required=True)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run thesis pipeline")
    add_common_run_args(pipeline_parser)
    pipeline_parser.add_argument("--region-id", default="act_canberra")
    pipeline_parser.add_argument("--nextgen-dir", default="data_raw/nextgen")
    pipeline_parser.add_argument("--act-weather-csv", default="data_raw/era5/act_canberra_hourly.csv")
    pipeline_parser.add_argument("--rye-generation-csv", default="data_raw/rye/rye_generation_and_load.csv")
    pipeline_parser.add_argument("--rye-weather-csv", default="data_raw/era5/rye_template_hourly.csv")
    pipeline_parser.add_argument("--output-dir", default="data_processed/multi_portfolio")
    pipeline_parser.add_argument("--portfolio-size", type=int, default=5)
    pipeline_parser.add_argument("--wind-penetration-target", type=float, default=0.15)
    pipeline_parser.add_argument("--audit-year", type=int, default=2018)
    pipeline_parser.add_argument("--source-timezone", default="Australia/Sydney")
    pipeline_parser.add_argument("--min-feature-availability", type=float, default=0.99)
    pipeline_parser.add_argument("--mapping-csv", required=True)

    return parser


def main():
    parser = build_parser()
    cli_args = parser.parse_args()

    if cli_args.command == "build-dataset":
        run_build_dataset(cli_args)
        return

    cfg = load_config(cli_args.config)

    if cli_args.command == "benchmark":
        run_driver_jobs(cfg, cli_args, "benchmark")
        return

    if cli_args.command == "ablation":
        run_driver_jobs(cfg, cli_args, "ablation")
        return

    args = config_to_args(cfg)
    args, cfg = finalize_args(args, cfg, cli_args)

    if cli_args.print_config:
        print_config(args)
        return

    if cli_args.command == "train":
        run_train(args, cfg)
    elif cli_args.command == "test":
        run_test(args, cfg)
    elif cli_args.command == "export-forecast":
        run_export_forecast(args, cli_args.config)
    elif cli_args.command == "validate-powerflow":
        run_validate_powerflow(args, cli_args.mapping_csv)
    elif cli_args.command == "pipeline":
        run_pipeline(args, cfg, cli_args)
    else:
        raise ValueError(f"Unsupported command: {cli_args.command}")
