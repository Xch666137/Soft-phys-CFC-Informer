import argparse
import json
from pathlib import Path

from .config import config_to_args, finalize_args, load_config, materialize_resolved_config
from .data import data_provider
from .train import set_seed, create_experiment, create_pretrain_experiment


def add_common_run_args(parser):
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--run-name", help="Explicit run name")
    parser.add_argument("--run-dir", help="Explicit run directory")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--gpu", type=int, help="GPU id override")
    parser.add_argument("--seed", type=int, help="Random seed override")
    parser.add_argument("--print-config", action="store_true", help="Print effective args and exit")
    parser.add_argument("--epochs", type=int, help="Training epochs override")
    parser.add_argument("--lr", type=float, help="Learning rate override")
    parser.add_argument("--batch-size", type=int, help="Batch size override")
    parser.add_argument("--num-workers", type=int, help="Dataloader worker override")
    parser.add_argument("--patience", type=int, help="Early stopping patience override")
    parser.add_argument("--debug-nan", action="store_true", help="Enable anomaly detection")
    parser.add_argument("--save-gate-details", action="store_true", help="Save detailed gate values")
    parser.add_argument("--init-from-run", help="Stage A run directory for operational-fit initialization")


def build_parser():
    parser = argparse.ArgumentParser(description="PhysFormer unified runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bd = subparsers.add_parser("build-dataset", help="Build multi-portfolio benchmark dataset")
    bd.add_argument("--region-id", default="act_canberra")
    bd.add_argument("--nextgen-dir", default="data_raw/nextgen")
    bd.add_argument("--act-weather-csv", default="data_raw/era5/act_canberra_hourly.csv")
    bd.add_argument("--rye-generation-csv", default="data_raw/rye/rye_generation_and_load.csv")
    bd.add_argument("--rye-weather-csv", default="data_raw/era5/rye_template_hourly.csv")
    bd.add_argument("--output-dir", default="data_processed/multi_portfolio")
    bd.add_argument("--portfolio-size", type=int, default=5)
    bd.add_argument("--wind-penetration-target", type=float, default=0.15)
    bd.add_argument("--audit-year", type=int, default=2018)
    bd.add_argument("--source-timezone", default="Australia/Sydney")
    bd.add_argument("--min-feature-availability", type=float, default=0.99)

    train_parser = subparsers.add_parser("train", help="Train one experiment")
    add_common_run_args(train_parser)

    pretrain_parser = subparsers.add_parser("pretrain", help="Masked Component Pretraining (Phase B)")
    add_common_run_args(pretrain_parser)

    test_parser = subparsers.add_parser("test", help="Test one experiment")
    add_common_run_args(test_parser)

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


def run_train(args, cfg):
    set_seed(getattr(args, "seed", 2024))
    resolved_cfg = materialize_resolved_config(cfg, args)
    exp = create_experiment(args)
    exp.save_config(resolved_cfg)
    exp.train()
    return exp


def run_pretrain(args, cfg):
    set_seed(getattr(args, "seed", 2024))
    resolved_cfg = materialize_resolved_config(cfg, args)
    exp = create_pretrain_experiment(args)
    exp.save_config(resolved_cfg)
    exp.train()
    return exp


def run_test(args, cfg):
    set_seed(getattr(args, "seed", 2024))
    resolved_cfg = materialize_resolved_config(cfg, args)
    exp = create_experiment(args)
    exp.save_config(resolved_cfg)
    exp.test(load=True)
    return exp


def run_build_dataset(cli_args):
    from physformer.pipelines.semisynthetic_vpp import build_multi_portfolio_dataset
    outputs = build_multi_portfolio_dataset(
        nextgen_dir=cli_args.nextgen_dir,
        act_weather_csv=cli_args.act_weather_csv,
        rye_generation_csv=cli_args.rye_generation_csv,
        rye_weather_csv=cli_args.rye_weather_csv,
        output_dir=cli_args.output_dir,
        portfolio_size=cli_args.portfolio_size,
        target_penetration=cli_args.wind_penetration_target,
        audit_year=cli_args.audit_year,
        source_timezone=cli_args.source_timezone,
        region_id=cli_args.region_id,
        min_feature_availability=cli_args.min_feature_availability,
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    return outputs


def run_export_forecast(args, config_path):
    from physformer.pipelines.thesis_ops import export_portfolio_forecasts
    run_dir = Path(args.run_dir)
    include_operational_interface = bool(getattr(args, "include_operational_interface", False))
    output_name = "portfolio_forecasts_operational.csv" if include_operational_interface else "portfolio_forecasts.csv"
    output_csv = run_dir / "exports" / output_name
    output_path = export_portfolio_forecasts(
        load_config, config_to_args, data_provider,
        config_path, str(run_dir), str(output_csv),
        include_operational_interface=include_operational_interface,
    )
    print(f"Saved forecast export to: {output_path}")
    return output_path


def run_validate_powerflow(args, mapping_csv):
    from physformer.pipelines.thesis_ops import validate_portfolio_forecasts
    run_dir = Path(args.run_dir)
    forecast_csv = run_dir / "exports" / "portfolio_forecasts.csv"
    summary = validate_portfolio_forecasts(str(forecast_csv), mapping_csv, str(run_dir / "powerflow"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def run_pipeline(args, cfg, cli_args):
    from physformer.pipelines.semisynthetic_vpp import build_multi_portfolio_dataset
    from physformer.pipelines.thesis_ops import export_portfolio_forecasts, validate_portfolio_forecasts
    run_dir = Path(args.run_dir)
    outputs = build_multi_portfolio_dataset(
        nextgen_dir=cli_args.nextgen_dir,
        act_weather_csv=cli_args.act_weather_csv,
        rye_generation_csv=cli_args.rye_generation_csv,
        rye_weather_csv=cli_args.rye_weather_csv,
        output_dir=cli_args.output_dir,
        portfolio_size=cli_args.portfolio_size,
        target_penetration=cli_args.wind_penetration_target,
        audit_year=cli_args.audit_year,
        source_timezone=cli_args.source_timezone,
        region_id=cli_args.region_id,
        min_feature_availability=cli_args.min_feature_availability,
    )
    data_path = outputs["portfolio_dataset_for_training_csv"]
    args.data_path = data_path
    cfg.setdefault("data", {})["data_path"] = data_path

    exp = run_train(args, cfg)
    exp.test(load=True)

    export_path = run_export_forecast(args, str(run_dir / "config_merged.yaml"))
    if cli_args.mapping_csv:
        summary = validate_portfolio_forecasts(str(export_path), cli_args.mapping_csv, str(run_dir / "powerflow"))
        print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    # Disable fused RNN/MIOpen paths for AMD GPU compatibility
    import torch
    for flag in ["cudnn", "miopen"]:
        backend = getattr(torch.backends, flag, None)
        if backend is not None and hasattr(backend, "enabled"):
            backend.enabled = False

    parser = build_parser()
    cli_args = parser.parse_args()

    if cli_args.command == "build-dataset":
        run_build_dataset(cli_args)
        return

    cfg = load_config(cli_args.config)
    args = config_to_args(cfg)
    args, cfg = finalize_args(args, cfg, cli_args)

    if cli_args.print_config:
        from .config import print_config
        print_config(args)
        return

    if cli_args.command == "train":
        run_train(args, cfg)
    elif cli_args.command == "pretrain":
        run_pretrain(args, cfg)
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
