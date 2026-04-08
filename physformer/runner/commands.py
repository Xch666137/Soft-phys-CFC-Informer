import json
from pathlib import Path

from physformer.data.data_factory import data_provider
from physformer.exp.exp_baseline import BaselineExperiment
from physformer.pipelines import (
    build_multi_portfolio_dataset,
    export_portfolio_forecasts,
    validate_portfolio_forecasts,
)
from physformer.utils.tools import set_seed

from .config import config_to_args, load_config, materialize_resolved_config


def create_experiment(args):
    if args.model == "PhysFormer":
        from physformer.exp.exp_physformer import Exp_PhysFormer

        return Exp_PhysFormer(args)
    return BaselineExperiment(args)


def run_train(args, cfg):
    set_seed(getattr(args, "seed", 2024))
    resolved_cfg = materialize_resolved_config(cfg, args)
    exp = create_experiment(args)
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
    run_dir = Path(args.run_dir)
    include_operational_interface = bool(getattr(args, "include_operational_interface", False))
    output_name = "portfolio_forecasts_operational.csv" if include_operational_interface else "portfolio_forecasts.csv"
    output_csv = run_dir / "exports" / output_name
    output_path = export_portfolio_forecasts(
        load_config,
        config_to_args,
        data_provider,
        config_path,
        str(run_dir),
        str(output_csv),
        include_operational_interface=include_operational_interface,
    )
    print(f"Saved forecast export to: {output_path}")
    return output_path


def run_validate_powerflow(args, mapping_csv):
    run_dir = Path(args.run_dir)
    forecast_csv = run_dir / "exports" / "portfolio_forecasts.csv"
    summary = validate_portfolio_forecasts(str(forecast_csv), mapping_csv, str(run_dir / "powerflow"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def run_pipeline(args, cfg, cli_args):
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
