import argparse
import json
import sys
import traceback
from pathlib import Path

import yaml

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
    parser.add_argument(
        "--disable-fused-rnn-backends",
        action="store_true",
        help="Disable cuDNN/MIOpen fused RNN backends for ROCm compatibility debugging",
    )


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

    bench_parser = subparsers.add_parser("benchmark", help="Run a batch of experiments from a driver YAML")
    bench_parser.add_argument("--config", required=True, help="Path to benchmark driver YAML")
    bench_parser.add_argument("--gpu", type=int, help="GPU id override")
    bench_parser.add_argument("--print-config", action="store_true", help="Print parsed jobs and exit (dry run)")
    bench_parser.add_argument("--disable-fused-rnn-backends", action="store_true",
                              help="Disable cuDNN/MIOpen fused RNN backends")
    bench_parser.add_argument("--stop-on-error", action="store_true",
                              help="Stop on first job failure instead of continuing")

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


# ── driver override keys mapped to config sections ─────────────────────
_TRAINING_OVERRIDES = {
    "batch_size", "lr", "patience", "warmup_epochs", "warmup_start_factor",
    "early_stop_metric", "early_stop_start_epoch", "log_interval", "val_interval",
}
_HARDWARE_OVERRIDES = {
    "num_workers", "pin_memory", "persistent_workers", "prefetch_factor",
}


def _parse_driver(driver_path):
    """Parse a benchmark/ablation driver YAML.

    Supports top-level keys ``benchmark:`` and ``ablation:``.
    Returns ``(driver_label, jobs_list)``.
    """
    with open(driver_path, "r", encoding="utf-8") as fh:
        driver = yaml.safe_load(fh) or {}

    for top_key in ("benchmark", "ablation"):
        if top_key in driver:
            return top_key, driver[top_key].get("jobs", [])

    available = [k for k in driver if isinstance(driver[k], dict) and "jobs" in driver[k]]
    if available:
        top_key = available[0]
        return top_key, driver[top_key]["jobs"]
    raise ValueError(f"No 'benchmark:' or 'ablation:' section found in {driver_path}")


def run_benchmark(cli_args):
    """Batch-launch experiments from a driver YAML."""
    driver_path = Path(cli_args.config)
    driver_label, jobs = _parse_driver(driver_path)

    if not jobs:
        print(f"[benchmark] No jobs found in {driver_path} ({driver_label})")
        return

    print(f"[benchmark] Driver : {driver_path}")
    print(f"[benchmark] Section: {driver_label}")
    print(f"[benchmark] Jobs   : {len(jobs)}")

    if cli_args.print_config:
        for job in jobs:
            seeds = job.get("seeds", [2024])
            print(f"  {job['name']}  config={job['config']}  seeds={seeds}  "
                  f"runs={len(seeds)}  total={len(seeds)}")
        return

    # Accumulators for final summary
    completed = []
    failed = []

    for job_idx, job in enumerate(jobs):
        job_name = job.get("name", job.get("run_name", f"job-{job_idx}"))
        base_config_path = job["config"]
        seeds = job.get("seeds", [2024])

        for seed in seeds:
            run_name_seed = f"{job.get('run_name', job_name)}_s{seed}"
            label = f"[benchmark] {job_name} / seed={seed}"
            print(f"\n{'='*60}")
            print(f"{label}  RUN: {run_name_seed}")
            print(f"{'='*60}")

            try:
                # 1. Load base config
                cfg = load_config(base_config_path)

                # 2. Apply training overrides from driver
                training = cfg.setdefault("training", {})
                for key in _TRAINING_OVERRIDES:
                    if key in job:
                        # Map 'lr' → 'learning_rate' in training dict
                        mapped = "learning_rate" if key == "lr" else key
                        training[mapped] = job[key]
                if "train_epochs" in job:
                    training["train_epochs"] = job["train_epochs"]

                # 3. Apply hardware overrides
                hw = cfg.setdefault("hardware", {})
                for key in _HARDWARE_OVERRIDES:
                    if key in job:
                        hw[key] = job[key]

                # 4. Merge ablation flags (write-through to config)
                if "ablation" in job:
                    cfg_ablation = cfg.setdefault("ablation", {})
                    cfg_ablation.update(job["ablation"])

                # 5. Checkpoint name
                cfg.setdefault("checkpoint", {})["name"] = run_name_seed

                # 6. Convert to args + finalize
                args = config_to_args(cfg)
                args.seed = seed

                # Build a minimal CLI namespace so finalize_args can apply its
                # defaults without requiring an actual CLI invocation.
                cli_ns = argparse.Namespace(
                    epochs=None, lr=None, batch_size=None, gpu=cli_args.gpu,
                    num_workers=None, patience=None, seed=seed,
                    run_name=run_name_seed, run_dir=None, init_from_run=None,
                    resume=False, debug_nan=False, save_gate_details=False,
                    disable_fused_rnn_backends=cli_args.disable_fused_rnn_backends,
                )
                args, cfg = finalize_args(args, cfg, cli_ns)
                apply_backend_policy(args)

                # 7. Train
                set_seed(seed)
                resolved_cfg = materialize_resolved_config(cfg, args)
                exp = create_experiment(args)
                exp.save_config(resolved_cfg)
                exp.train()
                exp.test(load=True)

                print(f"{label}  DONE")
                completed.append({"job": job_name, "seed": seed, "run_name": run_name_seed})

            except Exception:
                print(f"{label}  FAILED", file=sys.stderr)
                traceback.print_exc()
                failed.append({"job": job_name, "seed": seed, "run_name": run_name_seed})
                if cli_args.stop_on_error:
                    raise

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"[benchmark] COMPLETE  completed={len(completed)}  failed={len(failed)}")
    if completed:
        print("[benchmark] Completed runs:")
        for r in completed:
            print(f"  {r['run_name']}")
    if failed:
        print("[benchmark] Failed runs:")
        for r in failed:
            print(f"  {r['run_name']}")
    print(f"{'='*60}")


def apply_backend_policy(args):
    import torch

    is_rocm = bool(getattr(torch.version, "hip", None))
    disable_fused = bool(getattr(args, "disable_fused_rnn_backends", False)) or is_rocm
    for flag in ["cudnn", "miopen"]:
        backend = getattr(torch.backends, flag, None)
        if backend is not None and hasattr(backend, "enabled"):
            backend.enabled = not disable_fused


def main():
    parser = build_parser()
    cli_args = parser.parse_args()

    if cli_args.command == "build-dataset":
        run_build_dataset(cli_args)
        return

    if cli_args.command == "benchmark":
        run_benchmark(cli_args)
        return

    cfg = load_config(cli_args.config)
    args = config_to_args(cfg)
    args, cfg = finalize_args(args, cfg, cli_args)
    apply_backend_policy(args)

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
