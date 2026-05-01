import argparse
import copy
import json
from pathlib import Path

from physformer.pipelines import summarize_runs

from .commands import run_test, run_train
from .config import config_to_args, finalize_args, load_config, materialize_resolved_config, print_config

ABLATON_JOB_TO_ARG = {
    "no_phys_stream": "ablation_no_phys_stream",
    "no_battery_branch": "ablation_no_battery_branch",
    "no_soc_consistency": "ablation_no_soc_consistency",
    "no_future_weather": "ablation_no_future_weather",
    "no_battery_physics_loss": "ablation_no_battery_physics_loss",
    "no_temporal_decoder": "ablation_no_temporal_decoder",
}


def apply_job_overrides(args, cfg, job):
    training_cfg = cfg.setdefault("training", {})
    hardware_cfg = cfg.setdefault("hardware", {})

    if "batch_size" in job:
        args.batch_size = int(job["batch_size"])
        training_cfg["batch_size"] = int(job["batch_size"])
    if "epochs" in job:
        args.train_epochs = int(job["epochs"])
        training_cfg["train_epochs"] = int(job["epochs"])
    if "lr" in job:
        args.learning_rate = float(job["lr"])
        training_cfg["learning_rate"] = float(job["lr"])
    if "patience" in job:
        args.patience = int(job["patience"])
        training_cfg["patience"] = int(job["patience"])
    if "num_workers" in job:
        args.num_workers = int(job["num_workers"])
        hardware_cfg["num_workers"] = int(job["num_workers"])
    if "gpu" in job:
        args.gpu = int(job["gpu"])
        args.device_ids = [int(job["gpu"])]
        hardware_cfg["gpu"] = int(job["gpu"])
    if "pin_memory" in job:
        args.pin_memory = bool(job["pin_memory"])
        hardware_cfg["pin_memory"] = bool(job["pin_memory"])
    if "persistent_workers" in job:
        args.persistent_workers = bool(job["persistent_workers"])
        hardware_cfg["persistent_workers"] = bool(job["persistent_workers"])
    if "prefetch_factor" in job:
        args.prefetch_factor = int(job["prefetch_factor"])
        hardware_cfg["prefetch_factor"] = int(job["prefetch_factor"])
    if "log_interval" in job:
        args.log_interval = int(job["log_interval"])
        training_cfg["log_interval"] = int(job["log_interval"])
    if "warmup_epochs" in job:
        args.warmup_epochs = int(job["warmup_epochs"])
        training_cfg["warmup_epochs"] = int(job["warmup_epochs"])
    if "warmup_start_factor" in job:
        args.warmup_start_factor = float(job["warmup_start_factor"])
        training_cfg["warmup_start_factor"] = float(job["warmup_start_factor"])
    if "early_stop_metric" in job:
        args.early_stop_metric = str(job["early_stop_metric"])
        training_cfg["early_stop_metric"] = str(job["early_stop_metric"])
    if "early_stop_start_epoch" in job:
        args.early_stop_start_epoch = int(job["early_stop_start_epoch"])
        training_cfg["early_stop_start_epoch"] = int(job["early_stop_start_epoch"])
    if "soc_weight" in job:
        args.soc_weight = float(job["soc_weight"])
        training_cfg["soc_weight"] = float(job["soc_weight"])
    if "overlap_weight" in job:
        args.overlap_weight = float(job["overlap_weight"])
        training_cfg["overlap_weight"] = float(job["overlap_weight"])
    if "use_temporal_decoder" in job:
        args.use_temporal_decoder = bool(job["use_temporal_decoder"])
        cfg.setdefault("model", {})["use_temporal_decoder"] = bool(job["use_temporal_decoder"])
    if "film_scale" in job:
        args.film_scale = float(job["film_scale"])
        cfg.setdefault("model", {})["film_scale"] = float(job["film_scale"])

    if "ablation" in job:
        ablation_cfg = cfg.setdefault("ablation", {})
        for key, value in job["ablation"].items():
            ablation_cfg[key] = bool(value)
            arg_name = ABLATON_JOB_TO_ARG.get(key)
            if arg_name is not None:
                setattr(args, arg_name, bool(value))

    return args, cfg


def _prepare_job_args(job, cli_args, completed_jobs=None, current_seed=None):
    cfg = load_config(job["config"])
    args = config_to_args(cfg)
    base_run_name = job.get("run_name") or job.get("name") or getattr(args, "checkpoint_name", None)
    args.run_name = base_run_name
    args, cfg = finalize_args(args, cfg, cli_args)
    args, cfg = apply_job_overrides(args, cfg, job)

    if "init_from_job" in job and completed_jobs is not None:
        ref_name = job["init_from_job"]
        if ref_name not in completed_jobs:
            raise ValueError(f"init_from_job references '{ref_name}' but it has not completed yet.")
        ref_runs = completed_jobs[ref_name]
        if current_seed is not None and int(current_seed) in ref_runs:
            resolved = ref_runs[int(current_seed)]
        elif current_seed is not None and None in ref_runs:
            resolved = ref_runs[None]
        else:
            resolved = next(iter(ref_runs.values()))
        args.init_from_run = resolved
        cfg.setdefault("training", {})["init_from_run"] = resolved

    return args, cfg, base_run_name


def run_driver_jobs(driver_cfg, cli_args, job_kind):
    jobs = driver_cfg.get(job_kind, {}).get("jobs", [])
    if not jobs:
        raise ValueError(f"No jobs defined under '{job_kind}.jobs' in driver config.")

    if getattr(cli_args, "print_config", False):
        for job in jobs:
            seeds = job.get("seeds")
            if seeds is None:
                seeds = [getattr(cli_args, "seed", None)] if getattr(cli_args, "seed", None) is not None else [None]
            for seed in seeds:
                args, cfg, base_run_name = _prepare_job_args(job, cli_args)
                if seed is not None:
                    args.seed = int(seed)
                    args.run_name = f"{base_run_name}__s{seed}"
                    args.run_dir = str((Path.cwd() / ("runs" / Path(args.run_name))).resolve())
                    args.checkpoint_name = args.run_name
                    args.checkpoints = args.run_dir
                print(f"=== Driver Job: {job_kind}/{job.get('name', base_run_name)} ===")
                print_config(argparse.Namespace(**vars(args)))
                materialize_resolved_config(cfg, args)
        return {"status": "print_config_only", "jobs": len(jobs)}

    run_dirs = []
    completed_jobs = {}
    for job in jobs:
        job_name = job.get("run_name") or job.get("name")
        seeds = job.get("seeds")
        if seeds is None:
            seeds = [getattr(cli_args, "seed", None)] if getattr(cli_args, "seed", None) is not None else [None]

        job_run_dirs = {}
        for seed in seeds:
            args, cfg, base_run_name = _prepare_job_args(job, cli_args, completed_jobs, current_seed=seed)
            if seed is not None:
                args.seed = int(seed)
                cfg.setdefault("training", {})["seed"] = int(seed)
                args.run_name = f"{base_run_name}__s{seed}"
                args.run_dir = str((Path.cwd() / ("runs" / Path(args.run_name))).resolve())
                args.checkpoint_name = args.run_name
                args.checkpoints = args.run_dir

            exp = run_train(args, cfg)
            exp.test(load=True)
            run_dirs.append(args.run_dir)
            if seed is not None:
                job_run_dirs[int(seed)] = args.run_dir
            else:
                job_run_dirs[None] = args.run_dir

        if job_name:
            completed_jobs[job_name] = job_run_dirs

    driver_stem = Path(cli_args.config).stem
    summary_path = Path("runs") / "reports" / f"{driver_stem}_summary_raw.csv"
    summary = summarize_runs(run_dirs, str(summary_path), job_kind)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
