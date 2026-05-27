import argparse
import copy
from pathlib import Path

import torch
import yaml

MODEL_KEYS = (
    "enc_in",
    "dec_in",
    "c_out",
    "d_model",
    "n_heads",
    "e_layers",
    "d_layers",
    "d_ff",
    "factor",
    "dropout",
    "attn",
    "embed",
    "activation",
    "use_rope",
    "rope_base",
    "distil",
    "output_attention",
    "use_temporal_decoder",
    "film_scale",
    "decoder_n_heads",
    "time_feat_dim",
    "load_gru_hidden",
    "load_gru_use_temp",
)

DATA_KEYS = (
    "root_path",
    "data_path",
    "features",
    "target",
    "seq_len",
    "pred_len",
    "label_len",
    "freq",
    "time_col",
    "id_col",
    "region_col",
    "split_col",
    "split_strategy",
    "task_mode",
    "target_cols",
    "covariate_cols",
    "known_future_covariate_cols",
    "history_state_cols",
    "aux_target_cols",
)

TRAINING_KEYS = (
    "batch_size",
    "train_epochs",
    "learning_rate",
    "weight_decay",
    "grad_clip",
    "patience",
    "use_amp",
    "seed",
    "log_interval",
    "warmup_epochs",
    "warmup_start_factor",
    "early_stop_metric",
    "early_stop_start_epoch",
    "soc_weight",
    "component_loss_weight",
    "restart_t0",
    "restart_t_mult",
    "res_reg_weight",
    "phase_1_epochs",
    "phase_2_epochs",
    "phase_1_cw",
    "phase_1_rr",
    "phase_2_cw",
    "phase_2_rr",
    "theory_loss_weight",
    "phase_1_tw",
    "phase_2_tw",
    "battery_component_weight",
    "detach_mode_phase2",
    "detach_scale",
)

HARDWARE_KEYS = (
    "use_gpu",
    "gpu",
    "num_workers",
    "use_multi_gpu",
    "device_ids",
    "pin_memory",
    "persistent_workers",
    "prefetch_factor",
)

LIST_KEYS = {
    "target_cols",
    "covariate_cols",
    "known_future_covariate_cols",
    "history_state_cols",
    "aux_target_cols",
    "device_ids",
}

ABLATION_KEY_MAP = {
    "no_phys_stream": "ablation_no_phys_stream",
    "no_battery_branch": "ablation_no_battery_branch",
    "no_soc_consistency": "ablation_no_soc_consistency",
    "no_future_weather": "ablation_no_future_weather",
    "no_battery_physics_loss": "ablation_no_battery_physics_loss",
    "no_temporal_decoder": "ablation_no_temporal_decoder",
    "no_deep_battery_context": "no_deep_battery_context",
}


def deep_update(base, override):
    for key, value in override.items():
        if key == "_base_":
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path):
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    if "_base_" in cfg:
        base_path = config_path.parent / cfg["_base_"]
        base_cfg = load_config(base_path)
        cfg = deep_update(base_cfg, cfg)

    return cfg


def _set_from_cfg(flat, source_cfg, keys):
    for key in keys:
        if key in source_cfg:
            flat[key] = source_cfg[key]


def config_to_args(cfg):
    flat = {}

    model_cfg = cfg.get("model", {})
    flat["model"] = model_cfg.get("name", "PhysFormer")
    flat["use_temporal_decoder"] = model_cfg.get("use_temporal_decoder", True)
    flat["film_scale"] = model_cfg.get("film_scale", 0.5)
    flat["decoder_n_heads"] = model_cfg.get("decoder_n_heads", None)
    flat["time_feat_dim"] = model_cfg.get("time_feat_dim", 8)
    _set_from_cfg(flat, model_cfg, MODEL_KEYS)

    data_cfg = cfg.get("data", {})
    _set_from_cfg(flat, data_cfg, DATA_KEYS)

    for key in LIST_KEYS:
        if key in data_cfg and data_cfg[key] is not None:
            flat[key] = list(data_cfg[key])

    training_cfg = cfg.get("training", {})
    flat["batch_size"] = training_cfg.get("batch_size", 128)
    flat["train_epochs"] = training_cfg.get("train_epochs", 100)
    flat["learning_rate"] = training_cfg.get("learning_rate", 3e-4)
    flat["weight_decay"] = training_cfg.get("weight_decay", 1e-5)
    flat["grad_clip"] = training_cfg.get("grad_clip", 1.0)
    flat["patience"] = training_cfg.get("patience", 10)
    flat["use_amp"] = training_cfg.get("use_amp", True)
    flat["seed"] = training_cfg.get("seed", 2024)
    flat["log_interval"] = training_cfg.get("log_interval", 50)
    flat["warmup_epochs"] = training_cfg.get("warmup_epochs", 0)
    flat["warmup_start_factor"] = training_cfg.get("warmup_start_factor", 0.2)
    flat["early_stop_metric"] = training_cfg.get("early_stop_metric", "net_mse")
    flat["early_stop_start_epoch"] = training_cfg.get("early_stop_start_epoch", 10)
    flat["soc_weight"] = training_cfg.get("soc_weight", 0.1)
    flat["component_loss_weight"] = training_cfg.get("component_loss_weight", 0.05)
    flat["res_reg_weight"] = training_cfg.get("res_reg_weight", 0.01)
    flat["phase_1_epochs"] = training_cfg.get("phase_1_epochs", 15)
    flat["phase_2_epochs"] = training_cfg.get("phase_2_epochs", 40)
    flat["phase_1_cw"] = training_cfg.get("phase_1_cw", 0.1)
    flat["phase_1_rr"] = training_cfg.get("phase_1_rr", 0.05)
    flat["phase_2_cw"] = training_cfg.get("phase_2_cw", 0.05)
    flat["phase_2_rr"] = training_cfg.get("phase_2_rr", 0.01)
    flat["theory_loss_weight"] = training_cfg.get("theory_loss_weight", 0.1)
    flat["phase_1_tw"] = training_cfg.get("phase_1_tw", 0.2)
    flat["phase_2_tw"] = training_cfg.get("phase_2_tw", 0.1)
    flat["battery_component_weight"] = training_cfg.get("battery_component_weight", 1.0)
    flat["restart_t0"] = training_cfg.get("restart_t0", 15)
    flat["restart_t_mult"] = training_cfg.get("restart_t_mult", 1)
    flat["detach_scale"] = float(training_cfg.get("detach_scale", 0.0))
    flat["phase_2a_epochs"] = training_cfg.get("phase_2a_epochs", flat["phase_1_epochs"])
    flat["phase_2a_cw"] = training_cfg.get("phase_2a_cw", flat["phase_2_cw"] * 2)
    flat["phase_reset_mode"] = training_cfg.get("phase_reset_mode", "soft")
    flat["detach_mode_phase2"] = training_cfg.get("detach_mode_phase2", "none")
    flat["use_compile"] = training_cfg.get("use_compile", True)
    flat["val_interval"] = training_cfg.get("val_interval", 1)
    flat["grad_angle_interval"] = training_cfg.get("grad_angle_interval", 1)

    hardware_cfg = cfg.get("hardware", {})
    flat["use_gpu"] = hardware_cfg.get("use_gpu", True)
    flat["gpu"] = hardware_cfg.get("gpu", 0)
    flat["num_workers"] = hardware_cfg.get("num_workers", 8)
    flat["use_multi_gpu"] = hardware_cfg.get("use_multi_gpu", False)
    flat["device_ids"] = hardware_cfg.get("device_ids", [flat["gpu"]])
    flat["pin_memory"] = hardware_cfg.get("pin_memory", True)
    flat["persistent_workers"] = hardware_cfg.get("persistent_workers", True)
    flat["prefetch_factor"] = hardware_cfg.get("prefetch_factor", 4)

    checkpoint_cfg = cfg.get("checkpoint", {})
    flat["checkpoint_name"] = checkpoint_cfg.get("name")
    flat["checkpoints"] = checkpoint_cfg.get("path", "runs/")

    ablation_cfg = cfg.get("ablation", {})
    for cfg_key, arg_key in ABLATION_KEY_MAP.items():
        flat[arg_key] = ablation_cfg.get(cfg_key, False)

    target_cols = data_cfg.get("target_cols")
    covariate_cols = data_cfg.get("covariate_cols")
    known_future_covariate_cols = data_cfg.get("known_future_covariate_cols")
    history_state_cols = data_cfg.get("history_state_cols")

    feature_target_cols = list(target_cols or [])
    feature_known_future_cols = list(known_future_covariate_cols or covariate_cols or [])
    feature_history_state_cols = list(history_state_cols or [])

    if feature_target_cols:
        flat["c_out"] = len(feature_target_cols)
        feature_count = len(feature_target_cols) + len(feature_known_future_cols) + len(feature_history_state_cols)
        flat["enc_in"] = feature_count
        flat["dec_in"] = feature_count

    if "label_len" not in flat:
        flat["label_len"] = 0 if flat.get("model", "").startswith("PhysFormer") else 96
    if "features" not in flat:
        flat["features"] = "M"

    return argparse.Namespace(**flat)


def apply_cli_overrides(args, cli_args):
    if cli_args.epochs is not None:
        args.train_epochs = cli_args.epochs
    if cli_args.lr is not None:
        args.learning_rate = cli_args.lr
    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size
    if cli_args.gpu is not None:
        args.gpu = cli_args.gpu
        args.device_ids = [cli_args.gpu]
    if getattr(cli_args, "num_workers", None) is not None:
        args.num_workers = cli_args.num_workers
    if cli_args.patience is not None:
        args.patience = cli_args.patience
    if getattr(cli_args, "seed", None) is not None:
        args.seed = cli_args.seed
    if cli_args.run_name is not None:
        args.run_name = cli_args.run_name
    if cli_args.run_dir is not None:
        args.run_dir = cli_args.run_dir
    if getattr(cli_args, "init_from_run", None) is not None:
        args.init_from_run = cli_args.init_from_run
    if getattr(cli_args, "resume", False):
        args.resume = True
    if getattr(cli_args, "debug_nan", False):
        args.debug_nan = True
    if getattr(cli_args, "save_gate_details", False):
        args.save_gate_details = True
    return args


def infer_dataset_name(data_path: str):
    name = Path(data_path).stem
    return name or "dataset"


def default_run_name(args):
    if getattr(args, "checkpoint_name", None):
        return args.checkpoint_name
    task_mode = getattr(args, "task_mode", "task")
    dataset = infer_dataset_name(getattr(args, "data_path", "data"))
    return f"{args.model}_{task_mode}_{dataset}_sl{args.seq_len}_pl{args.pred_len}"


def finalize_args(args, cfg, cli_args):
    args = apply_cli_overrides(args, cli_args)
    args.use_amp = bool(getattr(args, "use_amp", False))
    args.use_gpu = bool(getattr(args, "use_gpu", False)) and torch.cuda.is_available()
    args.resume = bool(getattr(args, "resume", False))
    args.debug_nan = bool(getattr(args, "debug_nan", False))
    args.save_gate_details = bool(getattr(args, "save_gate_details", False))
    args.mix = getattr(args, "mix", True)
    args.output_attention = getattr(args, "output_attention", False)
    args.seed = int(getattr(args, "seed", 2024))

    run_name = getattr(args, "run_name", None) or default_run_name(args)
    args.run_name = run_name
    run_root = Path("runs")
    run_dir = Path(getattr(args, "run_dir", run_root / run_name))
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    args.run_dir = str(run_dir)
    args.checkpoint_name = run_name
    args.checkpoints = str(run_dir)
    return args, cfg


def materialize_resolved_config(cfg, args):
    resolved = copy.deepcopy(cfg)

    model_cfg = resolved.setdefault("model", {})
    model_cfg["name"] = getattr(args, "model", model_cfg.get("name", "PhysFormer"))
    for key in MODEL_KEYS:
        if hasattr(args, key):
            model_cfg[key] = getattr(args, key)

    data_cfg = resolved.setdefault("data", {})
    for key in DATA_KEYS:
        if hasattr(args, key):
            value = getattr(args, key)
            if key in LIST_KEYS and value is not None:
                value = list(value)
            data_cfg[key] = value

    training_cfg = resolved.setdefault("training", {})
    for key in TRAINING_KEYS:
        if hasattr(args, key):
            training_cfg[key] = getattr(args, key)

    hardware_cfg = resolved.setdefault("hardware", {})
    for key in HARDWARE_KEYS:
        if hasattr(args, key):
            value = getattr(args, key)
            if key in LIST_KEYS and value is not None:
                value = list(value)
            hardware_cfg[key] = value

    checkpoint_cfg = resolved.setdefault("checkpoint", {})
    checkpoint_cfg["name"] = getattr(args, "run_name", checkpoint_cfg.get("name"))
    checkpoint_cfg["path"] = getattr(args, "run_dir", checkpoint_cfg.get("path", "runs/"))

    ablation_cfg = resolved.setdefault("ablation", {})
    for cfg_key, arg_key in ABLATION_KEY_MAP.items():
        if hasattr(args, arg_key):
            ablation_cfg[cfg_key] = bool(getattr(args, arg_key))

    return resolved


def print_config(args):
    print("=== Effective Arguments ===")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")
