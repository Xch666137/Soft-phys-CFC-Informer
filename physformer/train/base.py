import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


def _unwrap_state_dict(model):
    """Return a clean state_dict with torch.compile _orig_mod. prefixes removed.

    Compatible with both compiled and uncompiled models (no-op if not compiled).
    Ensures checkpoints are always portable across compile states.
    """
    state = model.state_dict()
    if any("_orig_mod." in k for k in state.keys()):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    return state


def _strip_orig_mod_prefix(state_dict):
    """Strip _orig_mod. prefix from state_dict keys if present.

    Handles checkpoints saved by torch.compile models.
    """
    if any("_orig_mod." in k for k in state_dict.keys()):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


def _align_state_dict_keys(checkpoint_state, model_state):
    """Align checkpoint keys to match model's expected keys.

    Handles torch.compile prefix mismatch in both directions.
    Returns aligned checkpoint state_dict.
    """
    ckpt_has_orig = any("_orig_mod." in k for k in checkpoint_state.keys())
    model_has_orig = any("_orig_mod." in k for k in model_state.keys())
    
    if ckpt_has_orig == model_has_orig:
        # Both same style, no conversion needed
        return checkpoint_state
    elif ckpt_has_orig and not model_has_orig:
        # Checkpoint compiled, model not → strip prefix
        return {k.replace("_orig_mod.", ""): v for k, v in checkpoint_state.items()}
    else:
        # Checkpoint clean, model compiled → add prefix back
        # We need to map clean keys to compiled keys
        # Build mapping from model's state_dict
        clean_to_compiled = {}
        for mk in model_state.keys():
            clean = mk.replace("_orig_mod.", "")
            clean_to_compiled[clean] = mk
        
        aligned = {}
        for ck, cv in checkpoint_state.items():
            if ck in clean_to_compiled:
                aligned[clean_to_compiled[ck]] = cv
            else:
                aligned[ck] = cv
        return aligned


def set_seed(seed=2024):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Base experiment infrastructure
# ---------------------------------------------------------------------------

class BaseExperiment:
    def __init__(self, args):
        self.args = args
        self.run_name = getattr(args, 'run_name', None) or getattr(args, 'checkpoint_name', 'run')
        self.run_dir = Path(getattr(args, 'run_dir', Path('runs') / self.run_name))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.extras_dir = self.run_dir / 'extras'
        self.reports_dir = self.run_dir / 'reports'
        self.exports_dir = self.run_dir / 'exports'
        self.powerflow_dir = self.run_dir / 'powerflow'
        self.extras_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.powerflow_dir.mkdir(parents=True, exist_ok=True)

        self.args.run_name = self.run_name
        self.args.run_dir = str(self.run_dir)
        self.args.checkpoint_name = self.run_name
        self.args.checkpoints = str(self.run_dir)

        self.logger = self._setup_logger()
        self.device = self._setup_device()
        self._log_backend_policy()
        self.model = None

    def _setup_logger(self):
        import logging
        logger = logging.getLogger(f"{self.__class__.__name__}.{self.run_name}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if logger.hasHandlers():
            logger.handlers.clear()
        log_file = self.run_dir / 'train.log'
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.info(f"Run directory: {self.run_dir}")
        return logger

    def _setup_device(self):
        import os
        if getattr(self.args, 'use_gpu', False) and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            device = torch.device(f'cuda:{self.args.gpu}')
            self.logger.info(f'Use GPU: cuda:{self.args.gpu}')
            return device
        self.logger.info('Use CPU')
        return torch.device('cpu')

    def _backend_enabled(self, name):
        backend = getattr(torch.backends, name, None)
        if backend is None or not hasattr(backend, "enabled"):
            return None
        return bool(backend.enabled)

    def _device_name(self):
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                return torch.cuda.get_device_name(self.device)
            except Exception:
                return str(self.device)
        return str(self.device)

    def _log_backend_policy(self):
        self.logger.info(
            "Backend policy | use_compile=%s | cudnn.enabled=%s | miopen.enabled=%s | device=%s",
            bool(getattr(self.args, "use_compile", True)),
            self._backend_enabled("cudnn"),
            self._backend_enabled("miopen"),
            self._device_name(),
        )

    def save_config(self, cfg):
        import json
        config_path = self.run_dir / 'config_merged.yaml'
        try:
            import yaml
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        except Exception:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved merged config to: {config_path}")

    def checkpoint_path(self):
        return self.run_dir / 'checkpoint.pth'

    def training_state_path(self):
        return self.run_dir / 'training_state.pth'

    def save_metrics_json(self, metrics):
        import json
        metrics_path = self.run_dir / 'metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self._to_serializable(metrics), f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved metrics to: {metrics_path}")

    def save_numpy(self, relative_path, value):
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, value)
        self.logger.info(f"Saved numpy artifact to: {path}")

    def save_npz(self, relative_path, value):
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, dict):
            np.savez_compressed(path, **value)
        else:
            np.savez_compressed(path, arr=value)
        self.logger.info(f"Saved npz artifact to: {path}")

    def save_test_outputs(self, preds, trues, metrics, extras=None):
        self.save_numpy('pred.npy', preds)
        self.save_numpy('true.npy', trues)
        self.save_metrics_json(metrics)
        legacy = metrics.get('legacy_array')
        if legacy is not None:
            self.save_numpy('metrics.npy', np.asarray(legacy))
        if not extras:
            return
        import json
        for name, value in extras.items():
            if name.endswith('.json'):
                out_path = self.extras_dir / name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(self._to_serializable(value), f, indent=2, ensure_ascii=False)
                self.logger.info(f"Saved JSON artifact to: {out_path}")
            elif name.endswith('.npz'):
                self.save_npz(str(Path('extras') / name), value)
            else:
                self.save_numpy(str(Path('extras') / name), value)

    def _to_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, logger=None, metric_name="Validation loss", start_epoch=1):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.logger = logger
        self.metric_name = metric_name
        self.start_epoch = max(int(start_epoch), 1)

    def __call__(self, val_loss, model, path, optimizer=None, scheduler=None, scaler=None, epoch=None, global_step=None):
        if not np.isfinite(val_loss):
            if self.logger:
                self.logger.warning(f'Warning: EarlyStopping encountered invalid val_loss: {val_loss}. Skipping save.')
            elif self.verbose:
                print(f'Warning: EarlyStopping encountered invalid val_loss: {val_loss}. Skipping save.')
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, optimizer, scheduler, scaler, epoch, global_step)
            return

        if epoch is not None and int(epoch) < self.start_epoch:
            if score >= self.best_score + self.delta:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path, optimizer, scheduler, scaler, epoch, global_step)
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.logger:
                self.logger.info(
                    f'EarlyStopping counter: {self.counter} out of {self.patience} '
                    f'({self.metric_name} best: {-self.best_score:.6f})'
                )
            elif self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, optimizer, scheduler, scaler, epoch, global_step)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, optimizer=None, scheduler=None, scaler=None, epoch=None, global_step=None):
        checkpoint_path = Path(path) / 'checkpoint.pth'
        state_path = Path(path) / 'training_state.pth'
        if self.logger:
            self.logger.info(
                f'{self.metric_name} improved ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...'
            )
        elif self.verbose:
            print(f'{self.metric_name} improved ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(_unwrap_state_dict(model), checkpoint_path)
        if optimizer is not None:
            state = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'scaler': scaler.state_dict() if scaler else None,
                'epoch': epoch,
                'global_step': global_step,
                'best_score': self.best_score,
                'val_loss_min': val_loss,
                'counter': self.counter,
            }
            torch.save(state, state_path)
        self.val_loss_min = val_loss
