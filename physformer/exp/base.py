import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


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
        self.model = None

    def _setup_logger(self):
        logger_name = f"{self.__class__.__name__}.{self.run_name}"
        logger = logging.getLogger(logger_name)
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
        logger.info(f"Log file: {log_file}")
        return logger

    def _setup_device(self):
        if getattr(self.args, 'use_gpu', False) and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device(f'cuda:{self.args.gpu}')
            self.logger.info(f'Use GPU: cuda:{self.args.gpu}')
            return device

        self.logger.info('Use CPU')
        return torch.device('cpu')

    def save_config(self, cfg: dict[str, Any]):
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

    def save_metrics_json(self, metrics: dict[str, Any]):
        serializable = self._to_serializable(metrics)
        metrics_path = self.run_dir / 'metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved metrics to: {metrics_path}")

    def save_numpy(self, relative_path: str, value: Any):
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, value)
        self.logger.info(f"Saved numpy artifact to: {path}")

    def save_npz(self, relative_path: str, value: Any):
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, dict):
            np.savez_compressed(path, **value)
        else:
            np.savez_compressed(path, arr=value)
        self.logger.info(f"Saved npz artifact to: {path}")

    def save_test_outputs(self, preds, trues, metrics: dict[str, Any], extras: dict[str, Any] | None = None):
        self.save_numpy('pred.npy', preds)
        self.save_numpy('true.npy', trues)
        self.save_metrics_json(metrics)

        legacy = metrics.get('legacy_array')
        if legacy is not None:
            self.save_numpy('metrics.npy', np.asarray(legacy))

        if not extras:
            return

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

    def _to_serializable(self, obj: Any):
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj


class ForecastExperiment(BaseExperiment):
    def __init__(self, args):
        super().__init__(args)


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, logger=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.logger = logger

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
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.logger:
                self.logger.info(
                    f'EarlyStopping counter: {self.counter} out of {self.patience} (Best: {-self.best_score:.6f})'
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
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...'
            )
        elif self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')

        torch.save(model.state_dict(), checkpoint_path)
        if optimizer is not None:
            state = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'scaler': scaler.state_dict() if scaler else None,
                'epoch': epoch,
                'global_step': global_step,
                'best_score': self.best_score,
                'val_loss_min': val_loss,
                'counter': self.counter
            }
            torch.save(state, state_path)

        self.val_loss_min = val_loss
