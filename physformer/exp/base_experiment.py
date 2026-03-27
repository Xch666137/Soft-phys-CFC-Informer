import logging
import os
from contextlib import nullcontext

import numpy as np
import torch
import yaml

from ..data.data_factory import data_provider


class BaseExperiment:
    def __init__(self, args):
        self.args = args
        self._data_cache = {}
        self._writer_closed = False
        self.experiment_dir = os.path.join(self.args.checkpoints, self.args.checkpoint_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        self._init_logger()
        self.device = self._acquire_device()
        self._save_resolved_config()

    def close(self):
        if getattr(self, 'writer', None) is None or self._writer_closed:
            return

        self.writer.flush()
        self.writer.close()
        self._writer_closed = True

    def _resolve_setting_dir(self, setting=None, create=False):
        if not setting or setting == self.args.checkpoint_name:
            path = self.experiment_dir
        else:
            path = os.path.join(self.args.checkpoints, setting)

        if create:
            os.makedirs(path, exist_ok=True)

        return path

    def _checkpoint_path(self, setting=None):
        return os.path.join(self._resolve_setting_dir(setting), 'checkpoint.pth')

    def _training_state_path(self, setting=None):
        return os.path.join(self._resolve_setting_dir(setting), 'training_state.pth')

    def _load_model_checkpoint(self, setting=None):
        checkpoint_path = self._checkpoint_path(setting)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        )
        return checkpoint_path

    def _load_training_state(self, setting=None):
        state_path = self._training_state_path(setting)
        if not os.path.exists(state_path):
            return None

        return torch.load(state_path, map_location=self.device, weights_only=False)

    def _amp_enabled(self):
        return bool(getattr(self.args, 'use_amp', False) and self.device.type == 'cuda')

    def _create_grad_scaler(self):
        enabled = self._amp_enabled()
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            try:
                return torch.amp.GradScaler('cuda', enabled=enabled)
            except TypeError:
                return torch.amp.GradScaler(enabled=enabled)

        return torch.cuda.amp.GradScaler(enabled=enabled)

    def _autocast_context(self, enabled):
        if self.device.type != 'cuda':
            return nullcontext()

        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            try:
                return torch.amp.autocast('cuda', enabled=enabled)
            except TypeError:
                return torch.amp.autocast(enabled=enabled)

        return torch.cuda.amp.autocast(enabled=enabled)

    def _build_early_stopping(self):
        return EarlyStopping(patience=self.args.patience, verbose=True, logger=self.logger)

    def _restore_training_runtime(
        self,
        setting=None,
        optimizer=None,
        scheduler=None,
        scaler=None,
        criterion=None,
        early_stopping=None,
    ):
        start_epoch = 0
        global_step = 0
        checkpoint_path = self._checkpoint_path(setting)

        if not getattr(self.args, 'resume', False) or not os.path.exists(checkpoint_path):
            return start_epoch, global_step

        path = self._resolve_setting_dir(setting)
        self.logger.info(f">>> [RESUME] Loading checkpoint from {path} <<<")
        self._load_model_checkpoint(setting)

        state = self._load_training_state(setting)
        if state is None:
            self.logger.warning(">>> [RESUME WARNING] training_state.pth not found! Resuming model weights ONLY. <<<")
            return start_epoch, global_step

        if optimizer is not None and state.get('optimizer') is not None:
            optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None and state.get('scheduler') is not None:
            scheduler.load_state_dict(state['scheduler'])
        if scaler is not None and state.get('scaler') is not None:
            scaler.load_state_dict(state['scaler'])
        if criterion is not None and state.get('criterion') is not None:
            criterion.load_state_dict(state['criterion'])
        if early_stopping is not None:
            early_stopping.best_score = state.get('best_score')
            early_stopping.val_loss_min = state.get('val_loss_min', early_stopping.val_loss_min)
            early_stopping.counter = state.get('counter', 0)

        resumed_epoch = state.get('epoch')
        if resumed_epoch is not None:
            start_epoch = resumed_epoch + 1
        global_step = state.get('global_step', 0)

        if resumed_epoch is not None:
            self.logger.info(
                f">>> [RESUME] Successfully resumed from Epoch {resumed_epoch} (Next: {start_epoch}) <<<"
            )

        return start_epoch, global_step

    def _clip_gradients(self, parameters, optimizer=None, scaler=None, already_unscaled=False):
        max_norm = getattr(self.args, 'grad_clip', None)
        if max_norm is None or max_norm <= 0:
            return None

        if scaler is not None and self._amp_enabled() and optimizer is not None and not already_unscaled:
            scaler.unscale_(optimizer)

        return torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)

    def _backward_step(self, loss, optimizer, scaler=None, parameters=None, pre_step_hook=None):
        parameters = parameters if parameters is not None else self.model.parameters()
        use_amp = scaler is not None and self._amp_enabled()

        if use_amp:
            scaler.scale(loss).backward()
            already_unscaled = False
            if pre_step_hook is not None:
                scaler.unscale_(optimizer)
                already_unscaled = True
                pre_step_hook()

            self._clip_gradients(
                parameters,
                optimizer=optimizer,
                scaler=scaler,
                already_unscaled=already_unscaled,
            )
            scaler.step(optimizer)
            scaler.update()
            return

        loss.backward()
        if pre_step_hook is not None:
            pre_step_hook()

        self._clip_gradients(parameters)
        optimizer.step()

    def _save_numpy_artifacts(self, setting=None, **artifacts):
        folder_path = self._resolve_setting_dir(setting, create=True)
        for name, value in artifacts.items():
            np.save(os.path.join(folder_path, f'{name}.npy'), value)
        return folder_path

    def _finalize_training(self, setting=None):
        self._load_model_checkpoint(setting)
        if getattr(self, 'writer', None) is not None:
            self.writer.flush()
        return self.model

    def _save_resolved_config(self):
        resolved_config = getattr(self.args, 'resolved_config', None)
        if not resolved_config:
            return

        config_path = os.path.join(self.experiment_dir, 'resolved_config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(resolved_config, f, sort_keys=False, allow_unicode=True)

    def _init_logger(self):
        log_dir = os.path.join(self.args.checkpoints, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f'train_log_{self.args.checkpoint_name}.txt')
        model_label = getattr(self.args, 'model', 'Experiment')

        self.logger = logging.getLogger(f"{model_label}.{self.args.checkpoint_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(message)s'))

        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        self.logger.info(f"Log file created at: {log_file}")

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.args.gpu}')
            self.logger.info(f'Use GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            self.logger.info('Use CPU')
        return device

    def _get_data(self, flag):
        cached = self._data_cache.get(flag)
        if cached is not None:
            return cached

        data_set, data_loader = data_provider(self.args, flag)
        self._data_cache[flag] = (data_set, data_loader)
        return data_set, data_loader


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

    def __call__(
        self,
        val_loss,
        model,
        path,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=None,
        global_step=None,
        criterion=None,
    ):
        if not np.isfinite(val_loss):
            if self.logger:
                self.logger.warning(
                    f'Warning: EarlyStopping encountered invalid val_loss: {val_loss}. Skipping save.'
                )
            elif self.verbose:
                print(f'Warning: EarlyStopping encountered invalid val_loss: {val_loss}. Skipping save.')

            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, optimizer, scheduler, scaler, epoch, global_step, criterion)
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
            self.save_checkpoint(val_loss, model, path, optimizer, scheduler, scaler, epoch, global_step, criterion)
            self.counter = 0

    def save_checkpoint(
        self,
        val_loss,
        model,
        path,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=None,
        global_step=None,
        criterion=None,
    ):
        if self.logger:
            self.logger.info(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        elif self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        if optimizer is not None:
            state = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'scaler': scaler.state_dict() if scaler else None,
                'criterion': criterion.state_dict() if criterion is not None else None,
                'epoch': epoch,
                'global_step': global_step,
                'best_score': self.best_score,
                'val_loss_min': val_loss,
                'counter': self.counter,
            }
            torch.save(state, os.path.join(path, 'training_state.pth'))

        self.val_loss_min = val_loss
