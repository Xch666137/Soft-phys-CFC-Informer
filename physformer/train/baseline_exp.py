import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..data import data_provider
from ..metrics import compute_forecast_metrics
from ..models.autoformer import Autoformer
from ..models.dlinear import DLinear
from ..models.gru import GRU
from ..models.informer import Informer
from ..models.itransformer import iTransformer
from ..models.lstm import LSTM
from ..models.patchtst import PatchTST
from ..models.pinn import PINN
from ..models.persistence import Persistence
from ..models.tft import TFT
from ..models.tide import TiDE
from ..models.timexer import TimeXer
from .base import BaseExperiment, EarlyStopping


class BaselineExperiment(BaseExperiment):

    def __init__(self, args):
        super().__init__(args)
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model_dict = {
            "Informer": Informer, "Autoformer": Autoformer, "PINN": PINN,
            "DLinear": DLinear, "PatchTST": PatchTST, "iTransformer": iTransformer,
            "LSTM": LSTM, "GRU": GRU, "TFT": TFT, "TiDE": TiDE, "TimeXer": TimeXer,
            "Persistence": Persistence,
        }
        if self.args.model not in model_dict:
            raise ValueError(f"Model {self.args.model} not implemented")
        if self.args.model in ["Autoformer", "Informer", "iTransformer"]:
            if not hasattr(self.args, "moving_avg"):
                self.args.moving_avg = 5
            if not hasattr(self.args, "output_attention"):
                self.args.output_attention = False
            if not hasattr(self.args, "d_layers"):
                self.args.d_layers = 2
            if not hasattr(self.args, "distil"):
                self.args.distil = False
        model = model_dict[self.args.model](self.args).float()
        if getattr(self.args, "use_multi_gpu", False) and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=getattr(self.args, "device_ids", [self.args.gpu]))
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        wd = self.args.weight_decay if self.args.weight_decay > 0 else 1e-4
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=wd)

    def _select_criterion(self):
        return nn.MSELoss()

    def _estimate_ramp_limits(self, dataset):
        try:
            train_frame = getattr(dataset, "_train_feature_frame", None)
            target_cols = getattr(dataset, "target_cols", None)
            if train_frame is not None and target_cols:
                target_data = train_frame[target_cols].to_numpy(dtype=np.float32)
                diff = np.abs(target_data[1:] - target_data[:-1])
                if diff.size:
                    return np.percentile(diff, 99.9, axis=0) * 1.5

            raw_groups = []
            for group_tensor in dataset.group_x_tensors:
                group_np = group_tensor.cpu().numpy()
                raw_groups.append(dataset.inverse_transform(group_np))
            if raw_groups:
                raw = np.concatenate(raw_groups, axis=0)
                target_data = raw[:, : self.args.c_out]
                diff = np.abs(target_data[1:] - target_data[:-1])
                return np.percentile(diff, 99.9, axis=0) * 1.5
        except Exception as exc:
            self.logger.warning(f"Failed to estimate ramp limits from data: {exc}")
        return np.ones(self.args.c_out, dtype=np.float32)

    def _feature_role_slices(self):
        target_dim = len(getattr(self.args, "target_cols", [])) or self.args.c_out
        known_future_num = len(
            getattr(self.args, "known_future_covariate_cols", getattr(self.args, "covariate_cols", [])) or []
        )
        history_state_num = len(getattr(self.args, "history_state_cols", []) or [])
        cov_start = target_dim
        cov_end = cov_start + known_future_num
        state_start = cov_end
        state_end = state_start + history_state_num
        return {
            "target": (0, target_dim),
            "known_future": (cov_start, cov_end),
            "history_state": (state_start, state_end),
        }

    def _build_decoder_input(self, batch_y):
        role_slices = self._feature_role_slices()
        future_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
        cov_start, cov_end = role_slices["known_future"]
        if cov_end > cov_start:
            future_zeros[:, :, cov_start:cov_end] = batch_y[:, -self.args.pred_len :, cov_start:cov_end]
        return torch.cat([batch_y[:, : self.args.label_len, :], future_zeros], dim=1).float()

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device, non_blocking=True)
        batch_y = batch_y.float().to(self.device, non_blocking=True)
        batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
        batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)
        dec_inp = self._build_decoder_input(batch_y).to(self.device, non_blocking=True)
        return batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp

    def _forward_model(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        if self.args.model in ["Informer", "Autoformer", "iTransformer"]:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if getattr(self.args, "output_attention", False):
                outputs = outputs[0]
            return outputs
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def vali(self, vali_loader, criterion):
        self.model.eval()
        loss_total = None
        steps = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._process_one_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark,
                )
                outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len :, : self.args.c_out]
                batch_y = batch_y[:, -self.args.pred_len :, : self.args.c_out]
                loss = criterion(outputs, batch_y)
                loss_total = loss.detach() if loss_total is None else loss_total + loss.detach()
                steps += 1
        self.model.train()
        return float((loss_total / max(steps, 1)).detach().cpu()) if steps else 0.0

    def train(self):
        if self.args.model == "Persistence":
            torch.save(self.model.state_dict(), self.checkpoint_path())
            self.logger.info("Persistence baseline: training skipped (no learnable parameters).")
            return self.model
        train_data, train_loader = self._get_data(flag="train")
        _, vali_loader = self._get_data(flag="val")

        criterion = self._select_criterion()
        optimizer = self._select_optimizer()
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.train_epochs, eta_min=1e-6)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, logger=self.logger)
        use_amp = getattr(self.args, "use_amp", False) and self.args.use_gpu
        scaler = GradScaler(enabled=use_amp)
        log_interval = max(int(getattr(self.args, "log_interval", 50)), 1)

        self.logger.info(
            "Training throughput setup | samples=%d | steps_per_epoch=%d | batch_size=%d"
            % (len(train_data), len(train_loader), self.args.batch_size)
        )

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss_sum = None
            steps = 0
            epoch_time = time.time()
            if self.args.use_gpu and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)

            with tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{self.args.train_epochs}",
                mininterval=0.3, leave=False, ncols=100,
            ) as pbar:
                for step_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader, start=1):
                    optimizer.zero_grad()
                    batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._process_one_batch(
                        batch_x, batch_y, batch_x_mark, batch_y_mark,
                    )
                    with autocast(enabled=use_amp):
                        outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = outputs[:, -self.args.pred_len :, : self.args.c_out]
                        batch_y = batch_y[:, -self.args.pred_len :, : self.args.c_out]
                        loss = criterion(outputs, batch_y)

                    train_loss_sum = loss.detach() if train_loss_sum is None else train_loss_sum + loss.detach()
                    steps += 1
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    if step_idx % log_interval == 0 or step_idx == len(train_loader):
                        avg_loss = float((train_loss_sum / max(steps, 1)).detach().cpu())
                        pbar.set_postfix(loss=f"{avg_loss:.4f}")
                    pbar.update(1)

            epoch_seconds = time.time() - epoch_time
            train_loss = float((train_loss_sum / max(steps, 1)).detach().cpu()) if steps else 0.0
            vali_loss = self.vali(vali_loader, criterion)
            scheduler.step()
            samples_per_second = (steps * self.args.batch_size) / max(epoch_seconds, 1e-6)
            max_memory_gb = 0.0
            if self.args.use_gpu and torch.cuda.is_available():
                max_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

            self.logger.info(
                f"Epoch: {epoch + 1} | Cost: {epoch_seconds:.2f}s | Steps: {steps} | "
                f"Samples/s: {samples_per_second:.2f} | Max GPU Mem: {max_memory_gb:.2f} GB | "
                f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}"
            )
            early_stopping(
                vali_loss, self.model, str(self.run_dir),
                optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch + 1,
            )
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

        self.model.load_state_dict(torch.load(self.checkpoint_path(), map_location=self.device))
        return self.model

    def test(self, load=True, return_preds=False):
        test_data, test_loader = self._get_data(flag="test")
        if load:
            self.logger.info("loading model")
            self.model.load_state_dict(torch.load(self.checkpoint_path(), map_location=self.device))

        self.model.eval()
        preds = []
        trues = []
        last_hists = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._process_one_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark,
                )
                outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len :, : self.args.c_out]
                batch_y = batch_y[:, -self.args.pred_len :, : self.args.c_out]
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
                last_hists.append(batch_x[:, -1:, : self.args.c_out].detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)  # handles variable-length last batch
        trues = np.concatenate(trues, axis=0)
        last_hists = np.concatenate(last_hists, axis=0)

        if test_data.scale and test_data.feature_scaler is not None:
            preds = test_data.inverse_transform(preds)
            trues = test_data.inverse_transform(trues)
            last_hists = test_data.inverse_transform(last_hists)

        ramp_limits = self._estimate_ramp_limits(test_data)
        metrics = compute_forecast_metrics(preds, trues, ramp_limits=ramp_limits, last_hist=last_hists)
        target_cols = getattr(test_data, "target_cols", [f"target_{idx}" for idx in range(self.args.c_out)])
        metrics["per_channel_mse"] = {
            col: float(np.mean((preds[:, :, idx] - trues[:, :, idx]) ** 2)) for idx, col in enumerate(target_cols)
        }
        self.save_test_outputs(preds, trues, metrics)

        self.logger.info(f"MSE:{metrics['mse']:.6f}, MAE:{metrics['mae']:.6f}")
        self.logger.info(f"Ramp Violation:{metrics['net_ramp_violation']:.2f}%")

        if return_preds:
            return preds, trues, metrics
        return preds, trues
