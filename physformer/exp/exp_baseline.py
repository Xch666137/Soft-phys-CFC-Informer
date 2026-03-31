import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .base import EarlyStopping, ForecastExperiment
from ..data.data_factory import data_provider
from ..models.autoformer import Autoformer
from ..models.dlinear import DLinear
from ..models.gru import GRU
from ..models.informer import Informer
from ..models.itransformer import iTransformer
from ..models.lstm import LSTM
from ..models.patchtst import PatchTST
from ..models.pinn import PINN
from ..utils.metrics import metric

warnings.filterwarnings('ignore')


class PINN_SoftConstraintLoss(nn.Module):
    def __init__(self, means, stds, ramp_limits, lambda_bvr=1.0, lambda_rvr=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_bvr = lambda_bvr
        self.lambda_rvr = lambda_rvr
        self.means = torch.tensor(means, dtype=torch.float32).view(1, 1, -1)
        self.stds = torch.tensor(stds, dtype=torch.float32).view(1, 1, -1)
        self.ramp_limits = torch.tensor(ramp_limits, dtype=torch.float32).view(1, 1, -1)

    def forward(self, pred, true):
        loss_mse = self.mse(pred, true)
        device = pred.device
        means = self.means.to(device)
        stds = self.stds.to(device)
        ramp_limits = self.ramp_limits.to(device)

        pred_phys = pred * stds + means
        bvr_penalty = torch.mean(torch.relu(-pred_phys) ** 2)
        delta_p = torch.abs(pred_phys[:, 1:, :] - pred_phys[:, :-1, :])
        rvr_penalty = torch.mean(torch.relu(delta_p - ramp_limits) ** 2)
        return loss_mse + self.lambda_bvr * bvr_penalty + self.lambda_rvr * rvr_penalty


class BaselineExperiment(ForecastExperiment):
    def __init__(self, args):
        super().__init__(args)
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model_dict = {
            'Informer': Informer,
            'Autoformer': Autoformer,
            'PINN': PINN,
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'LSTM': LSTM,
            'GRU': GRU,
        }

        if self.args.model not in model_dict:
            raise ValueError(f"Model {self.args.model} not implemented")

        if self.args.model in ['Autoformer', 'Informer', 'iTransformer']:
            if not hasattr(self.args, 'moving_avg'):
                self.args.moving_avg = 5
            if not hasattr(self.args, 'output_attention'):
                self.args.output_attention = False
            if not hasattr(self.args, 'd_layers'):
                self.args.d_layers = 2
            if not hasattr(self.args, 'distil'):
                self.args.distil = False

        model = model_dict[self.args.model](self.args).float()
        if getattr(self.args, 'use_multi_gpu', False) and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=getattr(self.args, 'device_ids', [self.args.gpu]))

        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        wd = self.args.weight_decay if self.args.weight_decay > 0 else 1e-4
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=wd)

    def _select_criterion(self):
        if self.args.model == 'PINN':
            train_data, _ = self._get_data(flag='train')
            means = train_data.scaler.mean_[:self.args.c_out]
            stds = train_data.scaler.scale_[:self.args.c_out]
            ramp_limits = self._estimate_ramp_limits(train_data)
            self.logger.info(f"PINN soft-constraint ramp limits: {ramp_limits}")
            return PINN_SoftConstraintLoss(means, stds, ramp_limits, lambda_bvr=1.0, lambda_rvr=1.0)
        return nn.MSELoss()

    def _estimate_ramp_limits(self, dataset):
        try:
            raw_groups = []
            for group_tensor in dataset.group_x_tensors:
                group_np = group_tensor.cpu().numpy()
                raw_groups.append(dataset.inverse_transform(group_np))
            if raw_groups:
                raw = np.concatenate(raw_groups, axis=0)
                target_data = raw[:, :self.args.c_out]
                diff = np.abs(target_data[1:] - target_data[:-1])
                return np.percentile(diff, 99.9, axis=0) * 1.5
        except Exception as exc:
            self.logger.warning(f"Failed to estimate ramp limits from data: {exc}")
        return np.ones(self.args.c_out, dtype=np.float32)

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        return batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp

    def _forward_model(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        if self.args.model in ['Informer', 'Autoformer', 'iTransformer']:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if getattr(self.args, 'output_attention', False):
                outputs = outputs[0]
            return outputs
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._process_one_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, :self.args.c_out]
                batch_y = batch_y[:, -self.args.pred_len:, :self.args.c_out]
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        self.model.train()
        return float(np.average(total_loss)) if total_loss else 0.0

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')
        _, test_loader = self._get_data(flag='test')

        criterion = self._select_criterion()
        optimizer = self._select_optimizer()
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.train_epochs, eta_min=1e-6)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, logger=self.logger)
        use_amp = getattr(self.args, 'use_amp', False) and self.args.use_gpu
        scaler = GradScaler(enabled=use_amp)

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            with tqdm(total=len(train_loader),
                      desc=f"Epoch {epoch + 1}/{self.args.train_epochs}",
                      mininterval=0.3,
                      leave=False,
                      ncols=100) as pbar:
                for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                    optimizer.zero_grad()
                    batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._process_one_batch(
                        batch_x, batch_y, batch_x_mark, batch_y_mark
                    )

                    with autocast(enabled=use_amp):
                        outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = outputs[:, -self.args.pred_len:, :self.args.c_out]
                        batch_y = batch_y[:, -self.args.pred_len:, :self.args.c_out]
                        loss = criterion(outputs, batch_y)

                    train_loss.append(loss.item())

                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    pbar.update(1)

            train_loss = float(np.average(train_loss)) if train_loss else 0.0
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)
            scheduler.step()

            self.logger.info(
                f"Epoch: {epoch + 1} | Cost: {time.time() - epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}"
            )

            early_stopping(
                vali_loss,
                self.model,
                str(self.run_dir),
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
            )
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

        self.model.load_state_dict(torch.load(self.checkpoint_path(), map_location=self.device))
        return self.model

    def test(self, load=True, return_preds=False):
        test_data, test_loader = self._get_data(flag='test')
        if load:
            self.logger.info('loading model')
            self.model.load_state_dict(torch.load(self.checkpoint_path(), map_location=self.device))

        self.model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._process_one_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, :self.args.c_out]
                batch_y = batch_y[:, -self.args.pred_len:, :self.args.c_out]
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.array(preds).reshape(-1, self.args.pred_len, self.args.c_out)
        trues = np.array(trues).reshape(-1, self.args.pred_len, self.args.c_out)

        if test_data.scale and test_data.scaler is not None:
            preds = test_data.inverse_transform(preds)
            trues = test_data.inverse_transform(trues)

        ramp_limits = self._estimate_ramp_limits(test_data)
        mae, mse, rmse, bvr, rvr = metric(preds, trues, ramp_limits=ramp_limits)

        per_channel_mse = {}
        target_cols = getattr(test_data, 'target_cols', [f'target_{idx}' for idx in range(self.args.c_out)])
        for idx, col in enumerate(target_cols):
            per_channel_mse[col] = float(np.mean((preds[:, :, idx] - trues[:, :, idx]) ** 2))

        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'bvr_percent': float(bvr),
            'rvr_percent': float(rvr),
            'per_channel_mse': per_channel_mse,
            'legacy_array': [float(mae), float(mse), float(rmse), float(bvr), float(rvr)],
        }
        self.save_test_outputs(preds, trues, metrics)

        self.logger.info(f"MSE:{mse:.6f}, MAE:{mae:.6f}")
        self.logger.info(f'Physics Violation -> BVR:{bvr:.2f}%, RVR:{rvr:.2f}%')

        if return_preds:
            return preds, trues, metrics
        return preds, trues


Exp_Baselines = BaselineExperiment
