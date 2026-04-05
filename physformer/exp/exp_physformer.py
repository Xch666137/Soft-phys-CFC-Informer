import time
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .base import EarlyStopping, ForecastExperiment
from ..data.data_factory import data_provider
from ..models import PhysFormer
from ..utils.losses import PhysAwareBaseLoss, PhysLoss
from ..utils.metrics import compute_forecast_metrics, per_channel_mae

warnings.filterwarnings("ignore")


class Exp_PhysFormer(ForecastExperiment):
    AUX_NAMES = ["load", "pv", "wind", "battery_power", "battery_soc"]

    def __init__(self, args):
        super().__init__(args)
        self.train_dataset = None
        self.scaler_params = None
        self.training_stats = None
        self.model = self._build_model().to(self.device)
        self.criterion = self._select_criterion().to(self.device)

    def _ensure_train_dataset(self):
        if self.train_dataset is None:
            self.train_dataset, _ = self._get_data("train")
            self.scaler_params = self.train_dataset.get_scaler_params()
            self.training_stats = self.train_dataset.get_training_statistics()

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _build_model(self):
        self._ensure_train_dataset()
        self.logger.info(f"PhysFormer scaler params loaded: {list(self.scaler_params.keys())}")
        self.logger.info(
            "Training statistics | net_ramp_limit=%.6f | battery_ramp_limit=%.6f"
            % (
                self.training_stats.get("net_ramp_limit", 0.0),
                self.training_stats.get("battery_ramp_limit", 0.0),
            )
        )

        return PhysFormer(
            enc_in=self.args.enc_in,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            factor=self.args.factor,
            d_model=self.args.d_model,
            n_heads=self.args.n_heads,
            e_layers=self.args.e_layers,
            d_ff=self.args.d_ff,
            dropout=self.args.dropout,
            attn=self.args.attn,
            embed=self.args.embed,
            freq=self.args.freq,
            activation=self.args.activation,
            use_rope=getattr(self.args, "use_rope", False),
            rope_base=getattr(self.args, "rope_base", 10000),
            distil=getattr(self.args, "distil", False),
            weather_mean=self.scaler_params["weather_mean"],
            weather_std=self.scaler_params["weather_std"],
            state_mean=self.scaler_params["state_mean"],
            state_std=self.scaler_params["state_std"],
            target_mean=self.scaler_params["target_mean"],
            target_std=self.scaler_params["target_std"],
            aux_mean=self.scaler_params["aux_mean"],
            aux_std=self.scaler_params["aux_std"],
            no_phys_stream=getattr(self.args, "ablation_no_phys_stream", False),
            no_battery_branch=getattr(self.args, "ablation_no_battery_branch", False),
            no_aux_supervision=getattr(self.args, "ablation_no_aux_supervision", False),
            no_soc_consistency=getattr(self.args, "ablation_no_soc_consistency", False),
            no_future_weather=getattr(self.args, "ablation_no_future_weather", False),
            shared_query_only=getattr(self.args, "ablation_shared_query_only", False),
        ).float()

    def _select_optimizer(self):
        wd = self.args.weight_decay if self.args.weight_decay > 0 else 1e-4
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=wd)

    def _select_criterion(self):
        self._ensure_train_dataset()
        base_loss = PhysAwareBaseLoss(
            target_mean=self.scaler_params["target_mean"],
            target_std=self.scaler_params["target_std"],
            aux_mean=self.scaler_params["aux_mean"],
            aux_std=self.scaler_params["aux_std"],
            state_mean=self.scaler_params["state_mean"],
            state_std=self.scaler_params["state_std"],
            net_ramp_limit=self.training_stats.get("net_ramp_limit", 0.0),
            battery_ramp_limit=self.training_stats.get("battery_ramp_limit", 0.0),
            dt_hours=0.25,
        )
        return PhysLoss(
            base_loss_module=base_loss,
            total_epochs=self.args.train_epochs,
            no_aux_supervision=getattr(self.args, "ablation_no_aux_supervision", False),
            no_soc_consistency=getattr(self.args, "ablation_no_soc_consistency", False),
        )

    def _move_batch(self, batch_data):
        return [tensor.float().to(self.device, non_blocking=True) for tensor in batch_data]

    def _process_one_batch(self, batch_data, epoch=None, collect_debug=False):
        (
            x_net_hist,
            x_weather_hist,
            x_battery_hist,
            x_weather_future,
            y_target,
            y_aux,
            x_mark_enc,
            y_mark,
        ) = self._move_batch(batch_data)

        outputs = self.model(
            x_net_hist=x_net_hist,
            x_weather_hist=x_weather_hist,
            x_battery_hist=x_battery_hist,
            x_weather_future=x_weather_future,
            x_mark_enc=x_mark_enc,
            y_mark=y_mark,
        )
        batch_context = self.criterion.base_loss.build_batch_context(x_battery_hist)
        loss, debug, terms = self.criterion(
            outputs,
            y_target,
            y_aux,
            batch_context,
            epoch=epoch,
            collect_debug=collect_debug,
        )
        return {
            "loss": loss,
            "debug": debug,
            "terms": terms,
            "outputs": outputs,
            "y_target": y_target,
            "y_aux": y_aux,
            "batch_context": batch_context,
        }

    def vali(self, vali_loader, epoch=None):
        self.model.eval()
        loss_total = None
        net_mse_total = None
        steps = 0
        with torch.no_grad():
            for batch_data in vali_loader:
                result = self._process_one_batch(batch_data, epoch=epoch, collect_debug=False)
                loss_total = result["loss"].detach() if loss_total is None else loss_total + result["loss"].detach()
                net_mse_total = (
                    result["terms"]["net_mse"].detach()
                    if net_mse_total is None
                    else net_mse_total + result["terms"]["net_mse"].detach()
                )
                steps += 1
        self.model.train()
        return {
            "loss": float((loss_total / max(steps, 1)).detach().cpu()) if steps else 0.0,
            "net_mse": float((net_mse_total / max(steps, 1)).detach().cpu()) if steps else 0.0,
        }

    def train(self):
        train_data, train_loader = self._get_data("train")
        _, vali_loader = self._get_data("val")

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
                mininterval=0.3,
                leave=False,
                ncols=100,
            ) as pbar:
                for step_idx, batch_data in enumerate(train_loader, start=1):
                    optimizer.zero_grad()
                    with autocast(enabled=use_amp):
                        result = self._process_one_batch(batch_data, epoch=epoch, collect_debug=False)
                        loss = result["loss"]

                    train_loss_sum = loss.detach() if train_loss_sum is None else train_loss_sum + loss.detach()
                    steps += 1
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.args, "grad_clip", 1.0))
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.args, "grad_clip", 1.0))
                        optimizer.step()

                    if step_idx % log_interval == 0 or step_idx == len(train_loader):
                        avg_loss = float((train_loss_sum / max(steps, 1)).detach().cpu())
                        pbar.set_postfix(loss=f"{avg_loss:.4f}")
                    pbar.update(1)

            epoch_seconds = time.time() - epoch_time
            train_loss = float((train_loss_sum / max(steps, 1)).detach().cpu()) if steps else 0.0
            vali_stats = self.vali(vali_loader, epoch=epoch)
            scheduler.step()
            samples_per_second = (steps * self.args.batch_size) / max(epoch_seconds, 1e-6)
            max_memory_gb = 0.0
            if self.args.use_gpu and torch.cuda.is_available():
                max_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

            self.logger.info(
                "Epoch: %d | Cost: %.2fs | Steps: %d | Samples/s: %.2f | Max GPU Mem: %.2f GB | "
                "Train Loss: %.7f | Vali Loss: %.7f | Vali Net MSE: %.7f"
                % (
                    epoch + 1,
                    epoch_seconds,
                    steps,
                    samples_per_second,
                    max_memory_gb,
                    train_loss,
                    vali_stats["loss"],
                    vali_stats["net_mse"],
                )
            )

            early_stopping(
                vali_stats["loss"],
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

    def _denorm_target_np(self, value):
        mean = np.asarray(self.scaler_params["target_mean"], dtype=np.float32).reshape(1, 1, -1)
        std = np.asarray(self.scaler_params["target_std"], dtype=np.float32).reshape(1, 1, -1)
        return value * std + mean

    def _denorm_aux_np(self, value):
        mean = np.asarray(self.scaler_params["aux_mean"], dtype=np.float32).reshape(1, 1, -1)
        std = np.asarray(self.scaler_params["aux_std"], dtype=np.float32).reshape(1, 1, -1)
        return value * std + mean

    def test(self, load=True, return_preds=False):
        _, test_loader = self._get_data("test")
        if load:
            self.logger.info("loading model")
            self.model.load_state_dict(torch.load(self.checkpoint_path(), map_location=self.device))

        self.model.eval()
        preds = []
        trues = []
        aux_preds = []
        aux_trues = []
        last_soc = []
        physics_states = {}
        charge_preds = []
        discharge_preds = []
        eta_charge = []
        eta_discharge = []

        with torch.no_grad():
            for batch_data in test_loader:
                result = self._process_one_batch(batch_data, epoch=self.args.train_epochs - 1, collect_debug=False)
                outputs = result["outputs"]
                preds.append(outputs["pred_net"].detach().cpu().numpy())
                trues.append(result["y_target"].detach().cpu().numpy())
                aux_preds.append(outputs["pred_aux"].detach().cpu().numpy())
                aux_trues.append(result["y_aux"].detach().cpu().numpy())
                last_soc.append(result["batch_context"]["last_soc_real"].detach().cpu().numpy())
                charge_preds.append(outputs["pred_charge_real"].detach().cpu().numpy())
                discharge_preds.append(outputs["pred_discharge_real"].detach().cpu().numpy())
                eta_charge.append(outputs["battery_eta_charge"].detach().cpu().numpy())
                eta_discharge.append(outputs["battery_eta_discharge"].detach().cpu().numpy())

                for name, value in outputs["physics_states"].items():
                    physics_states.setdefault(name, []).append(value.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        aux_preds = np.concatenate(aux_preds, axis=0)
        aux_trues = np.concatenate(aux_trues, axis=0)
        last_soc = np.concatenate(last_soc, axis=0)
        charge_preds = np.concatenate(charge_preds, axis=0)
        discharge_preds = np.concatenate(discharge_preds, axis=0)
        eta_charge = np.concatenate(eta_charge, axis=0)
        eta_discharge = np.concatenate(eta_discharge, axis=0)
        physics_states = {name: np.concatenate(values, axis=0) for name, values in physics_states.items()}

        preds_real = self._denorm_target_np(preds)
        trues_real = self._denorm_target_np(trues)
        aux_preds_real = self._denorm_aux_np(aux_preds)
        aux_trues_real = self._denorm_aux_np(aux_trues)

        ramp_limits = np.asarray([self.training_stats.get("net_ramp_limit", 0.0)], dtype=np.float32)
        metrics = compute_forecast_metrics(preds_real, trues_real, ramp_limits=ramp_limits)
        metrics["battery_power_mae"] = float(np.mean(np.abs(aux_preds_real[..., 3] - aux_trues_real[..., 3])))
        metrics["battery_soc_mae"] = float(np.mean(np.abs(aux_preds_real[..., 4] - aux_trues_real[..., 4])))
        implied_soc = np.cumsum((eta_charge * charge_preds - discharge_preds / eta_discharge) * self.criterion.base_loss.dt_hours, axis=1) + last_soc
        metrics["soc_consistency_loss"] = float(np.mean(np.abs(aux_preds_real[..., 4:5] - implied_soc)))
        metrics["component_mae"] = per_channel_mae(aux_preds_real, aux_trues_real, self.AUX_NAMES)
        metrics["net_ramp_limit"] = float(self.training_stats.get("net_ramp_limit", 0.0))
        metrics["battery_ramp_limit"] = float(self.training_stats.get("battery_ramp_limit", 0.0))
        metrics["anti_overlap"] = float(np.mean(charge_preds * discharge_preds))

        extras = {
            "component_preds.npz": {name: aux_preds_real[..., idx] for idx, name in enumerate(self.AUX_NAMES)},
            "component_trues.npz": {name: aux_trues_real[..., idx] for idx, name in enumerate(self.AUX_NAMES)},
            "battery_state_preds.npz": {
                "battery_power": aux_preds_real[..., 3],
                "battery_soc": aux_preds_real[..., 4],
                "battery_charge": charge_preds[..., 0],
                "battery_discharge": discharge_preds[..., 0],
                "last_soc": last_soc[..., 0],
            },
            "physics_states.npz": physics_states,
        }
        self.save_test_outputs(preds_real, trues_real, metrics, extras=extras)

        self.logger.info(
            "MSE: %.6f, MAE: %.6f, RMSE: %.6f, Ramp: %.2f%%"
            % (metrics["mse"], metrics["mae"], metrics["rmse"], metrics["net_ramp_violation"])
        )
        self.logger.info(
            "Battery Power MAE: %.6f, Battery SOC MAE: %.6f, SOC Consistency: %.6f"
            % (metrics["battery_power_mae"], metrics["battery_soc_mae"], metrics["soc_consistency_loss"])
        )

        if return_preds:
            return preds_real, trues_real, metrics
        return preds_real, trues_real
