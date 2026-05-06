import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from tqdm import tqdm

from .base import EarlyStopping, ForecastExperiment
from ..data.data_factory import data_provider
from ..models import PhysFormer
from ..utils.losses import PhysAwareBaseLoss, PhysLoss
from ..utils.metrics import compute_forecast_metrics, per_channel_mae

warnings.filterwarnings("ignore")


class Exp_PhysFormer(ForecastExperiment):

    def __init__(self, args):
        super().__init__(args)
        self.train_dataset = None
        self.scaler_params = None
        self.training_stats = None
        self.battery_meta = None
        self.model = self._build_model().to(self.device)
        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.criterion = self._select_criterion().to(self.device)

    def _ensure_train_dataset(self):
        if self.train_dataset is None:
            self.train_dataset, _ = self._get_data("train")
            self.scaler_params = self.train_dataset.get_scaler_params()
            self.training_stats = self.train_dataset.get_training_statistics()
            self.battery_meta = self.train_dataset.get_battery_metadata()

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _build_model(self):
        self._ensure_train_dataset()
        self.logger.info(f"PhysFormer scaler params loaded: {list(self.scaler_params.keys())}")
        self.logger.info(
            "Training statistics | net_ramp_limit=%.6f"
            % self.training_stats.get("net_ramp_limit", 0.0)
        )
        return PhysFormer(
            enc_in=self.args.enc_in,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            factor=self.args.factor,
            d_model=getattr(self.args, "d_model", 256),
            n_heads=getattr(self.args, "n_heads", 8),
            e_layers=getattr(self.args, "e_layers", 2),
            d_ff=getattr(self.args, "d_ff", 512),
            dropout=self.args.dropout,
            attn=getattr(self.args, "attn", "full"),
            embed=getattr(self.args, "embed", "custom"),
            freq=getattr(self.args, "freq", "t"),
            activation=getattr(self.args, "activation", "gelu"),
            use_rope=getattr(self.args, "use_rope", True),
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
            no_phys_stream=bool(getattr(self.args, "ablation_no_phys_stream", False)),
            no_battery_branch=bool(getattr(self.args, "ablation_no_battery_branch", False)),
            no_soc_consistency=bool(getattr(self.args, "ablation_no_soc_consistency", False)),
            no_future_weather=bool(getattr(self.args, "ablation_no_future_weather", False)),
            battery_meta=self.battery_meta,
            use_temporal_decoder=(
                bool(getattr(self.args, "use_temporal_decoder", True))
                and not bool(getattr(self.args, "ablation_no_temporal_decoder", False))
            ),
            film_scale=getattr(self.args, "film_scale", 0.5),
            num_portfolios=len(self.scaler_params.get("per_portfolio", {})),
            time_feat_dim=getattr(self.args, "time_feat_dim", 8),
        ).float()

    def _select_optimizer(self):
        wd = self.args.weight_decay if self.args.weight_decay > 0 else 1e-4
        if not self.trainable_parameters:
            raise ValueError("No trainable parameters available for optimizer.")
        return optim.AdamW(self.trainable_parameters, lr=self.args.learning_rate, weight_decay=wd)

    def _select_criterion(self):
        self._ensure_train_dataset()
        base_loss = PhysAwareBaseLoss(
            target_mean=self.scaler_params["target_mean"],
            target_std=self.scaler_params["target_std"],
            aux_mean=self.scaler_params["aux_mean"],
            aux_std=self.scaler_params["aux_std"],
            state_mean=self.scaler_params["state_mean"],
            state_std=self.scaler_params["state_std"],
            dt_hours=0.25,
            battery_meta=self.battery_meta,
        )
        return PhysLoss(
            base_loss_module=base_loss,
            soc_weight=float(getattr(self.args, "soc_weight", 0.1)),
            no_soc_consistency=bool(getattr(self.args, "ablation_no_soc_consistency", False)),
            no_battery_physics_loss=bool(getattr(self.args, "ablation_no_battery_physics_loss", False)),
            component_loss_weight=float(getattr(self.args, "component_loss_weight", 0.05)),
            res_reg_weight=float(getattr(self.args, "res_reg_weight", 0.01)),
            phase_1_cw=float(getattr(self.args, "phase_1_cw", 0.1)),
            phase_1_rr=float(getattr(self.args, "phase_1_rr", 0.05)),
            phase_2_cw=float(getattr(self.args, "phase_2_cw", 0.05)),
            phase_2_rr=float(getattr(self.args, "phase_2_rr", 0.01)),
        )

    def _move_batch(self, batch_data):
        moved = []
        for tensor in batch_data:
            if tensor.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                moved.append(tensor.to(self.device, non_blocking=True))
            else:
                moved.append(tensor.float().to(self.device, non_blocking=True))
        return moved

    def _process_one_batch(self, batch_data, collect_debug=False):
        (
            x_net_hist, x_weather_hist, x_battery_hist,
            x_weather_future, y_target, y_aux,
            x_mark_enc, y_mark, portfolio_ids,
        ) = self._move_batch(batch_data)

        outputs = self.model(
            x_net_hist=x_net_hist,
            x_weather_hist=x_weather_hist,
            x_battery_hist=x_battery_hist,
            x_weather_future=x_weather_future,
            x_mark_enc=x_mark_enc,
            y_mark=y_mark,
            portfolio_ids=portfolio_ids,
        )
        batch_context = self.criterion.base_loss.build_batch_context(x_battery_hist)
        loss, debug, terms = self.criterion(
            outputs, y_target, batch_context, y_aux=y_aux, collect_debug=collect_debug,
        )
        return {
            "loss": loss,
            "debug": debug,
            "terms": terms,
            "outputs": outputs,
            "y_target": y_target,
            "y_aux": y_aux,
            "x_net_hist": x_net_hist,
            "batch_context": batch_context,
        }

    def vali(self, vali_loader):
        self.model.eval()
        loss_total = None
        net_mse_total = None
        soc_loss_total = None
        steps = 0
        with torch.no_grad():
            for batch_data in vali_loader:
                result = self._process_one_batch(batch_data, collect_debug=False)
                loss_total = result["loss"].detach() if loss_total is None else loss_total + result["loss"].detach()
                net_mse_total = (
                    result["terms"]["net_mse"].detach()
                    if net_mse_total is None
                    else net_mse_total + result["terms"]["net_mse"].detach()
                )
                soc_loss = result["terms"].get("soc_bounds_loss", result["loss"].new_tensor(0.0)).detach()
                soc_loss_total = soc_loss if soc_loss_total is None else soc_loss_total + soc_loss
                steps += 1
        self.model.train()
        return {
            "loss": float((loss_total / max(steps, 1)).detach().cpu()) if steps else 0.0,
            "net_mse": float((net_mse_total / max(steps, 1)).detach().cpu()) if steps else 0.0,
            "soc_loss": float((soc_loss_total / max(steps, 1)).detach().cpu()) if steps else 0.0,
        }

    def train(self):
        _, train_loader = self._get_data("train")
        _, vali_loader = self._get_data("val")

        optimizer = self._select_optimizer()
        warmup_epochs = max(int(getattr(self.args, "warmup_epochs", 0)), 0)
        warmup_start_factor = float(getattr(self.args, "warmup_start_factor", 0.2))
        if warmup_epochs > 0 and self.args.train_epochs > warmup_epochs:
            warmup_scheduler = LinearLR(optimizer, start_factor=warmup_start_factor,
                                        end_factor=1.0, total_iters=warmup_epochs)
            restart_t0 = int(getattr(self.args, "restart_t0", 15))
            restart_t_mult = int(getattr(self.args, "restart_t_mult", 1))
            restart_scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=restart_t0, T_mult=restart_t_mult, eta_min=1e-6,
            )
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, restart_scheduler],
                                     milestones=[warmup_epochs])
        elif warmup_epochs > 0:
            scheduler = LinearLR(optimizer, start_factor=warmup_start_factor,
                                 end_factor=1.0, total_iters=max(self.args.train_epochs, 1))
        else:
            restart_t0 = int(getattr(self.args, "restart_t0", 15))
            restart_t_mult = int(getattr(self.args, "restart_t_mult", 1))
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=restart_t0, T_mult=restart_t_mult, eta_min=1e-6,
            )

        early_stop_metric = str(getattr(self.args, "early_stop_metric", "net_mse")).lower()
        early_stop_start_epoch = max(int(getattr(self.args, "early_stop_start_epoch", 10)), 1)
        metric_name = "Validation Net MSE" if early_stop_metric == "net_mse" else "Validation loss"
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, logger=self.logger,
            metric_name=metric_name, start_epoch=early_stop_start_epoch,
        )
        use_amp = getattr(self.args, "use_amp", False) and self.args.use_gpu
        scaler = GradScaler(enabled=use_amp)
        log_interval = max(int(getattr(self.args, "log_interval", 50)), 1)

        self.logger.info(
            "Training | samples=%d | steps_per_epoch=%d | batch_size=%d"
            % (len(self.train_dataset), len(train_loader), self.args.batch_size)
        )
        self.logger.info(
            "Optimizer | lr=%.6g | warmup_epochs=%d | early_stop=%s | scheduler=WarmRestarts"
            % (self.args.learning_rate, warmup_epochs, early_stop_metric)
        )

        # Checkpoint resume (from training_state.pth saved by EarlyStopping)
        start_epoch = 0
        state_path = Path(self.run_dir) / "training_state.pth"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            chk = torch.load(self.checkpoint_path(), map_location=self.device)
            missing, unexpected = self.model.load_state_dict(chk, strict=False)
            if missing:
                self.logger.warning("Checkpoint missing keys (new params): %s", missing[:5])
            if unexpected:
                self.logger.warning("Checkpoint unexpected keys (removed params): %s", unexpected[:5])
            optimizer.load_state_dict(state["optimizer"])
            if state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            if state.get("scaler") is not None and use_amp:
                scaler.load_state_dict(state["scaler"])
            start_epoch = state.get("epoch", 0)
            self.logger.info("Resumed from checkpoint | start_epoch=%d", start_epoch)

        phase_1_end = int(getattr(self.args, "phase_1_epochs", 15))
        phase_2_end = int(getattr(self.args, "phase_2_epochs", 40))

        for epoch in range(start_epoch, self.args.train_epochs):
            # Curriculum phase switching
            if epoch < phase_1_end:
                self.criterion.set_phase(1)
            elif epoch < phase_2_end:
                self.criterion.set_phase(2)
            else:
                self.criterion.set_phase(3)

            self.model.train()
            train_loss_sum = None
            steps = 0
            epoch_time = time.time()
            if self.args.use_gpu and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)

            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{self.args.train_epochs}",
                      mininterval=0.3, leave=False, ncols=100) as pbar:
                for step_idx, batch_data in enumerate(train_loader, start=1):
                    optimizer.zero_grad()
                    with autocast(enabled=use_amp):
                        result = self._process_one_batch(batch_data, collect_debug=False)
                        loss = result["loss"]
                    train_loss_sum = loss.detach() if train_loss_sum is None else train_loss_sum + loss.detach()
                    steps += 1
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.trainable_parameters,
                                                       getattr(self.args, "grad_clip", 1.0))
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.trainable_parameters,
                                                       getattr(self.args, "grad_clip", 1.0))
                        optimizer.step()
                    if step_idx % log_interval == 0 or step_idx == len(train_loader):
                        pbar.set_postfix(loss=f"{float((train_loss_sum / max(steps, 1)).detach().cpu()):.4f}")
                    pbar.update(1)

            epoch_seconds = time.time() - epoch_time
            train_loss = float((train_loss_sum / max(steps, 1)).detach().cpu()) if steps else 0.0
            vali_stats = self.vali(vali_loader)
            scheduler.step()
            samples_per_second = (steps * self.args.batch_size) / max(epoch_seconds, 1e-6)
            max_memory_gb = 0.0
            current_lr = optimizer.param_groups[0]["lr"]
            if self.args.use_gpu and torch.cuda.is_available():
                max_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

            self.logger.info(
                "Epoch: %d | Time: %.1fs | Steps: %d | S/s: %.1f | GPU: %.2fGB | LR: %.6g | "
                "Train: %.6f | Val Loss: %.6f | Val MSE: %.6f | Val SOC: %.6f"
                % (epoch+1, epoch_seconds, steps, samples_per_second, max_memory_gb, current_lr,
                   train_loss, vali_stats["loss"], vali_stats["net_mse"], vali_stats["soc_loss"])
            )

            stop_value = vali_stats["net_mse"] if early_stop_metric == "net_mse" else vali_stats["loss"]
            early_stopping(stop_value, self.model, str(self.run_dir),
                           optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch+1)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

        self.model.load_state_dict(torch.load(self.checkpoint_path(), map_location=self.device))
        return self.model

    def _denorm_target_np(self, value):
        mean = np.asarray(self.scaler_params["target_mean"], dtype=np.float32).reshape(1, 1, -1)
        std = np.asarray(self.scaler_params["target_std"], dtype=np.float32).reshape(1, 1, -1)
        return value * std + mean

    def test(self, load=True, return_preds=False):
        _, test_loader = self._get_data("test")
        if load:
            self.logger.info("loading model")
            self.model.load_state_dict(torch.load(self.checkpoint_path(), map_location=self.device))

        self.model.eval()
        preds = []
        trues = []
        theory_nets = []
        residuals = []
        last_hists = []
        y_aux_batches = []
        physics_states_batches = {}

        with torch.no_grad():
            for batch_data in test_loader:
                result = self._process_one_batch(batch_data, collect_debug=False)
                outputs = result["outputs"]
                preds.append(outputs["pred_net"].detach().cpu().numpy())
                trues.append(result["y_target"].detach().cpu().numpy())
                theory_nets.append(outputs["theory_net"].detach().cpu().numpy())
                residuals.append(outputs["residual"].detach().cpu().numpy())
                last_hists.append(result.get("x_net_hist", torch.zeros(1, 1, 1))[:, -1:, :].detach().cpu().numpy())
                y_aux_batches.append(result["y_aux"].detach().cpu().numpy())
                for name, value in outputs["physics_states"].items():
                    if isinstance(value, torch.Tensor):
                        physics_states_batches.setdefault(name, []).append(value.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        theory_nets = np.concatenate(theory_nets, axis=0)
        residuals = np.concatenate(residuals, axis=0)
        last_hists = np.concatenate(last_hists, axis=0)
        physics_states = {n: np.concatenate(v, axis=0) for n, v in physics_states_batches.items()}

        preds_real = self._denorm_target_np(preds)
        trues_real = self._denorm_target_np(trues)
        theory_real = self._denorm_target_np(theory_nets)
        last_hist_real = self._denorm_target_np(last_hists)

        ramp_limits = np.asarray([self.training_stats.get("net_ramp_limit", 0.0)], dtype=np.float32)
        metrics = compute_forecast_metrics(preds_real, trues_real, ramp_limits=ramp_limits,
                                           last_hist=last_hist_real)
        metrics["theory_mae"] = float(np.mean(np.abs(theory_real - trues_real)))
        metrics["theory_rmse"] = float(np.sqrt(np.mean((theory_real - trues_real) ** 2)))
        residual_real = preds_real - theory_real
        metrics["residual_std_real_mw"] = float(np.std(residual_real))
        metrics["residual_mean_real_mw"] = float(np.mean(residual_real))
        metrics["net_ramp_limit"] = float(self.training_stats.get("net_ramp_limit", 0.0))

        # Battery physics diagnostics
        if "battery_soc_theory_real" in physics_states:
            soc = physics_states["battery_soc_theory_real"]
            cap = physics_states.get("battery_capacity_real",
                                     np.ones_like(soc[..., :1]))
            metrics["soc_min"] = float(soc.min())
            metrics["soc_max"] = float(soc.max())
            metrics["soc_bound_violation"] = float(np.mean((soc < 0) | (soc > cap)))

        # Per-component evaluation against aux ground truth
        y_aux_all = np.concatenate(y_aux_batches, axis=0)
        aux_mean_np = np.asarray(self.scaler_params["aux_mean"], dtype=np.float32).reshape(1, 1, -1)
        aux_std_np = np.asarray(self.scaler_params["aux_std"], dtype=np.float32).reshape(1, 1, -1)
        y_aux_real = y_aux_all * aux_std_np + aux_mean_np

        comp_names = ["load", "pv", "wind", "battery_power", "battery_soc"]
        component_theory = physics_states.get("component_theory_real")
        if component_theory is not None and component_theory.shape[-1] >= 5:
            comp_mae_theory = per_channel_mae(
                component_theory[..., :5], y_aux_real[..., :5], comp_names,
            )
            for name, val in comp_mae_theory.items():
                metrics[f"component_{name}_mae"] = val

        diagnostic_summary = {
            "theory_mae": metrics["theory_mae"],
            "residual_std_real_mw": metrics["residual_std_real_mw"],
            "residual_mean_real_mw": metrics["residual_mean_real_mw"],
            "soc_bound_violation": metrics.get("soc_bound_violation", None),
        }

        extras = {
            "theory_net.npy": theory_real,
            "residual_net.npy": residuals,
            "physics_states.npz": physics_states,
            "diagnostic_summary.json": diagnostic_summary,
        }
        self.save_test_outputs(preds_real, trues_real, metrics, extras=extras)

        self.logger.info(
            "MSE: %.6f | MAE: %.6f | RMSE: %.6f | Theory MAE: %.6f | "
            "Residual mean (MW): %.6f | Residual std (MW): %.6f"
            % (metrics["mse"], metrics["mae"], metrics["rmse"], metrics["theory_mae"],
               metrics["residual_mean_real_mw"], metrics["residual_std_real_mw"])
        )

        if return_preds:
            return preds_real, trues_real, metrics
        return preds_real, trues_real
