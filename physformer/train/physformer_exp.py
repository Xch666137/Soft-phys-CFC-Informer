import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR

from ..data import data_provider
from ..loss import PhysAwareBaseLoss, PhysLoss
from ..metrics import compute_forecast_metrics, per_channel_mae
from ..models import PhysFormer, PhysFormeriGT
from .base import BaseExperiment, EarlyStopping


class PhysFormerExperiment(BaseExperiment):

    def __init__(self, args):
        super().__init__(args)
        self.train_dataset = None
        self.scaler_params = None
        self.training_stats = None
        self.battery_meta = None
        self.model = self._build_model().to(self.device)
        if (
            hasattr(torch, "compile")
            and getattr(self.args, "use_compile", True)
            and self.args.use_gpu
        ):
            compile_attrs = [
                "encoder", "temporal_decoder", "weather_fusion",
                "physics_film", "unified_head", "flatten_head",
                "comp_embeddings", "weather_embeddings", "component_projectors",
            ]
            for attr in compile_attrs:
                sub = getattr(self.model, attr, None)
                if sub is not None:
                    setattr(self.model, attr, torch.compile(sub, mode="default"))
        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.criterion = self._select_criterion().to(self.device)
        self._last_vali_stats = None
        self._last_grad_info = None

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
        model_cls = PhysFormeriGT if self.args.model == "PhysFormer-iGT" else PhysFormer
        return model_cls(
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
            no_deep_battery_context=bool(getattr(self.args, "no_deep_battery_context", False)),
            battery_meta=self.battery_meta,
            use_temporal_decoder=(
                bool(getattr(self.args, "use_temporal_decoder", True))
                and not bool(getattr(self.args, "ablation_no_temporal_decoder", False))
            ),
            film_scale=getattr(self.args, "film_scale", 0.5),
            num_portfolios=len(self.scaler_params.get("per_portfolio", {})),
            time_feat_dim=getattr(self.args, "time_feat_dim", 8),
            load_gru_hidden=getattr(self.args, "load_gru_hidden", 96),
            load_gru_use_temp=bool(getattr(self.args, "load_gru_use_temp", True)),
            load_temp_model=str(getattr(self.args, "load_temp_model", "mlp")),
            detach_scale=float(getattr(self.args, "detach_scale", 0.0)),
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
            theory_loss_weight=float(getattr(self.args, "theory_loss_weight", 0.1)),
            phase_1_cw=float(getattr(self.args, "phase_1_cw", 0.1)),
            phase_1_rr=float(getattr(self.args, "phase_1_rr", 0.05)),
            phase_2_cw=float(getattr(self.args, "phase_2_cw", 0.05)),
            phase_2_rr=float(getattr(self.args, "phase_2_rr", 0.01)),
            phase_1_tw=float(getattr(self.args, "phase_1_tw", 0.2)),
            phase_2_tw=float(getattr(self.args, "phase_2_tw", 0.1)),
            battery_component_weight=float(getattr(self.args, "battery_component_weight", 1.0)),
        )

    def _move_batch(self, batch_data):
        moved = []
        for tensor in batch_data:
            if tensor.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                moved.append(tensor.to(self.device, non_blocking=True))
            else:
                moved.append(tensor.float().to(self.device, non_blocking=True))
        return moved

    def _process_one_batch(self, batch_data, collect_debug=False, compute_loss=True):
        (
            x_net_hist, x_weather_hist, x_battery_hist,
            x_weather_future, y_target, y_aux,
            x_mark_enc, y_mark, portfolio_ids,
            x_component_hist,
        ) = self._move_batch(batch_data)

        # Extract load history from component history (column 0)
        x_load_hist = x_component_hist[..., 0:1]

        outputs = self.model(
            x_net_hist=x_net_hist,
            x_weather_hist=x_weather_hist,
            x_battery_hist=x_battery_hist,
            x_weather_future=x_weather_future,
            x_mark_enc=x_mark_enc,
            y_mark=y_mark,
            portfolio_ids=portfolio_ids,
            x_load_hist=x_load_hist,
            x_component_hist=x_component_hist,
        )
        if compute_loss:
            batch_context = self.criterion.base_loss.build_batch_context(x_battery_hist)
            loss, debug, terms = self.criterion(
                outputs, y_target, batch_context, y_aux=y_aux, collect_debug=collect_debug,
            )
            return {
                "loss": loss, "debug": debug, "terms": terms, "outputs": outputs,
                "y_target": y_target, "y_aux": y_aux, "x_net_hist": x_net_hist,
                "batch_context": batch_context,
            }
        return {
            "outputs": outputs, "y_target": y_target, "y_aux": y_aux,
            "x_net_hist": x_net_hist,
        }

    def _compute_gradient_angle(self, train_loader, use_amp=False):
        if not hasattr(self.model, 'phys_layer'):
            return {"cos_sim": 0.0, "angle_deg": 0.0,
                    "norm_net": 0.0, "norm_theory": 0.0}
        # torch.compile donated-buffer optimisation (default in mode="default")
        # is incompatible with retain_graph=True, which we need here.
        use_compile = getattr(self.args, "use_compile", True)
        if use_compile:
            return {"cos_sim": 0.0, "angle_deg": 0.0,
                    "norm_net": 0.0, "norm_theory": 0.0}
        # Keep model in train mode: cuDNN RNN (used by LoadTemporalModule GRU)
        # does not support backward() in eval mode.
        batch_data = next(iter(train_loader))
        result = self._process_one_batch(batch_data, collect_debug=True)
        terms = result["terms"]
        net_mse = terms["net_mse"]
        theory_mse = terms["theory_mse"]
        bc_proj = self.model.phys_layer.battery_context_proj
        target_param = bc_proj.input_proj.weight if hasattr(bc_proj, 'input_proj') else bc_proj[0].weight
        self.model.zero_grad()
        net_mse.backward(retain_graph=True)
        grad_net = target_param.grad.clone()
        self.model.zero_grad()
        theory_mse.backward(retain_graph=True)
        grad_theory = target_param.grad.clone()
        self.model.zero_grad()
        dot = (grad_net * grad_theory).sum()
        norm_net = grad_net.norm()
        norm_theory = grad_theory.norm()
        cos_sim = float((dot / (norm_net * norm_theory + 1e-8)).cpu())
        angle_deg = float(torch.acos(torch.clamp(torch.tensor(cos_sim), -1, 1)).cpu() * 180 / 3.14159)
        return {
            "cos_sim": cos_sim, "angle_deg": angle_deg,
            "norm_net": float(norm_net.cpu()), "norm_theory": float(norm_theory.cpu()),
        }

    def vali(self, vali_loader):
        self.model.eval()
        loss_total = None
        net_mse_total = None
        net_mse_real_total = None
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
                mse_real = result["terms"].get("net_mse_real", result["loss"].new_tensor(0.0)).detach()
                net_mse_real_total = mse_real if net_mse_real_total is None else net_mse_real_total + mse_real
                soc_loss = result["terms"].get("soc_bounds_loss", result["loss"].new_tensor(0.0)).detach()
                soc_loss_total = soc_loss if soc_loss_total is None else soc_loss_total + soc_loss
                steps += 1
        self.model.train()
        return {
            "loss": float((loss_total / max(steps, 1)).detach().cpu()) if steps else 0.0,
            "net_mse": float((net_mse_total / max(steps, 1)).detach().cpu()) if steps else 0.0,
            "net_mse_real": float((net_mse_real_total / max(steps, 1)).detach().cpu()) if steps else 0.0,
            "soc_loss": float((soc_loss_total / max(steps, 1)).detach().cpu()) if steps else 0.0,
        }

    def _log_phase_transition(self, phase_label, phase_reset_mode, optimizer):
        if phase_reset_mode == "hard":
            for state in optimizer.state.values():
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()
            for pg in optimizer.param_groups:
                pg["lr"] = self.args.learning_rate
            self.logger.info("Phase %s (hard): optimizer state reset, LR=%.2e",
                             phase_label, self.args.learning_rate)
        else:
            self.logger.info("Phase %s (soft): preserving Adam momentum, cw=%.3f",
                             phase_label, self.criterion.component_loss_weight)

    def train(self):
        _, train_loader = self._get_data("train")
        _, vali_loader = self._get_data("val")

        optimizer = self._select_optimizer()
        warmup_epochs = max(int(getattr(self.args, "warmup_epochs", 0)), 0)
        steps_per_epoch = len(train_loader)
        total_steps = self.args.train_epochs * steps_per_epoch
        scheduler = OneCycleLR(
            optimizer, max_lr=self.args.learning_rate,
            total_steps=total_steps, pct_start=0.12,
            div_factor=25.0, final_div_factor=1e4,
            anneal_strategy='cos',
        )

        early_stop_metric = str(getattr(self.args, "early_stop_metric", "net_mse")).lower()
        early_stop_start_epoch = max(int(getattr(self.args, "early_stop_start_epoch", 5)), 1)
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
            "Optimizer | lr=%.6g | warmup_epochs=%d | early_stop=%s | scheduler=OneCycle"
            % (self.args.learning_rate, warmup_epochs, early_stop_metric)
        )

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
            scaler_state = state.get("scaler")
            if scaler_state is not None and len(scaler_state) > 0 and use_amp:
                scaler.load_state_dict(scaler_state)
            elif scaler_state is not None and len(scaler_state) == 0:
                self.logger.warning("Old GradScaler state is empty (was disabled), skipping load")
            start_epoch = state.get("epoch", 0)
            self.logger.info("Resumed from checkpoint | start_epoch=%d", start_epoch)

        phase_1_end = int(getattr(self.args, "phase_1_epochs", 15))
        phase_2a_end = int(getattr(self.args, "phase_2a_epochs", phase_1_end))
        phase_2_end = int(getattr(self.args, "phase_2_epochs", 40))
        phase_2_initialized = start_epoch >= phase_1_end
        phase_2a_initialized = start_epoch >= phase_2a_end
        phase_reset_mode = str(getattr(self.args, "phase_reset_mode", "soft"))

        for epoch in range(start_epoch, self.args.train_epochs):
            if epoch < phase_1_end:
                self.criterion.set_phase(1)
                self.model.set_detach_mode("none")
            elif epoch < phase_2a_end:
                self.criterion.set_phase("2a")
                self.model.set_detach_mode("none")
                if not phase_2_initialized:
                    self._log_phase_transition("2a", phase_reset_mode, optimizer)
                    phase_2_initialized = True
            elif epoch < phase_2_end:
                self.criterion.set_phase(2)
                detach_mode = str(getattr(self.args, "detach_mode_phase2", "none"))
                self.model.set_detach_mode(detach_mode)
                if not phase_2a_initialized:
                    self._log_phase_transition("2b (detach=%s)" % detach_mode, phase_reset_mode, optimizer)
                    phase_2a_initialized = True
            else:
                self.criterion.set_phase(3)
                self.model.set_detach_mode("none")

            self.model.train()
            train_loss_sum = None
            steps = 0
            epoch_time = time.time()
            if self.args.use_gpu and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)

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
                scheduler.step()
                if step_idx % log_interval == 0 or step_idx == len(train_loader):
                    avg_loss = float((train_loss_sum / steps).detach().cpu())
                    self.logger.info(
                        "Epoch %d/%d [%d/%d] loss=%.4f" % (
                            epoch+1, self.args.train_epochs, step_idx, len(train_loader), avg_loss,
                        )
                    )

            epoch_seconds = time.time() - epoch_time
            train_loss = float((train_loss_sum / max(steps, 1)).detach().cpu()) if steps else 0.0

            val_interval = int(getattr(self.args, "val_interval", 1))
            do_val = (epoch + 1) % val_interval == 0 or epoch == start_epoch
            if do_val:
                vali_stats = self.vali(vali_loader)
                self._last_vali_stats = vali_stats
            else:
                vali_stats = self._last_vali_stats

            grad_angle_interval = int(getattr(self.args, "grad_angle_interval", 1))
            do_grad = (epoch + 1) % grad_angle_interval == 0 or epoch == start_epoch
            if do_grad:
                grad_info = self._compute_gradient_angle(train_loader, use_amp=use_amp)
                self._last_grad_info = grad_info
            else:
                grad_info = self._last_grad_info or {"cos_sim": 0.0, "angle_deg": 0.0,
                                                      "norm_net": 0.0, "norm_theory": 0.0}

            samples_per_second = (steps * self.args.batch_size) / max(epoch_seconds, 1e-6)
            max_memory_gb = 0.0
            current_lr = optimizer.param_groups[0]["lr"]
            if self.args.use_gpu and torch.cuda.is_available():
                max_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

            val_loss_str = "%.6f" % vali_stats["loss"] if vali_stats else "---"
            val_mse_str = "%.6f" % vali_stats["net_mse"] if vali_stats else "---"
            val_mser_str = "%.3e" % vali_stats["net_mse_real"] if vali_stats else "---"
            val_soc_str = "%.6f" % vali_stats["soc_loss"] if vali_stats else "---"
            self.logger.info(
                "Epoch: %d | Time: %.1fs | Steps: %d | S/s: %.1f | GPU: %.2fGB | LR: %.6g | "
                "Train: %.6f | Val Loss: %s | Val MSE: %s | Val MSE(MW²): %s | Val SOC: %s | "
                "GradCos: %.3f | GradAngle: %.1f° | GradNorm(Net): %.2e | GradNorm(Theory): %.2e"
                % (epoch+1, epoch_seconds, steps, samples_per_second, max_memory_gb, current_lr,
                   train_loss, val_loss_str, val_mse_str, val_mser_str, val_soc_str,
                   grad_info["cos_sim"], grad_info["angle_deg"],
                   grad_info["norm_net"], grad_info["norm_theory"])
            )

            if do_val and vali_stats is not None:
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
        loss_terms_sum = {}
        loss_batches = 0

        with torch.no_grad():
            for batch_data in test_loader:
                result = self._process_one_batch(batch_data, collect_debug=False, compute_loss=True)
                outputs = result["outputs"]
                preds.append(outputs["pred_net"].detach())
                trues.append(result["y_target"].detach())
                theory_nets.append(outputs["theory_net"].detach())
                residuals.append(outputs["residual"].detach())
                x_hist = result.get("x_net_hist", torch.zeros(1, 1, 1, device=self.device))
                last_hists.append(x_hist[:, -1:, :].detach())
                y_aux_batches.append(result["y_aux"].detach())
                for name, value in outputs["physics_states"].items():
                    if isinstance(value, torch.Tensor):
                        physics_states_batches.setdefault(name, []).append(value.detach())
                terms = result.get("terms", {})
                for k, v in terms.items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        loss_terms_sum[k] = loss_terms_sum.get(k, 0.0) + float(v.detach().cpu())
                loss_batches += 1

        preds = torch.cat(preds, dim=0).cpu().numpy()
        trues = torch.cat(trues, dim=0).cpu().numpy()
        theory_nets = torch.cat(theory_nets, dim=0).cpu().numpy()
        residuals = torch.cat(residuals, dim=0).cpu().numpy()
        last_hists = torch.cat(last_hists, dim=0).cpu().numpy()
        physics_states = {n: torch.cat(v, dim=0).cpu().numpy() for n, v in physics_states_batches.items()}

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

        if "battery_soc_theory_real" in physics_states:
            soc = physics_states["battery_soc_theory_real"]
            cap = physics_states.get("battery_capacity_real", np.ones_like(soc[..., :1]))
            metrics["soc_min"] = float(soc.min())
            metrics["soc_max"] = float(soc.max())
            metrics["soc_bound_violation"] = float(np.mean((soc < 0) | (soc > cap)))

        y_aux_all = torch.cat(y_aux_batches, dim=0).cpu().numpy()
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
            "test_loss_terms": {k: v / max(loss_batches, 1) for k, v in loss_terms_sum.items()},
        }

        extras = {
            "theory_net.npy": theory_real,
            "residual_net.npy": residuals,
            "physics_states.npz": physics_states,
            "diagnostic_summary.json": diagnostic_summary,
            "test_loss_terms.json": {k: v / max(loss_batches, 1) for k, v in loss_terms_sum.items()},
        }
        self.save_test_outputs(preds_real, trues_real, metrics, extras=extras)

        metrics["_units"] = {"mse": "MW²", "mae": "MW", "rmse": "MW", "theory_mae": "MW",
                             "theory_rmse": "MW", "residual_std_real_mw": "MW",
                             "residual_mean_real_mw": "MW"}

        self.logger.info(
            "Test (MW²/MW) | MSE: %.3e | MAE: %.6f | RMSE: %.6f | Theory MAE: %.6f | "
            "Residual mean: %.6f | Residual std: %.6f"
            % (metrics["mse"], metrics["mae"], metrics["rmse"], metrics["theory_mae"],
               metrics["residual_mean_real_mw"], metrics["residual_std_real_mw"])
        )

        if return_preds:
            return preds_real, trues_real, metrics
        return preds_real, trues_real
