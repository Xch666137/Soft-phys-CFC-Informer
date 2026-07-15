"""Masked Component Pretraining experiment for PhysFormer-iGT Phase B."""

import math
import random
import shutil
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR

from ..data import data_provider
from ..loss import PretrainLoss
from .base import BaseExperiment, EarlyStopping, _unwrap_state_dict, _align_state_dict_keys


class PretrainExperiment(BaseExperiment):
    """Masked Component Pretraining (MCP) experiment.

    Differences from PhysFormerExperiment:
      - Uses train U val data (pretraining_mode=True)
      - Randomly masks 1-2 component history channels per sample
      - PretrainLoss: component MAE (masked) + lambda_net * net MSE
      - No curriculum, single phase
      - Saves pretrained checkpoint for downstream finetuning
    """

    def __init__(self, args):
        super().__init__(args)
        self.scaler_params = None
        self.model = self._build_model().to(self.device)
        if (
            hasattr(torch, "compile")
            and getattr(self.args, "use_compile", True)
            and self.args.use_gpu
        ):
            compile_attrs = [
                "comp_embedding", "weather_embedding",
                "encoder",
            ]
            for attr in compile_attrs:
                sub = getattr(self.model, attr, None)
                if sub is not None:
                    setattr(self.model, attr, torch.compile(sub, mode="default"))
        self.criterion = self._build_criterion().to(self.device)
        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]

    def _get_data(self, flag, pretraining_mode=False):
        return data_provider(self.args, flag, pretraining_mode=pretraining_mode)

    def _build_model(self):
        from ..models import PhysFormeriGT
        dataset, _ = self._get_data("train", pretraining_mode=True)
        self.scaler_params = dataset.get_scaler_params()
        self.logger.info(f"Pretrain scaler params loaded: {list(self.scaler_params.keys())}")

        return PhysFormeriGT(
            enc_in=self.args.enc_in,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            factor=self.args.factor,
            d_model=getattr(self.args, "d_model", 256),
            n_heads=getattr(self.args, "n_heads", 8),
            e_layers=getattr(self.args, "e_layers", 2),
            d_ff=getattr(self.args, "d_ff", 512),
            dropout=self.args.dropout,
            time_feat_dim=getattr(self.args, "time_feat_dim", 8),
            weather_mean=self.scaler_params["weather_mean"],
            weather_std=self.scaler_params["weather_std"],
            state_mean=self.scaler_params["state_mean"],
            state_std=self.scaler_params["state_std"],
            target_mean=self.scaler_params["target_mean"],
            target_std=self.scaler_params["target_std"],
            aux_mean=self.scaler_params["aux_mean"],
            aux_std=self.scaler_params["aux_std"],
            comp_gru_hidden=getattr(self.args, "comp_gru_hidden", None),
        ).float()

    def _build_criterion(self):
        return PretrainLoss(
            lambda_net=float(getattr(self.args, "pretrain_lambda_net", 0.3)),
        )

    def _move_batch(self, batch_data):
        moved = []
        for tensor in batch_data:
            if tensor.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                moved.append(tensor.to(self.device, non_blocking=True))
            else:
                moved.append(tensor.float().to(self.device, non_blocking=True))
        return moved

    def _sample_mask_indices(self, batch_size):
        mask = torch.zeros(batch_size, 4, device=self.device, dtype=torch.bool)
        for row in range(batch_size):
            n_mask = random.choice([1, 2])
            # PV (index 1) has 50% probability of being masked
            if random.random() < 0.5:
                # Force PV to be masked, then choose remaining from others
                remaining = [0, 2, 3]
                extra = random.sample(remaining, n_mask - 1)
                indices = [1] + extra
            else:
                # Original behavior: random selection
                indices = random.sample([0, 1, 2, 3], n_mask)
            mask[row, indices] = True
        return mask

    def _validation_masks(self, batch_size):
        for idx in range(4):
            mask = torch.zeros(batch_size, 4, device=self.device, dtype=torch.bool)
            mask[:, idx] = True
            yield mask

    def _scaler_view(self, scaler_params, name, width, dtype):
        if not scaler_params:
            return None
        value = scaler_params.get(name)
        if value is None:
            return None
        return torch.as_tensor(value, device=self.device, dtype=dtype).view(1, 1, width)

    def _renormalize_tensor(self, tensor, source_scaler_params, source_prefix, target_scaler_params, target_prefix):
        if not source_scaler_params:
            return tensor

        width = tensor.shape[-1]
        source_mean = self._scaler_view(source_scaler_params, f"{source_prefix}_mean", width, tensor.dtype)
        source_std = self._scaler_view(source_scaler_params, f"{source_prefix}_std", width, tensor.dtype)
        target_mean = self._scaler_view(target_scaler_params, f"{target_prefix}_mean", width, tensor.dtype)
        target_std = self._scaler_view(target_scaler_params, f"{target_prefix}_std", width, tensor.dtype)
        if source_mean is None or source_std is None or target_mean is None or target_std is None:
            return tensor

        tensor_real = tensor * source_std + source_mean
        return (tensor_real - target_mean) / (target_std + 1e-6)

    def _renormalize_like_pretrain(self, y_target, y_aux, x_component_hist, x_weather_future, source_scaler_params):
        y_target = self._renormalize_tensor(
            y_target, source_scaler_params, "target", self.scaler_params, "target",
        )
        y_aux = self._renormalize_tensor(
            y_aux, source_scaler_params, "aux", self.scaler_params, "aux",
        )
        x_component_hist = self._renormalize_tensor(
            x_component_hist, source_scaler_params, "aux", self.scaler_params, "aux",
        )
        x_weather_future = self._renormalize_tensor(
            x_weather_future, source_scaler_params, "weather", self.scaler_params, "weather",
        )
        return y_target, y_aux, x_component_hist, x_weather_future

    def _process_one_batch(self, batch_data, collect_debug=False):
        (
            x_net_hist, x_weather_hist, x_battery_hist,
            x_weather_future, y_target, y_aux,
            x_mark_enc, y_mark, portfolio_ids,
            x_component_hist,
        ) = self._move_batch(batch_data)

        mask_indices = self._sample_mask_indices(x_component_hist.shape[0])

        outputs = self.model(
            x_net_hist=x_net_hist,
            x_weather_hist=x_weather_hist,
            x_battery_hist=x_battery_hist,
            x_weather_future=x_weather_future,
            x_mark_enc=x_mark_enc,
            y_mark=y_mark,
            portfolio_ids=portfolio_ids,
            x_component_hist=x_component_hist,
            mask_indices=mask_indices,
        )

        loss, debug, terms = self.criterion(outputs, y_target, y_aux, mask_indices)

        result = {"loss": loss, "debug": debug, "terms": terms, "outputs": outputs}
        if collect_debug:
            result["y_target"] = y_target
            result["y_aux"] = y_aux
            result["mask_indices"] = mask_indices
        return result

    def vali(self, vali_loader, source_scaler_params=None):
        """Validation with deterministic single-component masks.

        Forward inputs and labels are re-normalized into the pretrain scaler space.
        """
        self.model.eval()
        loss_total = 0.0
        net_loss_total = 0.0
        steps = 0
        with torch.no_grad():
            for batch_data in vali_loader:
                (
                    x_net_hist, x_weather_hist, x_battery_hist,
                    x_weather_future, y_target, y_aux,
                    x_mark_enc, y_mark, portfolio_ids,
                    x_component_hist,
                ) = self._move_batch(batch_data)

                (
                    y_target_eval, y_aux_eval,
                    x_component_hist_eval, x_weather_future_eval,
                ) = self._renormalize_like_pretrain(
                    y_target, y_aux, x_component_hist, x_weather_future, source_scaler_params,
                )

                for mask_indices in self._validation_masks(x_component_hist.shape[0]):
                    outputs = self.model(
                        x_net_hist=x_net_hist,
                        x_weather_hist=x_weather_hist,
                        x_battery_hist=x_battery_hist,
                        x_weather_future=x_weather_future_eval,
                        x_mark_enc=x_mark_enc,
                        y_mark=y_mark,
                        portfolio_ids=portfolio_ids,
                        x_component_hist=x_component_hist_eval,
                        mask_indices=mask_indices,
                    )

                    loss, debug, _ = self.criterion(outputs, y_target_eval, y_aux_eval, mask_indices)
                    loss_total += float(loss.detach().cpu())
                    net_loss_total += float(debug["net_loss"])
                    steps += 1

        self.model.train()
        return {
            "loss": loss_total / max(steps, 1) if steps else 0.0,
            "net_loss": net_loss_total / max(steps, 1) if steps else 0.0,
        }

    def train(self):
        train_dataset, train_loader = self._get_data("train", pretraining_mode=True)
        val_dataset, vali_loader = self._get_data("val", pretraining_mode=False)

        optimizer = optim.AdamW(
            self.trainable_parameters,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay if self.args.weight_decay > 0 else 1e-4,
        )

        steps_per_epoch = len(train_loader)
        total_steps = self.args.train_epochs * steps_per_epoch
        scheduler = OneCycleLR(
            optimizer, max_lr=self.args.learning_rate,
            total_steps=total_steps, pct_start=0.12,
            div_factor=25.0, final_div_factor=1e4,
            anneal_strategy='cos',
        )

        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, logger=self.logger,
            metric_name="Validation loss", start_epoch=5,
        )

        use_amp = getattr(self.args, "use_amp", False) and self.args.use_gpu
        scaler = GradScaler(enabled=use_amp)
        log_interval = max(int(getattr(self.args, "log_interval", 50)), 1)
        best_val_net = float("inf")
        best_val_net_path = self.run_dir / "best_val_net_checkpoint.pth"
        save_best_val_net = bool(getattr(self.args, "save_best_val_net_checkpoint", True))

        self.logger.info(
            "Pretraining | samples=%d | steps_per_epoch=%d | batch_size=%d"
            % (len(train_dataset), len(train_loader), self.args.batch_size)
        )
        self.logger.info(
            "PretrainLoss | lambda_net=%.2f | mask=per-sample 1-2 of [load,pv,wind,batt_p]"
            % getattr(self.args, "pretrain_lambda_net", 0.3)
        )
        val_scaler_params = val_dataset.get_scaler_params()
        self.logger.info(
            "Pretrain validation | deterministic single-component masks | tensors re-normalized to pretrain scaler"
        )

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss_sum = 0.0
            steps = 0
            epoch_time = time.time()
            if self.args.use_gpu and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)

            for step_idx, batch_data in enumerate(train_loader, start=1):
                optimizer.zero_grad()
                with autocast(enabled=use_amp):
                    result = self._process_one_batch(batch_data)
                    loss = result["loss"]
                train_loss_sum += float(loss.detach().cpu())
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
                    avg_loss = train_loss_sum / steps
                    self.logger.info(
                        "Epoch %d/%d [%d/%d] loss=%.4f" % (
                            epoch + 1, self.args.train_epochs, step_idx,
                            len(train_loader), avg_loss,
                        )
                    )

            epoch_seconds = time.time() - epoch_time
            train_loss = train_loss_sum / max(steps, 1)

            vali_stats = self.vali(vali_loader, source_scaler_params=val_scaler_params)
            samples_per_second = (steps * self.args.batch_size) / max(epoch_seconds, 1e-6)
            current_lr = optimizer.param_groups[0]["lr"]

            self.logger.info(
                "Epoch: %d | Time: %.1fs | S/s: %.1f | LR: %.6g | "
                "Train: %.6f | Val Loss: %.6f | Val Net: %.6f"
                % (epoch + 1, epoch_seconds, samples_per_second, current_lr,
                   train_loss, vali_stats["loss"], vali_stats["net_loss"])
            )

            if save_best_val_net and math.isfinite(vali_stats["net_loss"]) and vali_stats["net_loss"] < best_val_net:
                self.logger.info(
                    "Val Net improved (%.6f --> %.6f). Saving net-best checkpoint ...",
                    best_val_net, vali_stats["net_loss"],
                )
                best_val_net = vali_stats["net_loss"]
                torch.save(_unwrap_state_dict(self.model), best_val_net_path)

            early_stopping(vali_stats["loss"], self.model, str(self.run_dir),
                           optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                           epoch=epoch + 1)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

        # Load best checkpoint — prefer best_val_net if available
        if save_best_val_net and best_val_net_path.exists():
            state_dict = torch.load(best_val_net_path, map_location=self.device)
            state_dict = _align_state_dict_keys(state_dict, self.model.state_dict())
            self.model.load_state_dict(state_dict)
            self.logger.info("Loaded best_val_net checkpoint for pretrained export")
        else:
            state_dict = torch.load(self.checkpoint_path(), map_location=self.device)
            state_dict = _align_state_dict_keys(state_dict, self.model.state_dict())
            self.model.load_state_dict(state_dict)
            self.logger.info("Loaded early-stopping checkpoint for pretrained export")

        # Save pretrained checkpoint for finetuning
        pretrained_path = self.run_dir / "pretrained_checkpoint.pth"
        torch.save(_unwrap_state_dict(self.model), pretrained_path)
        self.logger.info(f"Pretrained checkpoint saved to: {pretrained_path}")

        export_dir = Path.cwd() / "results" / self.run_name
        export_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pretrained_path, export_dir / "pretrained_checkpoint.pth")
        self.logger.info("Exported pretrained checkpoint to: %s", export_dir / "pretrained_checkpoint.pth")

        if save_best_val_net:
            if best_val_net_path.exists():
                shutil.copy2(best_val_net_path, export_dir / "best_val_net_checkpoint.pth")
                self.logger.info(f"Best Val Net checkpoint available at: {best_val_net_path}")
                self.logger.info(
                    "Exported best Val Net checkpoint to: %s",
                    export_dir / "best_val_net_checkpoint.pth",
                )
            else:
                raise RuntimeError(
                    "best_val_net_checkpoint.pth was NOT saved during training. "
                    "Check if vali() returned valid net_loss values."
                )

        return self.model
