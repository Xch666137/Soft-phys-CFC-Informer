import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysAwareBaseLoss(nn.Module):
    COMPONENT_NAMES = ("load", "pv", "wind", "battery_power", "battery_soc")

    def __init__(
        self,
        target_mean,
        target_std,
        aux_mean,
        aux_std,
        state_mean,
        state_std,
        net_ramp_limit=0.0,
        battery_ramp_limit=0.0,
        dt_hours=0.25,
        component_weights=None,
    ):
        super().__init__()
        self.register_buffer("target_mean", self._to_buffer(target_mean, 1, 0.0))
        self.register_buffer("target_std", self._to_buffer(target_std, 1, 1.0))
        self.register_buffer("aux_mean", self._to_buffer(aux_mean, 5, 0.0))
        self.register_buffer("aux_std", self._to_buffer(aux_std, 5, 1.0))
        self.register_buffer("state_mean", self._to_buffer(state_mean, 2, 0.0))
        self.register_buffer("state_std", self._to_buffer(state_std, 2, 1.0))

        weights = component_weights or [1.0, 1.0, 1.0, 1.0, 0.5]
        self.register_buffer("component_weights", torch.as_tensor(weights, dtype=torch.float32).view(1, 1, -1))
        self.net_ramp_limit = float(net_ramp_limit)
        self.battery_ramp_limit = float(battery_ramp_limit)
        self.dt_hours = float(dt_hours)

    @staticmethod
    def _to_buffer(value, dim, default):
        if value is None:
            return torch.full((dim,), float(default), dtype=torch.float32)
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if tensor.numel() != dim:
            raise ValueError(f"Expected {dim} values, got {tensor.numel()}.")
        return tensor

    def denorm_target(self, value):
        return value * self.target_std.view(1, 1, -1) + self.target_mean.view(1, 1, -1)

    def denorm_aux(self, value):
        return value * self.aux_std.view(1, 1, -1) + self.aux_mean.view(1, 1, -1)

    def denorm_state(self, value):
        return value * self.state_std.view(1, 1, -1) + self.state_mean.view(1, 1, -1)

    def build_batch_context(self, x_battery_hist):
        battery_hist_real = self.denorm_state(x_battery_hist)
        soc_hist = battery_hist_real[..., 1:2]
        last_soc_real = soc_hist[:, -1:, :]
        capacity_real = soc_hist.amax(dim=1, keepdim=True).clamp_min(1e-6)
        return {
            "last_soc_real": last_soc_real,
            "capacity_real": capacity_real,
        }

    def compute_terms(self, pred_dict, y_target, y_aux, batch_context):
        pred_net = pred_dict["pred_net"]
        pred_aux = pred_dict["pred_aux"]

        net_mse = F.mse_loss(pred_net, y_target)
        net_mae_norm = F.l1_loss(pred_net, y_target)

        aux_error = (pred_aux - y_aux) ** 2
        weighted_aux = aux_error * self.component_weights
        component_loss = weighted_aux[..., :3].mean()
        battery_state_loss = weighted_aux[..., 3:].mean()

        pred_net_real = self.denorm_target(pred_net)
        true_net_real = self.denorm_target(y_target)
        pred_aux_real = self.denorm_aux(pred_aux)
        true_aux_real = self.denorm_aux(y_aux)

        pred_battery_power = pred_aux_real[..., 3:4]
        pred_battery_soc = pred_aux_real[..., 4:5]
        true_battery_power = true_aux_real[..., 3:4]
        true_battery_soc = true_aux_real[..., 4:5]

        pred_charge = pred_dict["pred_charge_real"]
        pred_discharge = pred_dict["pred_discharge_real"]
        eta_charge = pred_dict["battery_eta_charge"]
        eta_discharge = pred_dict["battery_eta_discharge"]

        if pred_net_real.shape[1] > 1:
            net_diff = pred_net_real[:, 1:, :] - pred_net_real[:, :-1, :]
            battery_diff = pred_battery_power[:, 1:, :] - pred_battery_power[:, :-1, :]
            net_ramp_penalty = F.relu(torch.abs(net_diff) - self.net_ramp_limit).mean() if self.net_ramp_limit > 0 else torch.mean(torch.abs(net_diff))
            battery_ramp_penalty = (
                F.relu(torch.abs(battery_diff) - self.battery_ramp_limit).mean()
                if self.battery_ramp_limit > 0
                else torch.mean(torch.abs(battery_diff))
            )
            battery_smoothness = torch.mean(torch.abs(battery_diff))
        else:
            zero = pred_net_real.new_tensor(0.0)
            net_ramp_penalty = zero
            battery_ramp_penalty = zero
            battery_smoothness = zero

        last_soc_real = batch_context["last_soc_real"]
        capacity_real = batch_context["capacity_real"]
        implied_soc = torch.cumsum((eta_charge * pred_charge - pred_discharge / eta_discharge) * self.dt_hours, dim=1) + last_soc_real
        soc_transition_loss = F.l1_loss(pred_battery_soc, implied_soc)
        soc_bounds_loss = F.relu(-pred_battery_soc).mean() + F.relu(pred_battery_soc - capacity_real).mean()
        anti_overlap_loss = torch.mean(pred_charge * pred_discharge)

        battery_power_mae = F.l1_loss(pred_battery_power, true_battery_power)
        battery_soc_mae = F.l1_loss(pred_battery_soc, true_battery_soc)

        component_mae = {
            name: float(F.l1_loss(pred_aux_real[..., idx], true_aux_real[..., idx]).detach().cpu())
            for idx, name in enumerate(self.COMPONENT_NAMES)
        }

        net_mae_real = F.l1_loss(pred_net_real, true_net_real)
        net_mse_real = F.mse_loss(pred_net_real, true_net_real)

        return {
            "net_mse": net_mse,
            "net_mae_norm": net_mae_norm,
            "net_mae_real": net_mae_real,
            "net_mse_real": net_mse_real,
            "component_loss": component_loss,
            "battery_state_loss": battery_state_loss,
            "net_ramp_penalty": net_ramp_penalty,
            "battery_ramp_penalty": battery_ramp_penalty,
            "battery_smoothness": battery_smoothness,
            "soc_transition_loss": soc_transition_loss,
            "soc_bounds_loss": soc_bounds_loss,
            "anti_overlap_loss": anti_overlap_loss,
            "battery_power_mae": battery_power_mae,
            "battery_soc_mae": battery_soc_mae,
            "component_mae": component_mae,
            "pred_net_real": pred_net_real,
            "true_net_real": true_net_real,
            "pred_aux_real": pred_aux_real,
            "true_aux_real": true_aux_real,
        }


class PhysLoss(nn.Module):
    def __init__(
        self,
        base_loss_module,
        total_epochs,
        aux_weight=0.4,
        battery_weight=0.4,
        ramp_weight=0.1,
        soc_weight=0.2,
        overlap_weight=0.02,
        no_aux_supervision=False,
        no_soc_consistency=False,
    ):
        super().__init__()
        self.base_loss = base_loss_module
        self.total_epochs = max(int(total_epochs), 1)
        self.aux_weight = float(aux_weight)
        self.battery_weight = float(battery_weight)
        self.ramp_weight = float(ramp_weight)
        self.soc_weight = float(soc_weight)
        self.overlap_weight = float(overlap_weight)
        self.no_aux_supervision = bool(no_aux_supervision)
        self.no_soc_consistency = bool(no_soc_consistency)

    def _curriculum_weights(self, epoch):
        if epoch is None:
            return {"aux": 1.0, "physics": 1.0}

        progress = max(float(epoch), 0.0) / float(self.total_epochs)
        aux = min(max((progress - 0.10) / 0.30, 0.0), 1.0)
        physics = min(max((progress - 0.25) / 0.40, 0.0), 1.0)
        return {"aux": aux, "physics": physics}

    def forward(self, pred_dict, y_target, y_aux, batch_context, epoch=None):
        terms = self.base_loss.compute_terms(pred_dict, y_target, y_aux, batch_context)
        curriculum = self._curriculum_weights(epoch)

        total_loss = terms["net_mse"]
        if not self.no_aux_supervision:
            total_loss = total_loss + curriculum["aux"] * (
                self.aux_weight * terms["component_loss"]
                + self.battery_weight * terms["battery_state_loss"]
            )

        total_loss = total_loss + curriculum["physics"] * self.ramp_weight * (
            terms["net_ramp_penalty"]
            + terms["battery_ramp_penalty"]
            + 0.5 * terms["battery_smoothness"]
        )

        total_loss = total_loss + curriculum["physics"] * self.overlap_weight * terms["anti_overlap_loss"]

        if not self.no_soc_consistency:
            total_loss = total_loss + curriculum["physics"] * self.soc_weight * (
                terms["soc_transition_loss"] + terms["soc_bounds_loss"]
            )

        debug = {
            "total_loss": float(total_loss.detach().cpu()),
            "net_mse": float(terms["net_mse"].detach().cpu()),
            "net_mae_norm": float(terms["net_mae_norm"].detach().cpu()),
            "net_mae_real": float(terms["net_mae_real"].detach().cpu()),
            "net_mse_real": float(terms["net_mse_real"].detach().cpu()),
            "component_loss": float(terms["component_loss"].detach().cpu()),
            "battery_state_loss": float(terms["battery_state_loss"].detach().cpu()),
            "net_ramp_penalty": float(terms["net_ramp_penalty"].detach().cpu()),
            "battery_ramp_penalty": float(terms["battery_ramp_penalty"].detach().cpu()),
            "battery_smoothness": float(terms["battery_smoothness"].detach().cpu()),
            "soc_transition_loss": float(terms["soc_transition_loss"].detach().cpu()),
            "soc_bounds_loss": float(terms["soc_bounds_loss"].detach().cpu()),
            "anti_overlap_loss": float(terms["anti_overlap_loss"].detach().cpu()),
            "battery_power_mae": float(terms["battery_power_mae"].detach().cpu()),
            "battery_soc_mae": float(terms["battery_soc_mae"].detach().cpu()),
            "curriculum_aux": float(curriculum["aux"]),
            "curriculum_physics": float(curriculum["physics"]),
            "component_mae": terms["component_mae"],
        }
        return total_loss, debug, terms
