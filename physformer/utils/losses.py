import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysAwareBaseLoss(nn.Module):
    """Computes individual loss *terms* in real physical units.

    The caller (PhysLoss) decides how to weight and combine them.
    """

    def __init__(
        self,
        target_mean,
        target_std,
        aux_mean,
        aux_std,
        state_mean,
        state_std,
        dt_hours=0.25,
        battery_meta=None,
    ):
        super().__init__()
        self.register_buffer("target_mean", self._to_buffer(target_mean, 1, 0.0))
        self.register_buffer("target_std", self._to_buffer(target_std, 1, 1.0))
        self.register_buffer("aux_mean", self._to_buffer(aux_mean, 5, 0.0))
        self.register_buffer("aux_std", self._to_buffer(aux_std, 5, 1.0))
        self.register_buffer("state_mean", self._to_buffer(state_mean, 2, 0.0))
        self.register_buffer("state_std", self._to_buffer(state_std, 2, 1.0))
        self.dt_hours = float(dt_hours)

        if battery_meta and battery_meta.get("E_max") is not None:
            self.register_buffer("_E_max", torch.tensor(battery_meta["E_max"]))
            self._has_E_max = True
        else:
            self._has_E_max = False

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
        if self._has_E_max:
            capacity_real = self._E_max.view(1, 1, 1).expand_as(last_soc_real)
        else:
            capacity_real = soc_hist.amax(dim=1, keepdim=True).clamp_min(1e-6)
        return {"last_soc_real": last_soc_real, "capacity_real": capacity_real}

    def compute_terms(self, pred_dict, y_target, batch_context, y_aux=None, collect_debug=False):
        pred_net = pred_dict["pred_net"]
        theory_net = pred_dict.get("theory_net", torch.zeros_like(pred_net))
        physics_states = pred_dict.get("physics_states", {})

        # -- primary --
        net_mse = F.mse_loss(pred_net, y_target)
        net_mae_norm = F.l1_loss(pred_net, y_target)
        pred_net_real = self.denorm_target(pred_net)
        true_net_real = self.denorm_target(y_target)
        net_mae_real = F.l1_loss(pred_net_real, true_net_real)
        net_mse_real = F.mse_loss(pred_net_real, true_net_real)

        # -- theory diagnostics (not optimised, just tracked) --
        theory_net_real = self.denorm_target(theory_net)
        theory_mae_real = F.l1_loss(theory_net_real, true_net_real)
        residual_net_real = pred_net_real - theory_net_real
        residual_std_real = residual_net_real.std()

        # -- battery physics (from physics_states) --
        last_soc_real = batch_context["last_soc_real"]
        capacity_real = batch_context["capacity_real"]

        soc = physics_states.get("battery_soc_theory_real")

        if soc is not None:
            # Direct penalty on SOC physical bounds violation.
            soc_bounds_loss = F.relu(-soc).mean() + F.relu(soc - capacity_real).mean()
        else:
            soc_bounds_loss = pred_net_real.new_tensor(0.0)

        # -- component supervision (normalized to aux_std space) --
        battery_power_mae = pred_net_real.new_tensor(0.0)
        component_load_mae = pred_net_real.new_tensor(0.0)
        component_pv_mae = pred_net_real.new_tensor(0.0)
        component_wind_mae = pred_net_real.new_tensor(0.0)
        component_theory_real = physics_states.get("component_theory_real")
        if y_aux is not None and component_theory_real is not None:
            aux_std_view = self.aux_std.view(1, 1, -1)
            aux_mean_view = self.aux_mean.view(1, 1, -1)
            theory_comp_norm = (component_theory_real - aux_mean_view) / (aux_std_view + 1e-8)
            component_load_mae = F.l1_loss(theory_comp_norm[..., 0:1], y_aux[..., 0:1])
            component_pv_mae = F.l1_loss(theory_comp_norm[..., 1:2], y_aux[..., 1:2])
            component_wind_mae = F.l1_loss(theory_comp_norm[..., 2:3], y_aux[..., 2:3])
            batt_power_theory = physics_states.get("battery_power_theory_real")
            if batt_power_theory is not None:
                battery_power_mae = F.l1_loss(theory_comp_norm[..., 3:4], y_aux[..., 3:4])

        # -- component theory diagnostics --
        if component_theory_real is not None:
            load_theory_d = component_theory_real[..., 0:1]
            pv_theory_d = component_theory_real[..., 1:2]
            wind_theory_d = component_theory_real[..., 2:3]
            batt_power_theory_d = component_theory_real[..., 3:4]
            batt_soc_theory_d = component_theory_real[..., 4:5]
            theory_components = {
                "load_theory_mean": load_theory_d.mean(),
                "pv_theory_mean": pv_theory_d.mean(),
                "wind_theory_mean": wind_theory_d.mean(),
                "batt_power_theory_mean": batt_power_theory_d.mean(),
                "batt_soc_theory_mean": batt_soc_theory_d.mean(),
            }
        else:
            theory_components = {}

        terms = {
            "net_mse": net_mse,
            "net_mae_norm": net_mae_norm,
            "net_mae_real": net_mae_real,
            "net_mse_real": net_mse_real,
            "theory_mae_real": theory_mae_real,
            "residual_std_real": residual_std_real,
            "soc_bounds_loss": soc_bounds_loss,
            "battery_power_mae": battery_power_mae,
            "component_load_mae": component_load_mae,
            "component_pv_mae": component_pv_mae,
            "component_wind_mae": component_wind_mae,
            "pred_net_real": pred_net_real,
            "true_net_real": true_net_real,
            "theory_net_real": theory_net_real,
        }
        terms.update(theory_components)

        if collect_debug:
            terms["component_theory_real"] = component_theory_real
        return terms


class PhysLoss(nn.Module):
    """Single-stage loss: net MSE + SOC bounds penalty.

    Removed terms (dual-draft audit):
      - soc_transition_loss: redundant with soc_bounds_loss; both penalise
        the same physical violation (SOC out of [0, capacity]).  soc_bounds
        is a direct ReLU penalty that is simpler and more interpretable.
      - anti_overlap_loss: identically zero by construction — charge and
        discharge come from opposite ReLU sides of the same signed power,
        so charge * discharge == 0 always.
    """

    def __init__(
        self,
        base_loss_module,
        soc_weight=0.1,
        no_soc_consistency=False,
        no_battery_physics_loss=False,
        component_loss_weight=0.05,
    ):
        super().__init__()
        self.base_loss = base_loss_module
        self.soc_weight = float(soc_weight)
        self.no_soc_consistency = bool(no_soc_consistency)
        self.no_battery_physics_loss = bool(no_battery_physics_loss)
        self.component_loss_weight = float(component_loss_weight)

    def forward(self, pred_dict, y_target, batch_context, y_aux=None, collect_debug=False):
        terms = self.base_loss.compute_terms(
            pred_dict, y_target, batch_context, y_aux=y_aux, collect_debug=collect_debug,
        )

        total_loss = terms["net_mse"]

        if not self.no_battery_physics_loss:
            if not self.no_soc_consistency:
                total_loss = total_loss + self.soc_weight * terms["soc_bounds_loss"]

        if self.component_loss_weight > 0:
            total_loss = total_loss + self.component_loss_weight * (
                terms["component_load_mae"]
                + terms["component_pv_mae"]
                + terms["component_wind_mae"]
                + terms["battery_power_mae"]
            )

        debug = None
        if collect_debug:
            debug = {
                "total_loss": float(total_loss.detach().cpu()),
                "net_mse": float(terms["net_mse"].detach().cpu()),
                "net_mae_norm": float(terms["net_mae_norm"].detach().cpu()),
                "net_mae_real": float(terms["net_mae_real"].detach().cpu()),
                "net_mse_real": float(terms["net_mse_real"].detach().cpu()),
                "theory_mae_real": float(terms["theory_mae_real"].detach().cpu()),
                "residual_std_real": float(terms["residual_std_real"].detach().cpu()),
                "soc_bounds_loss": float(terms["soc_bounds_loss"].detach().cpu()),
                "battery_power_mae": float(terms["battery_power_mae"].detach().cpu()),
                "component_load_mae": float(terms["component_load_mae"].detach().cpu()),
                "component_pv_mae": float(terms["component_pv_mae"].detach().cpu()),
                "component_wind_mae": float(terms["component_wind_mae"].detach().cpu()),
            }
        return total_loss, debug, terms
