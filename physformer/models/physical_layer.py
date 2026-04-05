import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplicitVPPPhysicalLayer(nn.Module):
    """
    Explicit gray-box component layer for portfolio-level VPP forecasting.

    Outputs normalized component theory trajectories in the fixed order:
    [load, pv, wind, battery_power, battery_soc]
    """

    def __init__(
        self,
        d_model,
        weather_dim,
        battery_state_dim,
        time_feat_dim,
        weather_mean=None,
        weather_std=None,
        state_mean=None,
        state_std=None,
        target_mean=None,
        target_std=None,
        aux_mean=None,
        aux_std=None,
        dt_hours=0.25,
        no_battery_branch=False,
        load_state_dim=2,
    ):
        super().__init__()
        self.d_model = d_model
        self.weather_dim = weather_dim
        self.battery_state_dim = battery_state_dim
        self.time_feat_dim = time_feat_dim
        self.dt_hours = dt_hours
        self.no_battery_branch = no_battery_branch
        self.load_state_dim = load_state_dim

        self.register_buffer("weather_mean", self._to_buffer(weather_mean, weather_dim, 0.0))
        self.register_buffer("weather_std", self._to_buffer(weather_std, weather_dim, 1.0))
        self.register_buffer("state_mean", self._to_buffer(state_mean, battery_state_dim, 0.0))
        self.register_buffer("state_std", self._to_buffer(state_std, battery_state_dim, 1.0))
        self.register_buffer("target_mean", self._to_buffer(target_mean, 1, 0.0))
        self.register_buffer("target_std", self._to_buffer(target_std, 1, 1.0))
        self.register_buffer("aux_mean", self._to_buffer(aux_mean, 5, 0.0))
        self.register_buffer("aux_std", self._to_buffer(aux_std, 5, 1.0))

        # Load branch: base + asymmetric temperature response + calendar + lightweight linear latent state.
        self.load_base = nn.Parameter(torch.tensor(2.5))
        self.load_heat_sens = nn.Parameter(torch.tensor(0.3))
        self.load_cool_sens = nn.Parameter(torch.tensor(0.25))
        self.load_comfort_low = nn.Parameter(torch.tensor(18.0))
        self.load_comfort_gap = nn.Parameter(torch.tensor(6.0))
        self.load_calendar_gain = nn.Parameter(torch.tensor(0.5))
        self.load_calendar_proj = nn.Sequential(
            nn.Linear(time_feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.load_state_init = nn.Linear(3, load_state_dim)
        self.load_state_input = nn.Linear(time_feat_dim + 2, load_state_dim)
        self.load_state_A = nn.Parameter(torch.eye(load_state_dim) * 0.8)
        self.load_state_out = nn.Linear(load_state_dim, 1)

        # PV branch: explicit irradiance-temperature conversion with learnable scale.
        self.pv_scale = nn.Parameter(torch.tensor(0.5))
        self.pv_temp_coeff = nn.Parameter(torch.tensor(0.2))

        # Wind branch: smooth cut-in / rated / cut-out curve with learnable scale.
        self.wind_scale = nn.Parameter(torch.tensor(-4.0))
        self.wind_cut_in = nn.Parameter(torch.tensor(1.0))
        self.wind_rated_delta = nn.Parameter(torch.tensor(2.0))
        self.wind_cut_out_delta = nn.Parameter(torch.tensor(2.0))

        # Battery branch: split charge/discharge with weak exclusivity through loss only.
        self.battery_limit_scale = nn.Parameter(torch.tensor(-1.0))
        self.battery_capacity_scale = nn.Parameter(torch.tensor(0.0))
        self.eta_charge_raw = nn.Parameter(torch.tensor(2.0))
        self.eta_discharge_raw = nn.Parameter(torch.tensor(2.0))
        self.battery_context_proj = nn.Sequential(
            nn.Linear(5, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.battery_step_proj = nn.Sequential(
            nn.Linear(d_model + weather_dim + time_feat_dim + 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.battery_charge_head = nn.Linear(d_model, 1)
        self.battery_discharge_head = nn.Linear(d_model, 1)
        self.battery_power_persistence = nn.Parameter(torch.tensor(0.5))
        self.battery_mode_persistence = nn.Parameter(torch.tensor(0.25))

    @staticmethod
    def _to_buffer(value, dim, default):
        if value is None:
            return torch.full((dim,), float(default), dtype=torch.float32)
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if tensor.numel() != dim:
            raise ValueError(f"Expected buffer of dim {dim}, got {tensor.numel()}.")
        return tensor

    def _denorm_weather(self, x_weather):
        return x_weather * self.weather_std.view(1, 1, -1) + self.weather_mean.view(1, 1, -1)

    def _denorm_state(self, x_state):
        return x_state * self.state_std.view(1, 1, -1) + self.state_mean.view(1, 1, -1)

    def _denorm_target(self, x_target):
        return x_target * self.target_std.view(1, 1, -1) + self.target_mean.view(1, 1, -1)

    def _norm_aux(self, aux_real):
        return (aux_real - self.aux_mean.view(1, 1, -1)) / (self.aux_std.view(1, 1, -1) + 1e-6)

    def _load_branch(self, temp, y_mark, x_net_hist_real, hist_weather_real):
        heating = F.relu(self.load_comfort_low - temp)
        comfort_high = self.load_comfort_low + F.softplus(self.load_comfort_gap)
        cooling = F.relu(temp - comfort_high)

        calendar_profile = torch.sigmoid(self.load_calendar_proj(y_mark))

        last_net = x_net_hist_real[:, -1, 0:1]
        mean_net = x_net_hist_real.mean(dim=1)
        last_temp = hist_weather_real[:, -1, 0:1]
        init_features = torch.cat([last_net, mean_net, last_temp], dim=-1)
        state = torch.tanh(self.load_state_init(init_features))

        state_seq = []
        A = torch.tanh(self.load_state_A)
        for step in range(y_mark.shape[1]):
            step_input = torch.cat([y_mark[:, step, :], heating[:, step, :], cooling[:, step, :]], dim=-1)
            drive = self.load_state_input(step_input)
            state = torch.tanh(torch.matmul(state, A.transpose(0, 1)) + drive)
            state_seq.append(state)
        load_state = torch.stack(state_seq, dim=1)
        state_term = self.load_state_out(load_state)

        load_pre = (
            self.load_base
            + F.softplus(self.load_heat_sens) * heating
            + F.softplus(self.load_cool_sens) * cooling
            + F.softplus(self.load_calendar_gain) * calendar_profile
            + state_term
        )
        load_theory = F.softplus(load_pre)
        return load_theory, calendar_profile, load_state

    def _battery_branch(self, x_weather_future, y_mark, x_net_hist_real, x_battery_hist_real):
        batch_size, pred_len, _ = x_weather_future.shape
        last_power = x_battery_hist_real[:, -1, 0:1]
        last_soc = x_battery_hist_real[:, -1, 1:2]
        mean_power = x_battery_hist_real[:, :, 0:1].mean(dim=1)
        mean_soc = x_battery_hist_real[:, :, 1:2].mean(dim=1)
        mean_net = x_net_hist_real.mean(dim=1)

        max_abs_power = x_battery_hist_real[:, :, 0].abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
        hist_soc_max = x_battery_hist_real[:, :, 1].amax(dim=1, keepdim=True).clamp_min(1e-4)
        power_limit = max_abs_power * (F.softplus(self.battery_limit_scale) + 0.5)
        capacity = hist_soc_max * (F.softplus(self.battery_capacity_scale) + 0.5)

        eta_charge = 0.80 + 0.19 * torch.sigmoid(self.eta_charge_raw)
        eta_discharge = 0.80 + 0.19 * torch.sigmoid(self.eta_discharge_raw)

        if self.no_battery_branch:
            zeros = torch.zeros(batch_size, pred_len, 1, device=x_weather_future.device, dtype=x_weather_future.dtype)
            battery_soc = last_soc.unsqueeze(1).repeat(1, pred_len, 1)
            return zeros, zeros, zeros, battery_soc, capacity, eta_charge, eta_discharge

        context = torch.cat([last_power, last_soc, mean_power, mean_soc, mean_net], dim=-1)
        base_context = self.battery_context_proj(context)

        charge_prev = F.relu(last_power)
        discharge_prev = F.relu(-last_power)
        soc_prev = last_soc

        charge_seq = []
        discharge_seq = []
        power_seq = []
        soc_seq = []

        for step in range(pred_len):
            step_context = torch.cat(
                [
                    base_context,
                    x_weather_future[:, step, :],
                    y_mark[:, step, :],
                    charge_prev,
                    discharge_prev,
                ],
                dim=-1,
            )
            hidden = self.battery_step_proj(step_context)

            charge_base = torch.sigmoid(self.battery_charge_head(hidden))
            discharge_base = torch.sigmoid(self.battery_discharge_head(hidden))

            charge = power_limit * torch.clamp(
                charge_base + torch.sigmoid(self.battery_power_persistence) * charge_prev / (power_limit + 1e-6),
                min=0.0,
                max=1.5,
            )
            discharge = power_limit * torch.clamp(
                discharge_base + torch.sigmoid(self.battery_mode_persistence) * discharge_prev / (power_limit + 1e-6),
                min=0.0,
                max=1.5,
            )

            power = charge - discharge
            soc_next = soc_prev + (eta_charge * charge - discharge / eta_discharge) * self.dt_hours
            soc_next = torch.minimum(torch.maximum(soc_next, torch.zeros_like(capacity)), capacity)

            charge_seq.append(charge)
            discharge_seq.append(discharge)
            power_seq.append(power)
            soc_seq.append(soc_next)

            charge_prev = charge
            discharge_prev = discharge
            soc_prev = soc_next

        battery_charge = torch.stack(charge_seq, dim=1)
        battery_discharge = torch.stack(discharge_seq, dim=1)
        battery_power = torch.stack(power_seq, dim=1)
        battery_soc = torch.stack(soc_seq, dim=1)
        return battery_charge, battery_discharge, battery_power, battery_soc, capacity, eta_charge, eta_discharge

    def forward(self, x_weather_hist, x_weather_future, y_mark, x_net_hist, x_battery_hist):
        hist_weather_real = self._denorm_weather(x_weather_hist)
        weather_real = self._denorm_weather(x_weather_future)
        x_net_hist_real = self._denorm_target(x_net_hist)
        battery_hist_real = self._denorm_state(x_battery_hist)

        temp = weather_real[..., 0:1]
        solar_energy = (weather_real[..., 1:2].clamp_min(0.0) / 3600000.0).clamp_min(0.0)
        wind = weather_real[..., 2:3].clamp_min(0.0)
        weather_phys = torch.cat([temp, solar_energy, wind], dim=-1)

        load_theory, calendar_profile, load_state = self._load_branch(
            temp=temp,
            y_mark=y_mark,
            x_net_hist_real=x_net_hist_real,
            hist_weather_real=hist_weather_real,
        )

        pv_temp_factor = (1.0 - F.softplus(self.pv_temp_coeff) * 0.01 * torch.relu(temp - 25.0)).clamp_min(0.0)
        pv_theory = F.softplus(self.pv_scale) * solar_energy * pv_temp_factor

        cut_in = F.softplus(self.wind_cut_in)
        rated = cut_in + F.softplus(self.wind_rated_delta)
        cut_out = rated + F.softplus(self.wind_cut_out_delta)
        rising_curve = ((wind - cut_in) / (rated - cut_in + 1e-6)).clamp(0.0, 1.0) ** 3
        running_mask = ((wind >= cut_in) & (wind <= cut_out)).float()
        plateau = ((wind > rated) & (wind <= cut_out)).float()
        wind_curve = plateau + (1.0 - plateau) * rising_curve
        wind_theory = F.softplus(self.wind_scale) * wind_curve * running_mask

        (
            battery_charge,
            battery_discharge,
            battery_power,
            battery_soc,
            battery_capacity,
            eta_charge,
            eta_discharge,
        ) = self._battery_branch(weather_phys, y_mark, x_net_hist_real, battery_hist_real)

        component_real = torch.cat([load_theory, pv_theory, wind_theory, battery_power, battery_soc], dim=-1)
        component_norm = self._norm_aux(component_real)

        states = {
            "component_theory_real": component_real,
            "load_theory_real": load_theory,
            "pv_theory_real": pv_theory,
            "wind_theory_real": wind_theory,
            "battery_power_theory_real": battery_power,
            "battery_soc_theory_real": battery_soc,
            "battery_charge_theory_real": battery_charge,
            "battery_discharge_theory_real": battery_discharge,
            "battery_capacity_real": battery_capacity.unsqueeze(1).expand(-1, component_real.shape[1], -1),
            "battery_eta_charge": torch.ones_like(battery_soc) * eta_charge,
            "battery_eta_discharge": torch.ones_like(battery_soc) * eta_discharge,
            "load_calendar_profile": calendar_profile,
            "load_state": load_state,
        }
        return component_norm, states
