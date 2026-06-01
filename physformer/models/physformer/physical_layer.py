import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepBatteryContext(nn.Module):
    """Deep residual MLP for battery context encoding.

    Replaces a shallow 2-layer MLP to prevent early gradient saturation
    (observed: grad norm collapses from ~20 to ~1e-8 within 1 epoch).
    Residual connections preserve an identity gradient path even when
    individual blocks saturate.
    """

    def __init__(self, d_in=5, d_model=256, depth=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_proj(x)
        for blk in self.blocks:
            x = x + blk(x)
        return self.out_norm(x)


class LearnableTempResponse(nn.Module):
    """Nonlinear temperature-to-load mapping via a small MLP.

    Replaces the rigid HDD/CDD piecewise-linear formulation with a
    learnable function that can capture asymmetric, saturating, and
    region-dependent temperature responses (e.g. stronger heating
    sensitivity in cold climates, zero cooling response in
    heating-dominated regions).
    """

    def __init__(self, d_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
        # Near-zero init: starts as identity pass-through (~no temp effect)
        nn.init.normal_(self.net[-1].weight, std=1e-4)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, temp):
        """temp: (B, pred_len, 1) → (B, pred_len, 1) load contribution"""
        return self.net(temp)


class LearnedBalancePoint(nn.Module):
    """Physically-constrained temperature-to-load mapping with learned balance points.

    Models the asymmetric heating/cooling temperature response of a VPP load
    using two learned balance points and sensitivities. Unlike the unconstrained
    Temp MLP (1089 params), this has only 4 learnable parameters and enforces
    a physically meaningful piecewise structure:

        load_temp = α_h * softplus(T_bal_h − T) + α_c * softplus(T − T_bal_c)

    - Below T_bal_h: heating load (linear in degree-heating)
    - Between T_bal_h and T_bal_c: comfort zone (near-zero contribution)
    - Above T_bal_c: cooling load (linear in degree-cooling)

    This is the "middle ground" between fixed HDD/CDD (18°C reference) and
    the fully unconstrained Temp MLP. It preserves physical interpretability
    while letting the data determine the effective balance points and
    sensitivities for this specific VPP.

    Initialization: T_bal_h=18°C, T_bal_c=24°C (standard building defaults),
    sensitivities near zero so the module starts as a weak contributor.
    """

    def __init__(self):
        super().__init__()
        self.T_balance_heat = nn.Parameter(torch.tensor(18.0))
        self.T_balance_cool = nn.Parameter(torch.tensor(24.0))
        self.alpha_heat_raw = nn.Parameter(torch.tensor(-1.0))   # softplus → ~0.31
        self.alpha_cool_raw = nn.Parameter(torch.tensor(-1.0))   # softplus → ~0.31
        self.base_offset = nn.Parameter(torch.tensor(0.0))

    def forward(self, temp):
        """temp: (B, pred_len, 1) → (B, pred_len, 1) load contribution"""
        heat_deg = F.softplus(self.T_balance_heat - temp)
        cool_deg = F.softplus(temp - self.T_balance_cool)
        alpha_h = F.softplus(self.alpha_heat_raw)
        alpha_c = F.softplus(self.alpha_cool_raw)
        return self.base_offset + alpha_h * heat_deg + alpha_c * cool_deg

    def get_params_dict(self):
        """Return learned parameters as a dict for logging/reporting."""
        with torch.no_grad():
            return {
                "T_balance_heat_C": float(self.T_balance_heat.item()),
                "T_balance_cool_C": float(self.T_balance_cool.item()),
                "alpha_heat_MW_per_deg": float(F.softplus(self.alpha_heat_raw).item()),
                "alpha_cool_MW_per_deg": float(F.softplus(self.alpha_cool_raw).item()),
                "base_offset_MW": float(self.base_offset.item()),
            }


class LoadTemporalModule(nn.Module):
    """GRU-based temporal encoder for load behavioral patterns.

    Processes historical net load + temperature + calendar features to
    produce a temporal correction added to the HDD/CDD load estimate.
    Captures sequential dependencies (daily/weekly patterns, thermal
    inertia) that the static HDD/CDD formulation cannot represent.

    Decoder output is initialized near zero — the module starts as an
    identity pass-through and progressively learns temporal corrections.
    """

    def __init__(self, d_gru=96, num_layers=2, dropout=0.1, time_feat_dim=10,
                 use_temp_input=True):
        super().__init__()
        self.use_temp_input = use_temp_input
        input_dim = 1 + (1 if use_temp_input else 0) + time_feat_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=d_gru,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_gru + time_feat_dim, d_gru),
            nn.GELU(),
            nn.Linear(d_gru, 1),
        )
        # Near-zero init: correction starts negligible, grows as needed
        nn.init.normal_(self.decoder[-1].weight, std=1e-4)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x_net_hist, temp_hist, x_mark_enc, y_mark):
        """Produce temporal correction from history.

        Args:
            x_net_hist:  (B, seq_len, 1)   historical net load (real MW)
            temp_hist:   (B, seq_len, 1)   historical temperature (real degC)
            x_mark_enc:  (B, seq_len, C)   historical calendar features
            y_mark:      (B, pred_len, C)  future calendar features

        Returns:
            (B, pred_len, 1) temporal correction in real MW
        """
        if self.use_temp_input:
            hist_input = torch.cat([x_net_hist, temp_hist, x_mark_enc], dim=-1)
        else:
            hist_input = torch.cat([x_net_hist, x_mark_enc], dim=-1)
        _, h_n = self.gru(hist_input)
        context = h_n[-1]  # (B, d_gru)

        B, pred_len, _ = y_mark.shape
        context_exp = context.unsqueeze(1).expand(-1, pred_len, -1)
        dec_input = torch.cat([context_exp, y_mark], dim=-1)
        return self.decoder(dec_input)


class ExplicitVPPPhysicalLayer(nn.Module):
    """Gray-box physical conditioning layer for VPP net-injection forecasting.

    Produces *theory* estimates (first-order physical priors) and
    battery-state features.  The data-driven path uses these as
    conditioning, not as prediction targets.

    Branches
    --------
    Load   — thermal response + calendar profile (no state-space).
    PV     — irradiance × temperature coefficient.
    Wind   — cubic power curve with cut-in / rated / cut-out.
    Battery — signed-power recurrence with guaranteed chg/dis exclusivity.
    """

    # Per-portfolio parameter delta indices into the 16-dim delta vector
    _DX_LOAD_BASE = 0
    _DX_LOAD_HEAT_SENS = 1
    _DX_LOAD_COOL_SENS = 2
    _DX_LOAD_COMFORT_LOW = 3
    _DX_LOAD_COMFORT_GAP = 4
    _DX_LOAD_CALENDAR_GAIN = 5
    _DX_PV_SCALE = 6
    _DX_PV_TEMP_COEFF = 7
    _DX_PV_CAPACITY = 8
    _DX_WIND_SCALE = 9
    _DX_WIND_CUT_IN = 10
    _DX_WIND_RATED_DELTA = 11
    _DX_WIND_CUT_OUT_DELTA = 12
    _DX_WIND_CAPACITY = 13
    _DX_BATT_LIMIT_SCALE = 14
    _DX_BATT_CAPACITY_SCALE = 15

    def __init__(
        self,
        d_model,
        weather_dim=3,
        battery_state_dim=2,
        time_feat_dim=8,
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
        no_deep_battery_context=False,
        battery_meta=None,
        num_portfolios=0,
        per_portfolio_dim=32,
        load_gru_hidden=96,
        load_gru_use_temp=True,
        load_temp_model="mlp",
    ):
        super().__init__()
        self.d_model = d_model
        self.weather_dim = weather_dim
        self.battery_state_dim = battery_state_dim
        self.time_feat_dim = time_feat_dim
        self.dt_hours = dt_hours
        self.no_battery_branch = no_battery_branch
        self.no_deep_battery_context = no_deep_battery_context
        self.num_portfolios = int(num_portfolios)

        if battery_meta and battery_meta.get("P_max") is not None:
            self.register_buffer("_P_max", torch.tensor(battery_meta["P_max"]))
            self.register_buffer("_E_max", torch.tensor(battery_meta["E_max"]))
        else:
            self.register_buffer("_P_max", torch.tensor(-1.0))
            self.register_buffer("_E_max", torch.tensor(-1.0))

        self.register_buffer("weather_mean", self._to_buffer(weather_mean, weather_dim, 0.0))
        self.register_buffer("weather_std", self._to_buffer(weather_std, weather_dim, 1.0))
        self.register_buffer("state_mean", self._to_buffer(state_mean, battery_state_dim, 0.0))
        self.register_buffer("state_std", self._to_buffer(state_std, battery_state_dim, 1.0))
        self.register_buffer("target_mean", self._to_buffer(target_mean, 1, 0.0))
        self.register_buffer("target_std", self._to_buffer(target_std, 1, 1.0))
        self.register_buffer("aux_mean", self._to_buffer(aux_mean, 5, 0.0))
        self.register_buffer("aux_std", self._to_buffer(aux_std, 5, 1.0))

        # ---- Load branch: behavioral temporal module ----
        if load_temp_model == "balance_point":
            self.temp_response = LearnedBalancePoint()
        else:
            self.temp_response = LearnableTempResponse(d_hidden=32)
        self.load_temp_model = load_temp_model
        # Legacy HDD/CDD params kept as inert checkpoints; not used in forward.
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
        self.load_autoreg_gain = nn.Parameter(torch.tensor(0.1))
        self.load_autoreg_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.load_temporal = LoadTemporalModule(
            d_gru=load_gru_hidden, num_layers=2, dropout=0.1, time_feat_dim=time_feat_dim,
            use_temp_input=load_gru_use_temp,
        )

        # ---- PV branch ----
        self.pv_scale = nn.Parameter(torch.tensor(2.0))  # larger init for stronger gradient signal
        self.pv_temp_coeff = nn.Parameter(torch.tensor(0.2))
        self.pv_capacity = nn.Parameter(torch.tensor(0.0))

        # ---- Wind branch ----
        self.wind_scale = nn.Parameter(torch.tensor(-4.0))
        self.wind_cut_in = nn.Parameter(torch.tensor(1.0))
        self.wind_rated_delta = nn.Parameter(torch.tensor(2.0))
        self.wind_cut_out_delta = nn.Parameter(torch.tensor(2.0))
        self.wind_capacity = nn.Parameter(torch.tensor(0.0))

        # ---- Battery branch: signed-power parameterisation ----
        if no_deep_battery_context:
            self.battery_context_proj = nn.Sequential(
                nn.Linear(5, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.battery_context_proj = DeepBatteryContext(
                d_in=5, d_model=d_model, depth=2, dropout=0.1,
            )
        self.battery_step_proj = nn.Sequential(
            nn.Linear(d_model + weather_dim + time_feat_dim + 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.battery_power_head = nn.Linear(d_model, 1)
        self.battery_limit_scale = nn.Parameter(torch.tensor(-1.0))
        self.battery_capacity_scale = nn.Parameter(torch.tensor(0.0))
        self.eta_charge_raw = nn.Parameter(torch.tensor(2.0))
        self.eta_discharge_raw = nn.Parameter(torch.tensor(2.0))

        # ---- Per-portfolio parameter deltas ----
        if self.num_portfolios > 0:
            self.portfolio_embed = nn.Embedding(self.num_portfolios, per_portfolio_dim)
            self.portfolio_delta = nn.Linear(per_portfolio_dim, 16)
            nn.init.normal_(self.portfolio_embed.weight, std=0.05)
            nn.init.normal_(self.portfolio_delta.weight, std=0.1)
            nn.init.zeros_(self.portfolio_delta.bias)
        else:
            self.portfolio_embed = None
            self.portfolio_delta = None

    @staticmethod
    def _to_buffer(value, dim, default):
        if value is None:
            return torch.full((dim,), float(default), dtype=torch.float32)
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if tensor.numel() != dim:
            raise ValueError(f"Expected buffer of dim {dim}, got {tensor.numel()}.")
        return tensor

    def _get_portfolio_delta(self, portfolio_ids):
        """Return per-portfolio parameter deltas shape (B, 16)."""
        if self.num_portfolios == 0 or portfolio_ids is None:
            return None
        emb = self.portfolio_embed(portfolio_ids)  # (B, per_portfolio_dim)
        return self.portfolio_delta(emb)            # (B, 16)

    # ------------------------------------------------------------------
    # Denorm helpers (internal — operate in real physical units)
    # ------------------------------------------------------------------
    def _denorm_weather(self, x):
        return x * self.weather_std.view(1, 1, -1) + self.weather_mean.view(1, 1, -1)

    def _denorm_state(self, x):
        return x * self.state_std.view(1, 1, -1) + self.state_mean.view(1, 1, -1)

    def _denorm_target(self, x):
        return x * self.target_std.view(1, 1, -1) + self.target_mean.view(1, 1, -1)

    def _norm_aux(self, aux_real):
        return (aux_real - self.aux_mean.view(1, 1, -1)) / (self.aux_std.view(1, 1, -1) + 1e-6)

    # ------------------------------------------------------------------
    # Load branch — behavioral: temperature MLP + calendar + GRU temporal
    # ------------------------------------------------------------------
    def _load_branch(self, temp, y_mark, x_net_hist_real, x_mark_enc=None,
                     temp_hist=None, dx=None, x_load_hist_real=None):
        calendar_profile = torch.sigmoid(self.load_calendar_proj(y_mark))
        temp_contribution = self.temp_response(temp)

        # Use load history for autoreg + GRU when available (P0-2 fix).
        # Falls back to net history for backward compatibility.
        hist_for_load = x_load_hist_real if x_load_hist_real is not None else x_net_hist_real
        recent_avg = hist_for_load[:, -24:, :].mean(dim=1, keepdim=True)
        autoreg_correction = self.load_autoreg_proj(recent_avg)

        if dx is not None:
            base = self.load_base + dx[:, self._DX_LOAD_BASE].view(-1, 1, 1)
            calendar_gain = F.softplus(self.load_calendar_gain + dx[:, self._DX_LOAD_CALENDAR_GAIN].view(-1, 1, 1))
        else:
            base = self.load_base
            calendar_gain = F.softplus(self.load_calendar_gain)

        load_pre = (
            base
            + temp_contribution
            + calendar_gain * calendar_profile
            + F.softplus(self.load_autoreg_gain) * autoreg_correction
        )

        # GRU temporal correction (uses load history when available)
        if x_mark_enc is not None and temp_hist is not None:
            load_pre = load_pre + self.load_temporal(
                hist_for_load, temp_hist, x_mark_enc, y_mark,
            )

        load_theory = F.softplus(load_pre)
        return load_theory, calendar_profile

    # ------------------------------------------------------------------
    # Battery branch — signed power, guaranteed chg/dis exclusivity
    # ------------------------------------------------------------------
    def _battery_branch(self, weather_phys, y_mark, x_battery_hist_real):
        batch_size, pred_len, _ = weather_phys.shape
        last_power = x_battery_hist_real[:, -1, 0:1]
        last_soc = x_battery_hist_real[:, -1, 1:2]
        mean_power = x_battery_hist_real[:, :, 0:1].mean(dim=1)
        mean_soc = x_battery_hist_real[:, :, 1:2].mean(dim=1)

        if self._E_max.item() > 0 and self._P_max.item() > 0:
            power_limit = self._P_max.view(1).expand(batch_size, 1)
            capacity = self._E_max.view(1).expand(batch_size, 1)
        else:
            max_abs_power = x_battery_hist_real[:, :, 0].abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
            hist_soc_max = x_battery_hist_real[:, :, 1].amax(dim=1, keepdim=True).clamp_min(1e-4)
            power_limit = max_abs_power * (F.softplus(self.battery_limit_scale) + 0.5)
            capacity = hist_soc_max * (F.softplus(self.battery_capacity_scale) + 0.5)

        eta_charge = 0.80 + 0.19 * torch.sigmoid(self.eta_charge_raw)
        eta_discharge = 0.80 + 0.19 * torch.sigmoid(self.eta_discharge_raw)

        if self.no_battery_branch:
            zeros = torch.zeros(batch_size, pred_len, 1, device=weather_phys.device, dtype=weather_phys.dtype)
            battery_soc = last_soc.unsqueeze(1).expand(-1, pred_len, -1)
            return zeros, zeros, zeros, battery_soc, capacity, eta_charge, eta_discharge

        context = torch.cat([last_power, last_soc, mean_power, mean_soc, last_soc], dim=-1)
        base_context = self.battery_context_proj(context)

        soc_prev = last_soc
        power_seq = []
        charge_seq = []
        discharge_seq = []
        soc_seq = []

        for step in range(pred_len):
            step_context = torch.cat(
                [
                    base_context,
                    weather_phys[:, step, :],
                    y_mark[:, step, :],
                    soc_prev / (capacity + 1e-6),
                    (soc_prev - 0.5 * capacity) / (0.5 * capacity + 1e-6),
                ],
                dim=-1,
            )
            hidden = self.battery_step_proj(step_context)
            power_raw = self.battery_power_head(hidden)  #  (B, 1)
            power = power_limit * torch.tanh(power_raw)     # signed, ∈ [-Pmax, Pmax]

            charge = F.relu(power)
            discharge = F.relu(-power)

            soc_next = soc_prev + (eta_charge * charge - discharge / eta_discharge) * self.dt_hours
            soc_next = torch.minimum(torch.maximum(soc_next, torch.zeros_like(soc_next)), capacity)

            power_seq.append(power)
            charge_seq.append(charge)
            discharge_seq.append(discharge)
            soc_seq.append(soc_next)
            soc_prev = soc_next

        battery_power = torch.stack(power_seq, dim=1)
        battery_charge = torch.stack(charge_seq, dim=1)
        battery_discharge = torch.stack(discharge_seq, dim=1)
        battery_soc = torch.stack(soc_seq, dim=1)

        return battery_charge, battery_discharge, battery_power, battery_soc, capacity, eta_charge, eta_discharge

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x_weather_hist, x_weather_future, y_mark, x_net_hist, x_battery_hist,
                portfolio_ids=None, x_mark_enc=None, x_load_hist=None):
        hist_weather_real = self._denorm_weather(x_weather_hist)
        weather_real = self._denorm_weather(x_weather_future)
        x_net_hist_real = self._denorm_target(x_net_hist)
        battery_hist_real = self._denorm_state(x_battery_hist)
        x_load_hist_real = None
        if x_load_hist is not None:
            x_load_hist_real = (x_load_hist * self.aux_std[0] + self.aux_mean[0])
            if x_load_hist_real.dim() == 2:
                x_load_hist_real = x_load_hist_real.unsqueeze(-1)

        dx = self._get_portfolio_delta(portfolio_ids)  # (B, 16) or None

        temp = weather_real[..., 0:1]
        temp_hist = hist_weather_real[..., 0:1]
        solar_energy = (weather_real[..., 1:2].clamp_min(0.0) / 3600000.0).clamp_min(0.0)
        wind = weather_real[..., 2:3].clamp_min(0.0)
        weather_phys = torch.cat([temp, solar_energy, wind], dim=-1)

        # -- load --
        load_theory, calendar_profile = self._load_branch(
            temp=temp, y_mark=y_mark, x_net_hist_real=x_net_hist_real,
            x_mark_enc=x_mark_enc, temp_hist=temp_hist, dx=dx,
            x_load_hist_real=x_load_hist_real,
        )

        # -- pv (with per-portfolio deltas) --
        if dx is not None:
            pv_scale = F.softplus(self.pv_scale + dx[:, self._DX_PV_SCALE].view(-1, 1, 1))
            pv_cap = self.pv_capacity + dx[:, self._DX_PV_CAPACITY].view(-1, 1, 1)
            pv_temp_coeff = F.softplus(self.pv_temp_coeff + dx[:, self._DX_PV_TEMP_COEFF].view(-1, 1, 1))
        else:
            pv_scale = F.softplus(self.pv_scale)
            pv_cap = self.pv_capacity
            pv_temp_coeff = F.softplus(self.pv_temp_coeff)
        # Symmetric temperature effect: efficiency ↑ below 25°C, ↓ above
        pv_temp_factor = (1.0 - pv_temp_coeff * 0.01 * (temp - 25.0)).clamp(0.5, 1.5)
        pv_theory = (pv_scale + F.softplus(pv_cap)) * solar_energy * pv_temp_factor

        # -- wind (with per-portfolio deltas) --
        if dx is not None:
            w_cut_in = F.softplus(self.wind_cut_in + dx[:, self._DX_WIND_CUT_IN].view(-1, 1, 1))
            w_rated_delta = F.softplus(self.wind_rated_delta + dx[:, self._DX_WIND_RATED_DELTA].view(-1, 1, 1))
            w_cut_out_delta = F.softplus(self.wind_cut_out_delta + dx[:, self._DX_WIND_CUT_OUT_DELTA].view(-1, 1, 1))
            w_scale = F.softplus(self.wind_scale + dx[:, self._DX_WIND_SCALE].view(-1, 1, 1))
            w_cap = self.wind_capacity + dx[:, self._DX_WIND_CAPACITY].view(-1, 1, 1)
        else:
            w_cut_in = F.softplus(self.wind_cut_in)
            w_rated_delta = F.softplus(self.wind_rated_delta)
            w_cut_out_delta = F.softplus(self.wind_cut_out_delta)
            w_scale = F.softplus(self.wind_scale)
            w_cap = self.wind_capacity
        rated = w_cut_in + w_rated_delta
        cut_out = rated + w_cut_out_delta
        # Differentiable soft sigmoid gates (replaces boolean masks that blocked gradient)
        temp = 0.5  # transition sharpness — smaller = sharper
        running_mask = torch.sigmoid((wind - w_cut_in) / temp) * torch.sigmoid((cut_out - wind) / temp)
        plateau = torch.sigmoid((wind - rated) / temp) * torch.sigmoid((cut_out - wind) / temp)
        rising_curve = ((wind - w_cut_in) / (rated - w_cut_in + 1e-6)).clamp(0.0, 1.0) ** 3
        wind_curve = plateau + (1.0 - plateau) * rising_curve
        wind_theory = (w_scale + F.softplus(w_cap)) * wind_curve * running_mask

        # -- battery --
        batt_result = self._battery_branch(weather_phys, y_mark, battery_hist_real)
        battery_charge = batt_result[0]
        battery_discharge = batt_result[1]
        battery_power = batt_result[2]
        battery_soc = batt_result[3]
        battery_capacity = batt_result[4]
        eta_charge = batt_result[5]
        eta_discharge = batt_result[6]

        # -- theory net --
        theory_net_real = load_theory - pv_theory - wind_theory + battery_power

        # -- battery conditioning features (real units, normalised ratios) --
        cap_safe = battery_capacity.clamp_min(1e-6).unsqueeze(1).expand(-1, theory_net_real.shape[1], -1)
        if self._P_max.item() > 0:
            pwr_safe = self._P_max.view(1, 1, 1).expand(-1, theory_net_real.shape[1], -1)
        else:
            pwr_safe = battery_hist_real[:, :, 0].abs().amax(dim=1, keepdim=True).clamp_min(1e-4).unsqueeze(1)
        battery_feats = torch.cat(
            [
                battery_soc / cap_safe,                                    # soc_norm ∈ [0,1]
                (cap_safe - battery_soc) / pwr_safe.clamp_min(1e-4),       # headroom (hours)
                torch.ones_like(battery_soc) * eta_charge,                  # η_c
                torch.ones_like(battery_soc) * eta_discharge,               # η_d
            ],
            dim=-1,
        )  # (B, pred_len, 4)

        # -- normalised theory components (kept for diagnostics / loss) --
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
            "theory_net_real": theory_net_real,
            "battery_feats_real": battery_feats,
        }
        return component_norm, states
