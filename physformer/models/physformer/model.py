import torch
import torch.nn as nn

from ...layers.attention import FullAttention, ProbAttention
from ...layers.embedding import DataEmbedding
from ...layers.encoder import Encoder
from .conditioning import PhysicsFiLM, UnifiedResidualHead, WeatherFusion
from .flatten_head import FlattenHead
from .physical_layer import ExplicitVPPPhysicalLayer
from .temporal_decoder import TemporalDecoder


class PhysFormer(nn.Module):
    """Physics-guided Transformer for VPP net-injection forecasting.

    Data flow::

        Input → Encoder → TemporalDecoder(seq→pred) → WeatherFusion
                                                           ↓
        PhysicalLayer → theory_net + battery_feats → PhysicsFiLM
                                                           ↓
                                              UnifiedResidualHead
                                                           ↓
                                    pred_net = theory_net + residual

    The physical layer provides *conditioning features* (theory_net,
    battery state ratios) that modulate the data-driven pathway via
    FiLM, not via separate prediction branches.
    """

    def __init__(
        self,
        enc_in,
        seq_len,
        pred_len,
        factor=5,
        d_model=256,
        n_heads=8,
        e_layers=2,
        d_ff=512,
        dropout=0.1,
        attn="full",
        embed="custom",
        freq="t",
        activation="gelu",
        use_rope=True,
        rope_base=10000,
        distil=False,
        weather_mean=None,
        weather_std=None,
        state_mean=None,
        state_std=None,
        target_mean=None,
        target_std=None,
        aux_mean=None,
        aux_std=None,
        no_phys_stream=False,
        no_battery_branch=False,
        no_soc_consistency=False,
        no_future_weather=False,
        no_deep_battery_context=False,
        battery_meta=None,
        use_temporal_decoder=True,
        film_scale=0.5,
        decoder_n_heads=None,
        num_portfolios=0,
        time_feat_dim=8,
        load_gru_hidden=96,
        load_gru_use_temp=True,
        load_temp_model="mlp",
        detach_scale=0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.no_phys_stream = no_phys_stream
        self.no_battery_branch = no_battery_branch
        self.no_soc_consistency = no_soc_consistency
        self.no_future_weather = no_future_weather
        self.use_temporal_decoder = use_temporal_decoder
        self.detach_mode = "none"  # "none" | "selective" | "full"
        self.detach_scale = detach_scale  # gradient scaling factor (0=full detach, 1=no detach)

        # --- statistics buffers ---
        self.register_buffer("target_mean", self._to_buffer(target_mean, 1, 0.0))
        self.register_buffer("target_std", self._to_buffer(target_std, 1, 1.0))
        self.register_buffer("aux_mean", self._to_buffer(aux_mean, 5, 0.0))
        self.register_buffer("aux_std", self._to_buffer(aux_std, 5, 1.0))
        self.register_buffer("state_mean", self._to_buffer(state_mean, 2, 0.0))
        self.register_buffer("state_std", self._to_buffer(state_std, 2, 1.0))

        # --- encoder ---
        self.stat_embedding = DataEmbedding(
            c_in=enc_in, d_model=d_model, embed_type=embed, freq=freq, dropout=dropout,
            time_enc_in=time_feat_dim,
        )
        attn_cls = ProbAttention if attn == "prob" else FullAttention
        self.encoder = Encoder(
            num_layers=e_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            attn_cls=attn_cls,
            dropout=dropout,
            use_distillation=distil,
            use_rope=use_rope,
            rope_base=rope_base,
        )

        # --- temporal decoder (replaces FlattenHead) ---
        dec_heads = decoder_n_heads if decoder_n_heads is not None else max(1, n_heads // 2)
        if use_temporal_decoder:
            self.temporal_decoder = TemporalDecoder(
                seq_len=seq_len, pred_len=pred_len, d_model=d_model,
                n_heads=dec_heads, dropout=dropout, time_enc_in=time_feat_dim,
            )
        else:
            self.temporal_decoder = None
        self.flatten_head = FlattenHead(seq_len, d_model, pred_len, dropout)

        # --- weather fusion ---
        self.weather_fusion = WeatherFusion(
            d_model=d_model, weather_dim=3, time_dim=time_feat_dim,
            n_heads=dec_heads, dropout=dropout,
        )

        # --- physical layer ---
        self.phys_layer = ExplicitVPPPhysicalLayer(
            d_model=d_model,
            weather_dim=3,
            battery_state_dim=2,
            time_feat_dim=time_feat_dim,
            weather_mean=weather_mean,
            weather_std=weather_std,
            state_mean=state_mean,
            state_std=state_std,
            target_mean=target_mean,
            target_std=target_std,
            aux_mean=aux_mean,
            aux_std=aux_std,
            no_battery_branch=no_battery_branch,
            no_deep_battery_context=no_deep_battery_context,
            battery_meta=battery_meta,
            num_portfolios=num_portfolios,
            load_gru_hidden=load_gru_hidden,
            load_gru_use_temp=load_gru_use_temp,
            load_temp_model=load_temp_model,
        )

        # --- physics conditioning (FiLM) ---
        self.physics_film = PhysicsFiLM(d_model=d_model, physics_dim=9, film_scale=film_scale)

        # --- output head ---
        self.unified_head = UnifiedResidualHead(d_model=d_model, dropout=dropout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_buffer(value, dim, default):
        if value is None:
            return torch.full((dim,), float(default))
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if tensor.numel() != dim:
            raise ValueError(f"Expected {dim} values, got {tensor.numel()}.")
        return tensor

    def _norm_target(self, target_real):
        return (target_real - self.target_mean.view(1, 1, -1)) / (self.target_std.view(1, 1, -1) + 1e-6)

    def _norm_aux(self, aux_real):
        return (aux_real - self.aux_mean.view(1, 1, -1)) / (self.aux_std.view(1, 1, -1) + 1e-6)

    def _build_component_net_real(self, component_real, component_residual):
        """Reconstruct net injection from component theory + residual in REAL MW.

        component_real columns: [load, pv, wind, batt_pwr, batt_soc] in MW
        component_residual columns: same, in per-component z-score space

        Returns pred_net in target z-score space.
        Power balance: net = load - pv - wind + batt
        """
        # Denorm residual: zero-mean delta → scale by aux_std only, NO aux_mean
        component_residual_real = component_residual * self.aux_std.view(1, 1, -1)

        # Add to theory in real units
        component_pred_real = component_real + component_residual_real

        # Power balance in real MW
        pred_net_real = (
            component_pred_real[..., 0:1]   # load
            - component_pred_real[..., 1:2] # pv
            - component_pred_real[..., 2:3] # wind
            + component_pred_real[..., 3:4] # batt_p
        )

        # Normalize to target z-score space for MSE loss
        return self._norm_target(pred_net_real)

    def _build_zero_physics(self, batch_size, pred_len, device, dtype):
        """Return zero-filled physics features when no_phys_stream=True."""
        zeros_1 = torch.zeros(batch_size, pred_len, 1, device=device, dtype=dtype)
        zeros_4 = torch.zeros(batch_size, pred_len, 4, device=device, dtype=dtype)
        zeros_5 = torch.zeros(batch_size, pred_len, 5, device=device, dtype=dtype)
        theory_net_norm = zeros_1
        component_norm = zeros_5
        physics_features = torch.cat([zeros_5, zeros_4], dim=-1)  # 9 dim
        states = {
            "component_theory_real": zeros_5,
            "load_theory_real": zeros_1,
            "pv_theory_real": zeros_1,
            "wind_theory_real": zeros_1,
            "battery_power_theory_real": zeros_1,
            "battery_soc_theory_real": zeros_1,
            "battery_charge_theory_real": zeros_1,
            "battery_discharge_theory_real": zeros_1,
            "battery_capacity_real": torch.ones_like(zeros_1),
            "battery_eta_charge": torch.ones_like(zeros_1),
            "battery_eta_discharge": torch.ones_like(zeros_1),
            "theory_net_real": zeros_1,
            "battery_feats_real": zeros_4,
        }
        return theory_net_norm, physics_features, states

    # ------------------------------------------------------------------
    # Freeze helpers (minimal — single-stage training)
    # ------------------------------------------------------------------
    def freeze_for_physics_warmup(self):
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith("phys_layer.")

    def phys_layer_parameters(self):
        return list(self.phys_layer.parameters())

    def non_phys_layer_parameters(self):
        ids = {id(p) for p in self.phys_layer.parameters()}
        return [p for p in self.parameters() if id(p) not in ids]

    def set_detach_mode(self, mode):
        """Phase-aware gradient decoupling: 'none' | 'selective' | 'full'."""
        self.detach_mode = mode

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x_net_hist, x_weather_hist, x_battery_hist, x_weather_future,
                x_mark_enc, y_mark, portfolio_ids=None, x_load_hist=None, x_component_hist=None):
        B = x_net_hist.shape[0]
        P = self.pred_len
        device = x_net_hist.device
        dtype = x_net_hist.dtype

        if self.no_future_weather:
            x_weather_future = torch.zeros_like(x_weather_future)

        # ---- Step 1: Encoder ----
        history_input = torch.cat([x_net_hist, x_weather_hist, x_battery_hist], dim=-1)
        stat_hist = self.stat_embedding(history_input, x_mark_enc)
        enc_out = self.encoder(stat_hist)  # (B, seq_len, d_model)

        # ---- Step 2: Temporal decoding (seq → pred) ----
        if self.use_temporal_decoder and self.temporal_decoder is not None:
            coarse_future = self.temporal_decoder(enc_out, y_mark)  # (B, P, d_model)
        else:
            coarse_future = self.flatten_head(enc_out)

        # ---- Step 3: Weather fusion ----
        weather_latent = self.weather_fusion(coarse_future, x_weather_future, y_mark)

        # ---- Step 4: Physical layer ----
        if self.no_phys_stream:
            theory_net, physics_features, physics_states = self._build_zero_physics(
                B, P, device, dtype,
            )
            component_real = physics_states["component_theory_real"]
        else:
            _component_norm, physics_states = self.phys_layer(
                x_weather_hist=x_weather_hist,
                x_weather_future=x_weather_future,
                y_mark=y_mark,
                x_net_hist=x_net_hist,
                x_battery_hist=x_battery_hist,
                portfolio_ids=portfolio_ids,
                x_mark_enc=x_mark_enc,
                x_load_hist=x_load_hist,
            )
            component_real = physics_states["component_theory_real"]
            theory_net_real = physics_states["theory_net_real"]
            battery_feats = physics_states["battery_feats_real"]
            component_norm = self._norm_aux(component_real)
            theory_net = self._norm_target(theory_net_real)
            physics_features = torch.cat([component_norm, battery_feats], dim=-1)  # (B, P, 9)

        # ---- Step 5: Physics conditioning (FiLM) ----
        conditioned = self.physics_film(weather_latent, physics_features)

        # ---- Step 6: Component residual head (progressive decoupling) ----
        # Phase 1 (none): full gradient flow — joint optimization
        # Phase 2 (selective): detach load/pv/wind conflict, keep battery gradient
        # Phase 3 (full): detach everything (unused by default)
        if self.detach_mode == "selective":
            physics_for_head = physics_features.clone()
            if self.detach_scale > 0.0:
                s = self.detach_scale
                detached = physics_features[..., :3].detach()
                physics_for_head[..., :3] = physics_features[..., :3] * s + detached * (1.0 - s)
            else:
                physics_for_head[..., :3] = physics_features[..., :3].detach()  # load, pv, wind
        elif self.detach_mode == "full":
            physics_for_head = physics_features.detach()
        else:
            physics_for_head = physics_features
        component_residual = self.unified_head(conditioned, physics_for_head)  # (B, P, 5)

        # ---- Step 7: Reconstruct net injection in real MW ----
        pred_net = self._build_component_net_real(component_real, component_residual)
        theory_net = theory_net if self.no_phys_stream else self._norm_target(theory_net_real)

        return {
            "pred_net": pred_net,
            "theory_net": theory_net,
            "residual": component_residual,
            "component_residual": component_residual,
            "physics_states": physics_states,
        }
