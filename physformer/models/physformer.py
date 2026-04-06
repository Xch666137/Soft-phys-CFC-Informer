import torch
import torch.nn as nn

from ..layers.attention import FullAttention, ProbAttention
from ..layers.embedding import DataEmbedding
from ..layers.encoder import Encoder
from .flatten_head import FlattenHead
from .physical_layer import ExplicitVPPPhysicalLayer


class PhysFormer(nn.Module):
    """
    Thesis-only PhysFormer v2 for multi-portfolio net-injection forecasting.

    Inputs:
      - historical net injection
      - historical weather
      - historical battery power / SOC
      - future known weather
      - encoder / decoder time features

    Outputs:
      - pred_net: normalized net injection forecast [B, P, 1]
      - pred_aux: normalized auxiliary component forecasts [B, P, 5]
    """

    COMPONENT_NAMES = ("load", "pv", "wind", "battery")

    def __init__(
        self,
        enc_in,
        seq_len,
        pred_len,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_ff=512,
        dropout=0.1,
        attn="prob",
        embed="custom",
        freq="h",
        activation="gelu",
        use_rope=False,
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
        no_aux_supervision=False,
        no_soc_consistency=False,
        no_future_weather=False,
        shared_query_only=False,
        training_mode="net_first",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.no_phys_stream = no_phys_stream
        self.no_battery_branch = no_battery_branch
        self.no_aux_supervision = no_aux_supervision
        self.no_soc_consistency = no_soc_consistency
        self.no_future_weather = no_future_weather
        self.shared_query_only = shared_query_only
        self.training_mode = training_mode

        self.register_buffer("target_mean", self._to_buffer(target_mean, 1, 0.0))
        self.register_buffer("target_std", self._to_buffer(target_std, 1, 1.0))
        self.register_buffer("aux_mean", self._to_buffer(aux_mean, 5, 0.0))
        self.register_buffer("aux_std", self._to_buffer(aux_std, 5, 1.0))
        self.register_buffer("state_mean", self._to_buffer(state_mean, 2, 0.0))
        self.register_buffer("state_std", self._to_buffer(state_std, 2, 1.0))

        self.stat_embedding = DataEmbedding(c_in=enc_in, d_model=d_model, embed_type=embed, freq=freq, dropout=dropout)
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
        self.flatten_head = FlattenHead(seq_len, d_model, pred_len, dropout)

        self.phys_layer = ExplicitVPPPhysicalLayer(
            d_model=d_model,
            weather_dim=3,
            battery_state_dim=2,
            time_feat_dim=8,
            weather_mean=weather_mean,
            weather_std=weather_std,
            state_mean=state_mean,
            state_std=state_std,
            target_mean=target_mean,
            target_std=target_std,
            aux_mean=aux_mean,
            aux_std=aux_std,
            no_battery_branch=no_battery_branch,
        )

        self.future_context_proj = nn.Sequential(
            nn.Linear(3 + 8, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.refinement_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=max(1, n_heads // 2),
            dropout=dropout,
            batch_first=True,
        )
        self.refinement_ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.shared_query_adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.component_query_adapters = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
                for name in self.COMPONENT_NAMES
            }
        )
        self.component_type_embedding = nn.Embedding(len(self.COMPONENT_NAMES), d_model)
        self.refinement_norm1 = nn.ModuleDict({name: nn.LayerNorm(d_model) for name in self.COMPONENT_NAMES})
        self.refinement_norm2 = nn.ModuleDict({name: nn.LayerNorm(d_model) for name in self.COMPONENT_NAMES})

        head_input_dim = 2 * d_model + 1
        self.load_head = self._make_head(head_input_dim, 1, dropout)
        self.pv_head = self._make_head(head_input_dim, 1, dropout)
        self.wind_head = self._make_head(head_input_dim, 1, dropout)
        self.battery_head = self._make_head(2 * d_model + 2, 3, dropout)
        self.load_confidence_head = self._make_head(head_input_dim, 1, dropout)
        self.pv_confidence_head = self._make_head(head_input_dim, 1, dropout)
        self.wind_confidence_head = self._make_head(head_input_dim, 1, dropout)
        self.battery_confidence_head = self._make_head(2 * d_model + 2, 2, dropout)
        self.load_attribution_head = self._make_head(head_input_dim, 1, dropout)
        self.pv_attribution_head = self._make_head(head_input_dim, 1, dropout)
        self.wind_attribution_head = self._make_head(head_input_dim, 1, dropout)
        self.battery_attribution_head = self._make_head(2 * d_model + 2, 1, dropout)
        self.operational_scale = nn.Parameter(torch.zeros(5))
        self.operational_bias = nn.Parameter(torch.zeros(5))

    @staticmethod
    def _make_head(input_dim, output_dim, dropout):
        head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, output_dim),
        )
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    @staticmethod
    def _to_buffer(value, dim, default):
        if value is None:
            return torch.full((dim,), float(default))
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if tensor.numel() != dim:
            raise ValueError(f"Expected tensor with {dim} values, got {tensor.numel()}.")
        return tensor

    def _norm_aux(self, aux_real):
        return (aux_real - self.aux_mean.view(1, 1, -1)) / (self.aux_std.view(1, 1, -1) + 1e-6)

    def _norm_target(self, target_real):
        return (target_real - self.target_mean.view(1, 1, -1)) / (self.target_std.view(1, 1, -1) + 1e-6)

    def _build_zero_physics(self, batch_size, device, dtype, x_battery_hist):
        pred_len = self.pred_len
        zeros = torch.zeros(batch_size, pred_len, 1, device=device, dtype=dtype)
        battery_last_soc = (
            x_battery_hist[:, -1:, 1:2] * self.state_std[1].view(1, 1, 1) + self.state_mean[1].view(1, 1, 1)
        )
        battery_soc = battery_last_soc.expand(-1, pred_len, -1)
        component_real = torch.cat([zeros, zeros, zeros, zeros, battery_soc], dim=-1)
        states = {
            "component_theory_real": component_real,
            "load_theory_real": zeros,
            "pv_theory_real": zeros,
            "wind_theory_real": zeros,
            "battery_power_theory_real": zeros,
            "battery_soc_theory_real": battery_soc,
            "battery_charge_theory_real": zeros,
            "battery_discharge_theory_real": zeros,
            "battery_capacity_real": battery_soc.clone(),
            "battery_eta_charge": torch.ones_like(battery_soc),
            "battery_eta_discharge": torch.ones_like(battery_soc),
        }
        return component_real, self._norm_aux(component_real), states

    def _refine_component(self, component_name, coarse_future, future_context):
        if self.shared_query_only:
            query = self.shared_query_adapter(coarse_future)
        else:
            comp_idx = self.COMPONENT_NAMES.index(component_name)
            type_embed = self.component_type_embedding.weight[comp_idx].view(1, 1, -1)
            query = self.component_query_adapters[component_name](coarse_future + type_embed)

        attn_out, _ = self.refinement_attn(query, future_context, future_context, need_weights=False)
        refined = self.refinement_norm1[component_name](query + attn_out)
        refined = self.refinement_norm2[component_name](refined + self.refinement_ffn(refined))
        return refined

    def freeze_backbone_for_operational_fit(self):
        trainable_prefixes = (
            "load_head",
            "pv_head",
            "wind_head",
            "battery_head",
            "load_confidence_head",
            "pv_confidence_head",
            "wind_confidence_head",
            "battery_confidence_head",
            "load_attribution_head",
            "pv_attribution_head",
            "wind_attribution_head",
            "battery_attribution_head",
            "operational_scale",
            "operational_bias",
        )
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith(trainable_prefixes)

    def operational_parameter_names(self):
        names = []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                names.append(name)
        return names

    def forward(self, x_net_hist, x_weather_hist, x_battery_hist, x_weather_future, x_mark_enc, y_mark):
        if self.no_future_weather:
            x_weather_future = torch.zeros_like(x_weather_future)

        history_input = torch.cat([x_net_hist, x_weather_hist, x_battery_hist], dim=-1)
        stat_hist = self.stat_embedding(history_input, x_mark_enc)
        stat_hist = self.encoder(stat_hist)
        coarse_future = self.flatten_head(stat_hist)
        future_context = self.future_context_proj(torch.cat([x_weather_future, y_mark], dim=-1))

        batch_size = coarse_future.shape[0]
        device = coarse_future.device
        dtype = coarse_future.dtype

        if self.no_phys_stream:
            component_theory_real, component_theory_norm, physics_states = self._build_zero_physics(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                x_battery_hist=x_battery_hist,
            )
        else:
            component_theory_norm, physics_states = self.phys_layer(
                x_weather_hist=x_weather_hist,
                x_weather_future=x_weather_future,
                y_mark=y_mark,
                x_net_hist=x_net_hist,
                x_battery_hist=x_battery_hist,
            )
            component_theory_real = physics_states["component_theory_real"]

        load_latent = self._refine_component("load", coarse_future, future_context)
        pv_latent = self._refine_component("pv", coarse_future, future_context)
        wind_latent = self._refine_component("wind", coarse_future, future_context)
        battery_latent = self._refine_component("battery", coarse_future, future_context)

        component_embeddings = self.component_type_embedding.weight
        load_type = component_embeddings[0].view(1, 1, -1).expand(batch_size, self.pred_len, -1)
        pv_type = component_embeddings[1].view(1, 1, -1).expand(batch_size, self.pred_len, -1)
        wind_type = component_embeddings[2].view(1, 1, -1).expand(batch_size, self.pred_len, -1)
        battery_type = component_embeddings[3].view(1, 1, -1).expand(batch_size, self.pred_len, -1)

        load_theory = physics_states["load_theory_real"]
        pv_theory = physics_states["pv_theory_real"]
        wind_theory = physics_states["wind_theory_real"]
        battery_power_theory = physics_states["battery_power_theory_real"]
        battery_soc_theory = physics_states["battery_soc_theory_real"]
        battery_charge_theory = physics_states["battery_charge_theory_real"]
        battery_discharge_theory = physics_states["battery_discharge_theory_real"]
        battery_capacity = physics_states["battery_capacity_real"]

        load_head_in = torch.cat([load_latent, load_theory, load_type], dim=-1)
        pv_head_in = torch.cat([pv_latent, pv_theory, pv_type], dim=-1)
        wind_head_in = torch.cat([wind_latent, wind_theory, wind_type], dim=-1)
        battery_head_in = torch.cat([battery_latent, battery_power_theory, battery_soc_theory, battery_type], dim=-1)

        load_delta = self.load_head(load_head_in)
        pv_mod_raw = self.pv_head(pv_head_in)
        wind_mod_raw = self.wind_head(wind_head_in)
        battery_delta = self.battery_head(battery_head_in)
        load_confidence = torch.sigmoid(self.load_confidence_head(load_head_in))
        pv_confidence = torch.sigmoid(self.pv_confidence_head(pv_head_in))
        wind_confidence = torch.sigmoid(self.wind_confidence_head(wind_head_in))
        battery_confidence = torch.sigmoid(self.battery_confidence_head(battery_head_in))
        attribution_logits = torch.cat(
            [
                self.load_attribution_head(load_head_in),
                self.pv_attribution_head(pv_head_in),
                self.wind_attribution_head(wind_head_in),
                self.battery_attribution_head(battery_head_in),
            ],
            dim=-1,
        )

        if self.no_phys_stream:
            load_pred_real = torch.nn.functional.softplus(load_delta)
            pv_pred_real = torch.nn.functional.softplus(pv_mod_raw)
            wind_pred_real = torch.nn.functional.softplus(wind_mod_raw)
            pred_charge_real = torch.nn.functional.softplus(battery_delta[..., 0:1])
            pred_discharge_real = torch.nn.functional.softplus(battery_delta[..., 1:2])
            battery_soc_pred_real = torch.minimum(
                torch.maximum(battery_soc_theory + battery_delta[..., 2:3], torch.zeros_like(battery_capacity)),
                battery_capacity,
            )
        else:
            load_pred_real = torch.nn.functional.softplus(load_theory + load_delta)
            pv_scale = 1.0 + 0.5 * torch.tanh(pv_mod_raw)
            wind_scale = 1.0 + 0.5 * torch.tanh(wind_mod_raw)
            pv_pred_real = pv_theory * pv_scale
            wind_pred_real = wind_theory * wind_scale
            pred_charge_real = torch.nn.functional.softplus(battery_charge_theory + battery_delta[..., 0:1])
            pred_discharge_real = torch.nn.functional.softplus(battery_discharge_theory + battery_delta[..., 1:2])
            battery_soc_pred_real = torch.minimum(
                torch.maximum(battery_soc_theory + battery_delta[..., 2:3], torch.zeros_like(battery_capacity)),
                battery_capacity,
            )

        battery_power_pred_real = pred_charge_real - pred_discharge_real
        pred_aux_real = torch.cat(
            [
                load_pred_real,
                pv_pred_real,
                wind_pred_real,
                battery_power_pred_real,
                battery_soc_pred_real,
            ],
            dim=-1,
        )
        if self.training_mode == "operational_fit":
            scale = 1.0 + 0.1 * torch.tanh(self.operational_scale).view(1, 1, -1)
            bias = self.operational_bias.view(1, 1, -1)
            pred_aux_real = pred_aux_real * scale + bias
            pred_aux_real[..., 0:3] = pred_aux_real[..., 0:3].clamp_min(0.0)
            pred_aux_real[..., 4:5] = torch.minimum(
                torch.maximum(pred_aux_real[..., 4:5], torch.zeros_like(battery_capacity)),
                battery_capacity,
            )
        pred_aux = self._norm_aux(pred_aux_real)
        component_confidence = torch.cat(
            [
                load_confidence,
                pv_confidence,
                wind_confidence,
                battery_confidence[..., 0:1],
                battery_confidence[..., 1:2],
            ],
            dim=-1,
        )
        component_attribution = torch.softmax(attribution_logits, dim=-1)

        pred_net_real = (
            pred_aux_real[..., 0:1]
            - pred_aux_real[..., 1:2]
            - pred_aux_real[..., 2:3]
            + pred_aux_real[..., 3:4]
        )
        pred_net = self._norm_target(pred_net_real)

        return {
            "pred_net": pred_net,
            "pred_aux": pred_aux,
            "component_theory": component_theory_norm,
            "physics_states": physics_states,
            "pred_charge_real": pred_charge_real,
            "pred_discharge_real": pred_discharge_real,
            "battery_eta_charge": physics_states["battery_eta_charge"],
            "battery_eta_discharge": physics_states["battery_eta_discharge"],
            "component_confidence": component_confidence,
            "component_attribution": component_attribution,
        }
