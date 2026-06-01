"""
PhysFormer-iGT: Inverted Transformer for DVPP Forecasting.

P1-1 (Ablation A1): 8-token inverted Transformer PoC.
  5 component tokens (GRU over 672-step history) + 3 weather tokens (MLP over 96-step future)
  -> inverted self-attention across 8 tokens -> shared FFN decoder -> real-unit power balance
  -> net MSE only.

Phase B (B1): mask_indices parameter for Masked Component Pretraining.
  Zeroes out masked component history channels before GRU tokenization.
  Returns comp_preds_norm for pretraining component MAE loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Token embedders (batched — single GRU/MLP call for all components/weather)
# ---------------------------------------------------------------------------

class BatchedComponentEmbedding(nn.Module):
    """Encode ALL component histories into tokens in one batched GRU call.

    Input  (B, seq_len, 5)  ->  reshape (B*5, seq_len, 1)  ->  GRU  ->  project
    Output (B, 5, d_model)
    """

    def __init__(self, seq_len: int, d_model: int, gru_hidden: int = 64, num_components: int = 5):
        super().__init__()
        self.num_components = num_components
        self.gru = nn.GRU(
            input_size=1, hidden_size=gru_hidden, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        self.proj = nn.Linear(2 * gru_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.permute(0, 2, 1).reshape(B * self.num_components, -1)  # (B*C, seq_len)
        out, _ = self.gru(x.unsqueeze(-1))                             # (B*C, seq_len, 2*hidden)
        last = out[:, -1, :]                                           # (B*C, 2*hidden)
        tok = self.proj(last)                                          # (B*C, d_model)
        return tok.reshape(B, self.num_components, -1)                  # (B, C, d_model)


class BatchedWeatherEmbedding(nn.Module):
    """Encode ALL weather futures into tokens in one batched MLP call."""

    def __init__(self, pred_len: int, d_model: int, num_weather: int = 3):
        super().__init__()
        self.num_weather = num_weather
        self.mlp = nn.Sequential(
            nn.Linear(pred_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.permute(0, 2, 1).reshape(B * self.num_weather, -1)  # (B*W, pred_len)
        tok = self.mlp(x)                                          # (B*W, d_model)
        return tok.reshape(B, self.num_weather, -1)                 # (B, W, d_model)


# ---------------------------------------------------------------------------
# Inverted Encoder (attention across token/variate dimension)
# ---------------------------------------------------------------------------

class InvertedEncoderLayer(nn.Module):
    """Self-attention + FFN on the token dimension."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H = self.n_heads
        Q = self.w_q(x).view(B, N, H, -1).permute(0, 2, 1, 3)  # (B, H, N, d_k)
        K = self.w_k(x).view(B, N, H, -1).permute(0, 2, 1, 3)
        V = self.w_v(x).view(B, N, H, -1).permute(0, 2, 1, 3)

        V_out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale, is_causal=False,
        )
        attn_out = V_out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        attn_out = self.w_o(attn_out)

        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x


class InvertedEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, e_layers: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            InvertedEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# PhysFormer-iGT Model (pure A1 + mask_indices for Phase B pretraining)
# ---------------------------------------------------------------------------

class PhysFormeriGT(nn.Module):
    """8-token inverted Transformer — pure A1 architecture.

    No physics tokens, no graph bias, no twin tokens, no constraint tokens,
    no horizon decoder. 5 component tokens + 3 weather tokens + self-attention
    + shared FFN decoder + real-unit power balance + net MSE.

    Phase B: mask_indices parameter zeroes component history channels for
    Masked Component Pretraining.
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
        # --- Unused params (A2-A5 legacy, kept for _build_model compatibility) ---
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
        use_horizon_decoder=False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.register_buffer("target_mean", self._to_buffer(target_mean, 1, 0.0))
        self.register_buffer("target_std", self._to_buffer(target_std, 1, 1.0))
        self.register_buffer("aux_mean", self._to_buffer(aux_mean, 5, 0.0))
        self.register_buffer("aux_std", self._to_buffer(aux_std, 5, 1.0))

        gru_hidden = max(32, d_model // 4)

        self.comp_embedding = BatchedComponentEmbedding(
            seq_len, d_model, gru_hidden=gru_hidden, num_components=5,
        )
        self.weather_embedding = BatchedWeatherEmbedding(
            pred_len, d_model, num_weather=3,
        )
        self.encoder = InvertedEncoder(
            d_model=d_model, n_heads=n_heads, e_layers=e_layers,
            d_ff=d_ff, dropout=dropout,
        )

        # Per-component FFN projectors: one independent decoder per component type
        self.component_projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff // 2), nn.GELU(), nn.Linear(d_ff // 2, pred_len))
            for _ in range(4)
        ])

    @staticmethod
    def _to_buffer(value, dim, default):
        if value is None:
            return torch.full((dim,), float(default))
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if tensor.numel() != dim:
            raise ValueError(f"Expected {dim} values, got {tensor.numel()}.")
        return tensor

    # ------------------------------------------------------------------
    # Training protocol stubs (compatible with PhysFormerExperiment)
    # ------------------------------------------------------------------
    def set_detach_mode(self, mode: str):
        pass

    def freeze_for_physics_warmup(self):
        pass

    def phys_layer_parameters(self):
        return []

    def non_phys_layer_parameters(self):
        return list(self.parameters())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x_net_hist=None,
        x_weather_hist=None,
        x_battery_hist=None,
        x_weather_future=None,
        x_mark_enc=None,
        y_mark=None,
        portfolio_ids=None,
        x_load_hist=None,
        x_component_hist=None,
        mask_indices=None,
    ):
        """Forward pass.

        Args:
            x_component_hist: (B, seq_len, 5) component history [load,pv,wind,batt_p,batt_soc].
            x_weather_future:  (B, pred_len, 3) future weather [temp,irrad,wind_speed].
            mask_indices:      Optional list of component indices [0..3] to zero-mask
                               for Masked Component Pretraining. None during finetuning.
        Returns:
            dict with pred_net, comp_preds_norm, theory_net, residual, physics_states.
        """
        B = x_component_hist.shape[0]
        P = self.pred_len
        device = x_component_hist.device
        dtype = x_component_hist.dtype

        # ---- Apply component masking for pretraining ----
        x_comp = x_component_hist
        if mask_indices is not None:
            x_comp = x_component_hist.clone()
            for idx in mask_indices:
                x_comp[:, :, idx] = 0.0

        # ---- Tokenize components (single batched GRU call) ----
        comp_tokens = self.comp_embedding(x_comp)  # (B, 5, d_model)

        # ---- Tokenize weather (single batched MLP call) ----
        if x_weather_future is None:
            x_weather_future = torch.zeros(B, P, 3, device=device, dtype=dtype)
        weather_tokens = self.weather_embedding(x_weather_future)  # (B, 3, d_model)

        # ---- Combine: 8 tokens (5 comp + 3 weather) ----
        tokens = torch.cat([comp_tokens, weather_tokens], dim=1)  # (B, 8, d_model)

        # ---- Inverted self-attention ----
        tokens = self.encoder(tokens)  # (B, 8, d_model)

        # ---- Per-component FFN projection (4 independent decoders) ----
        comp_preds_norm = torch.stack([
            proj(tokens[:, i, :])
            for i, proj in enumerate(self.component_projectors)
        ], dim=-1)  # (B, 96, 4)

        # ---- Denorm component predictions to real MW ----
        comp_preds_real = (
            comp_preds_norm * self.aux_std[:4].view(1, 1, -1)
            + self.aux_mean[:4].view(1, 1, -1)
        )

        # ---- Real-unit power balance ----
        pred_net_real = (
            comp_preds_real[..., 0:1]   # load
            - comp_preds_real[..., 1:2] # pv
            - comp_preds_real[..., 2:3] # wind
            + comp_preds_real[..., 3:4] # batt_p
        )

        # ---- Normalize to target z-score for MSE loss ----
        pred_net = (pred_net_real - self.target_mean.view(1, 1, -1)) / (
            self.target_std.view(1, 1, -1) + 1e-6
        )

        # ---- Output dict (compatible with PhysLoss / PhysFormerExperiment) ----
        zeros_1 = torch.zeros(B, P, 1, device=device, dtype=dtype)
        zeros_4 = torch.zeros(B, P, 4, device=device, dtype=dtype)
        zeros_5 = torch.zeros(B, P, 5, device=device, dtype=dtype)

        return {
            "pred_net": pred_net,
            "comp_preds_norm": comp_preds_norm,
            "theory_net": zeros_1,
            "residual": zeros_5,
            "component_residual": zeros_5,
            "physics_states": {
                "component_theory_real": zeros_5,
                "theory_net_real": zeros_1,
                "battery_feats_real": zeros_4,
                "battery_soc_theory_real": zeros_1,
                "battery_capacity_real": torch.ones_like(zeros_1),
                "battery_eta_charge": torch.ones_like(zeros_1),
                "battery_eta_discharge": torch.ones_like(zeros_1),
                "load_theory_real": zeros_1,
                "pv_theory_real": zeros_1,
                "wind_theory_real": zeros_1,
                "battery_power_theory_real": zeros_1,
                "battery_charge_theory_real": zeros_1,
                "battery_discharge_theory_real": zeros_1,
            },
        }
