"""
PhysFormer-iGT: Physics-Twin Graph iTransformer for DVPP Forecasting.

P1-1 (Ablation A1): 8-token inverted Transformer PoC.
  5 component tokens (GRU over 672-step history) + 3 weather tokens (MLP over 96-step future)
  -> inverted self-attention across 8 tokens -> per-token FFN decoder -> real-unit power balance
  -> net MSE only.

Gate: single-seed MAE within 2x of c23 baseline (~2e-3 to ~4e-3 MW).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...layers.attention import FullAttention


# ---------------------------------------------------------------------------
# Token embedders
# ---------------------------------------------------------------------------

class ComponentTokenEmbedder(nn.Module):
    """Encode one component's 672-step history into a d_model token via GRU."""

    def __init__(self, seq_len: int, d_model: int, gru_hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1, hidden_size=gru_hidden, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        self.proj = nn.Linear(2 * gru_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len)
        out, _ = self.gru(x.unsqueeze(-1))            # (B, seq_len, 2*gru_hidden)
        return self.proj(out[:, -1, :])                # (B, d_model)


class WeatherTokenEmbedder(nn.Module):
    """Encode one weather variable's 96-step future into a d_model token via MLP."""

    def __init__(self, pred_len: int, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pred_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, pred_len)
        return self.mlp(x)                             # (B, d_model)


# ---------------------------------------------------------------------------
# Inverted Encoder (attention across token/variate dimension)
# ---------------------------------------------------------------------------

class InvertedEncoderLayer(nn.Module):
    """Self-attention + FFN on the token dimension."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = FullAttention(d_model, n_heads, dropout=dropout,
                                       output_attention=False, use_rope=False)
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
        attn_out, _ = self.attention(x, x, x, attn_mask=None)
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
# PhysFormer-iGT Model (P1-1 / A1)
# ---------------------------------------------------------------------------

class PhysFormeriGT(nn.Module):
    """8-token inverted Transformer — P1-1 minimal PoC.

    No PhysicalLayer, no physics tokens, no component loss, no graph bias.
    Only component tokens + weather tokens + self-attention + net MSE.
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

        self.register_buffer("target_mean", self._to_buffer(target_mean, 1, 0.0))
        self.register_buffer("target_std", self._to_buffer(target_std, 1, 1.0))
        self.register_buffer("aux_mean", self._to_buffer(aux_mean, 5, 0.0))
        self.register_buffer("aux_std", self._to_buffer(aux_std, 5, 1.0))

        gru_hidden = max(32, d_model // 4)

        # 5 component token embedders: [load, pv, wind, batt_p, batt_soc]
        self.comp_embeddings = nn.ModuleList([
            ComponentTokenEmbedder(seq_len, d_model, gru_hidden=gru_hidden)
            for _ in range(5)
        ])

        # 3 weather token embedders: [temp, irrad, wind_speed]
        self.weather_embeddings = nn.ModuleList([
            WeatherTokenEmbedder(pred_len, d_model)
            for _ in range(3)
        ])

        # Inverted encoder — attention across tokens
        self.encoder = InvertedEncoder(
            d_model=d_model, n_heads=n_heads, e_layers=e_layers,
            d_ff=d_ff, dropout=dropout,
        )

        # Per-token projectors for 4 main components (skip SOC)
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
    ):
        B = x_component_hist.shape[0]
        P = self.pred_len
        device = x_component_hist.device
        dtype = x_component_hist.dtype

        # ---- Tokenize components ----
        comp_tokens = torch.stack([
            embed(x_component_hist[..., i])
            for i, embed in enumerate(self.comp_embeddings)
        ], dim=1)                                          # (B, 5, d_model)

        # ---- Tokenize weather ----
        if x_weather_future is None:
            x_weather_future = torch.zeros(B, P, 3, device=device, dtype=dtype)
        weather_tokens = torch.stack([
            embed(x_weather_future[..., i])
            for i, embed in enumerate(self.weather_embeddings)
        ], dim=1)                                          # (B, 3, d_model)

        # ---- Combine tokens ----
        tokens = torch.cat([comp_tokens, weather_tokens], dim=1)  # (B, 8, d_model)

        # ---- Inverted self-attention ----
        tokens = self.encoder(tokens)                       # (B, 8, d_model)

        # ---- Per-component projection (first 4 tokens: load, pv, wind, batt_p) ----
        comp_preds_norm = torch.stack([
            proj(tokens[:, i, :])
            for i, proj in enumerate(self.component_projectors)
        ], dim=-1)                                         # (B, 96, 4)

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

        # ---- Output dict (compatible with PhysLoss) ----
        zeros_1 = torch.zeros(B, P, 1, device=device, dtype=dtype)
        zeros_4 = torch.zeros(B, P, 4, device=device, dtype=dtype)
        zeros_5 = torch.zeros(B, P, 5, device=device, dtype=dtype)

        return {
            "pred_net": pred_net,
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
