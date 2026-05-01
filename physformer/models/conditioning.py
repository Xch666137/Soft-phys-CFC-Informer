import torch
import torch.nn as nn


class WeatherFusion(nn.Module):
    """Cross-attention fusion of coarse future latents with future weather context.

    Coarse future latents (from TemporalDecoder) attend to future weather +
    time encodings.  This is the single fusion point where "known future
    information" enters the prediction pathway.
    """

    def __init__(self, d_model, weather_dim=3, time_dim=8, n_heads=8, dropout=0.1):
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.Linear(weather_dim + time_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, coarse_future, x_weather_future, y_mark):
        context = torch.cat([x_weather_future, y_mark], dim=-1)
        context = self.context_proj(context)  # (B, P, d_model)
        attn_out, _ = self.cross_attn(coarse_future, context, context, need_weights=False)
        fused = self.ln1(coarse_future + attn_out)
        fused = self.ln2(fused + self.ffn(fused))
        return fused


class PhysicsFiLM(nn.Module):
    """Feature-wise Linear Modulation from physics features.

    Physics theory estimates scale and shift the data-driven latents
    via FiLM, so the physical prior structurally transforms the
    representation rather than being an optional concatenated feature.

    ``film_scale`` (default 0.5) clamps the modulation magnitude to
    prevent early-training instability from uncalibrated physics params.
    """

    def __init__(self, d_model, physics_dim=5, film_scale=0.5):
        super().__init__()
        self.film_scale = film_scale
        self.physics_proj = nn.Sequential(
            nn.Linear(physics_dim, d_model),
            nn.GELU(),
        )
        self.gamma_proj = nn.Linear(d_model, d_model)
        self.beta_proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, latent, physics_features):
        projected = self.physics_proj(physics_features)
        raw_gamma = self.gamma_proj(projected)
        raw_beta = self.beta_proj(projected)
        gamma = 1.0 + self.film_scale * torch.tanh(raw_gamma)
        beta = self.film_scale * torch.tanh(raw_beta)
        modulated = latent * gamma + beta
        return self.ln(modulated + latent)


class UnifiedResidualHead(nn.Module):
    """Single MLP head predicting the residual correction to theory_net.

    Projects the scalar theory_net into a small embedding space before
    concatenation, giving the model more capacity to express non-linear
    physics corrections than a single scalar input dimension.

    Final layer is initialised with small weights so that initially
    ``pred_net ≈ theory_net`` while still allowing gradient flow
    through the data-driven pathway from step 1.
    """

    def __init__(self, d_model, dropout=0.1, theory_proj_dim=32):
        super().__init__()
        self.theory_proj = nn.Sequential(
            nn.Linear(1, theory_proj_dim),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Linear(d_model + theory_proj_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.net[-1].weight, std=0.01 / (d_model ** 0.5))
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, conditioned, theory_net):
        theory_expanded = self.theory_proj(theory_net)  # (B, P, theory_proj_dim)
        inp = torch.cat([conditioned, theory_expanded], dim=-1)
        residual = self.net(inp)
        return residual
