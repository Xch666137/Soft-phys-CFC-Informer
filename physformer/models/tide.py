import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm(residual + x)


class TiDE(nn.Module):
    """
    Dense encoder-decoder baseline for forecasting with future-known covariates.
    This implementation is tailored to the thesis benchmark schema:
    historical net/weather/battery_state -> global history context,
    future weather/time features -> horizon-wise decoder inputs.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = getattr(configs, "d_model", 256)
        self.d_ff = getattr(configs, "d_ff", self.d_model * 4)
        self.e_layers = getattr(configs, "e_layers", 2)
        self.dropout = getattr(configs, "dropout", 0.1)

        self.target_dim = len(getattr(configs, "target_cols", [])) or self.c_out
        self.known_future_num = len(
            getattr(configs, "known_future_covariate_cols", getattr(configs, "covariate_cols", [])) or []
        )
        self.time_dim = 8

        history_dim = self.seq_len * self.enc_in
        self.history_proj = nn.Sequential(
            nn.Linear(history_dim, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.d_model),
        )
        self.history_blocks = nn.ModuleList(
            [ResidualMLPBlock(self.d_model, self.d_ff, self.dropout) for _ in range(self.e_layers)]
        )

        horizon_input_dim = self.d_model + self.known_future_num + self.time_dim
        self.horizon_proj = nn.Linear(horizon_input_dim, self.d_model)
        self.horizon_blocks = nn.ModuleList(
            [ResidualMLPBlock(self.d_model, self.d_ff, self.dropout) for _ in range(max(1, self.e_layers - 1))]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.c_out),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        batch_size = x_enc.shape[0]

        history_context = self.history_proj(x_enc.reshape(batch_size, -1))
        for block in self.history_blocks:
            history_context = block(history_context)

        cov_start = self.target_dim
        cov_end = cov_start + self.known_future_num
        future_cov = x_dec[:, -self.pred_len :, cov_start:cov_end]
        future_marks = x_mark_dec[:, -self.pred_len :, : self.time_dim]
        repeated_context = history_context.unsqueeze(1).expand(-1, self.pred_len, -1)
        horizon_input = torch.cat([repeated_context, future_cov, future_marks], dim=-1)

        horizon_latent = self.horizon_proj(horizon_input)
        for block in self.horizon_blocks:
            horizon_latent = block(horizon_latent)

        return self.output_head(horizon_latent)
