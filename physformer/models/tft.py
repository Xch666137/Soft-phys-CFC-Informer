import torch
import torch.nn as nn


class GatedResidualBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, context):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)
        gate = torch.sigmoid(self.gate(torch.cat([x, context], dim=-1)))
        return self.norm(x + gate * y)


class TFT(nn.Module):
    """
    Simplified Temporal Fusion Transformer baseline.
    Historical sequence is encoded with an LSTM, future-known inputs are decoded
    with an LSTM initialized from the history state, followed by temporal fusion.
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.d_model = getattr(configs, "d_model", 256)
        self.d_ff = getattr(configs, "d_ff", self.d_model * 4)
        self.dropout = getattr(configs, "dropout", 0.1)
        self.n_heads = getattr(configs, "n_heads", 4)
        self.e_layers = max(1, min(2, getattr(configs, "e_layers", 2)))
        self.enc_in = configs.enc_in

        self.target_dim = len(getattr(configs, "target_cols", [])) or self.c_out
        self.known_future_num = len(
            getattr(configs, "known_future_covariate_cols", getattr(configs, "covariate_cols", [])) or []
        )
        self.time_dim = 8

        self.hist_input_proj = nn.Linear(self.enc_in + self.time_dim, self.d_model)
        self.fut_input_proj = nn.Linear(self.known_future_num + self.time_dim, self.d_model)

        self.hist_lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.e_layers,
            batch_first=True,
            dropout=self.dropout if self.e_layers > 1 else 0.0,
        )
        self.future_lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.e_layers,
            batch_first=True,
            dropout=self.dropout if self.e_layers > 1 else 0.0,
        )
        self.attn = nn.MultiheadAttention(self.d_model, self.n_heads, dropout=self.dropout, batch_first=True)
        self.gated_fusion = GatedResidualBlock(self.d_model, self.d_ff, self.dropout)
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.c_out),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        hist_input = torch.cat([x_enc, x_mark_enc[:, :, : self.time_dim]], dim=-1)
        hist_input = self.hist_input_proj(hist_input)
        hist_memory, hist_state = self.hist_lstm(hist_input)

        cov_start = self.target_dim
        cov_end = cov_start + self.known_future_num
        future_cov = x_dec[:, -self.pred_len :, cov_start:cov_end]
        future_marks = x_mark_dec[:, -self.pred_len :, : self.time_dim]
        future_input = self.fut_input_proj(torch.cat([future_cov, future_marks], dim=-1))
        future_latent, _ = self.future_lstm(future_input, hist_state)

        attn_out, _ = self.attn(future_latent, hist_memory, hist_memory, need_weights=False)
        fused = self.gated_fusion(future_latent, attn_out)
        return self.output_head(fused)
