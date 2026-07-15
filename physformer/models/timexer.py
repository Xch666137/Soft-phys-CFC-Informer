import torch
import torch.nn as nn


class TimeXerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, queries, memory):
        attn_out, _ = self.cross_attn(queries, memory, memory, need_weights=False)
        x = self.norm1(queries + attn_out)
        return self.norm2(x + self.ffn(x))


class TimeXer(nn.Module):
    """
    Exogenous-aware Transformer baseline.
    Shared historical memory is built from past target/weather/battery state,
    while future weather/time tokens act as horizon queries.
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = getattr(configs, "d_model", 256)
        self.d_ff = getattr(configs, "d_ff", self.d_model * 4)
        self.n_heads = getattr(configs, "n_heads", 4)
        self.e_layers = getattr(configs, "e_layers", 2)
        self.dropout = getattr(configs, "dropout", 0.1)

        self.target_dim = len(getattr(configs, "target_cols", [])) or self.c_out
        self.known_future_num = len(
            getattr(configs, "known_future_covariate_cols", getattr(configs, "covariate_cols", [])) or []
        )
        self.time_dim = getattr(configs, "time_feat_dim", 10)

        self.history_proj = nn.Linear(self.enc_in + self.time_dim, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.e_layers)

        self.future_proj = nn.Linear(self.known_future_num + self.time_dim, self.d_model)
        self.query_adapter = nn.Linear(self.d_model, self.d_model)
        self.refinement = TimeXerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout)
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.c_out),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        history_tokens = torch.cat([x_enc, x_mark_enc[:, :, : self.time_dim]], dim=-1)
        history_tokens = self.history_proj(history_tokens)
        history_memory = self.history_encoder(history_tokens)

        cov_start = self.target_dim
        cov_end = cov_start + self.known_future_num
        future_cov = x_dec[:, -self.pred_len :, cov_start:cov_end]
        future_marks = x_mark_dec[:, -self.pred_len :, : self.time_dim]
        future_tokens = self.future_proj(torch.cat([future_cov, future_marks], dim=-1))
        future_queries = self.query_adapter(future_tokens)
        refined = self.refinement(future_queries, history_memory)

        return self.output_head(refined)
