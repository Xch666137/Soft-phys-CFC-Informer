import torch
import torch.nn as nn


class TemporalDecoder(nn.Module):
    """Cross-attention decoder with learnable future query positions.

    Each prediction step has an independent learned query that attends
    to the encoder memory.  Replaces the channel-independent
    ``FlattenHead`` (Linear(seq_len → pred_len)) with a richer
    representation that can model diurnal patterns, ramp events, and
    lagged weather effects.
    """

    def __init__(self, seq_len, pred_len, d_model, n_heads=8, dropout=0.1, time_enc_in=10):
        super().__init__()
        self.pred_len = pred_len
        self.query_pos = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
        self.time_proj = nn.Linear(time_enc_in, d_model)
        nn.init.xavier_uniform_(self.time_proj.weight, gain=0.1)
        nn.init.zeros_(self.time_proj.bias)
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

    def forward(self, memory, y_mark=None):
        B = memory.shape[0]
        query = self.query_pos.expand(B, -1, -1)
        if y_mark is not None:
            query = query + self.time_proj(y_mark)
        attn_out, _ = self.cross_attn(query, memory, memory, need_weights=False)
        query = self.ln1(query + attn_out)
        query = self.ln2(query + self.ffn(query))
        return query
