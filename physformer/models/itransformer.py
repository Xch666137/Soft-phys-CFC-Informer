import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.attention import FullAttention

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    Inverted Transformer for multivariate time series forecasting.
    It embeds the entire time series of each variate into a token, 
    and applies self-attention across variates.
    """
    def __init__(self, configs):
        super(iTransformer, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = getattr(configs, 'output_attention', False)
        self.c_out = configs.c_out
        self.target_dim = len(getattr(configs, "target_cols", [])) or self.c_out
        self.known_future_num = len(
            getattr(configs, "known_future_covariate_cols", getattr(configs, "covariate_cols", [])) or []
        )
        
        # Inverted embedding: maps temporal sequence to embedding dimension
        self.enc_embedding = nn.Linear(configs.seq_len, configs.d_model)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    FullAttention(configs.d_model, configs.n_heads, dropout=configs.dropout,
                                  output_attention=self.output_attention), 
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=getattr(configs, 'activation', 'gelu')
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        
        # Decoder 
        self.projector = nn.Linear(configs.d_model, configs.pred_len)
        self.future_cov_projection = nn.Linear(self.known_future_num, self.c_out) if self.known_future_num > 0 else None
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc shape: [B, Seq, C]
        # Invert: map sequence length to feature dimension -> [B, C, Seq]
        x_enc = x_enc.permute(0, 2, 1)
        
        # Embedding: [B, C, Seq] -> [B, C, d_model]
        enc_out = self.enc_embedding(x_enc) 
        
        # Encoder applies self-attention across variates (C dimension)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # Project back to temporal dimension: [B, C, d_model] -> [B, C, pred_len]
        dec_out = self.projector(enc_out)
        
        # Permute back to standard format: [B, C, pred_len] -> [B, pred_len, C]
        dec_out = dec_out.permute(0, 2, 1)

        if self.future_cov_projection is not None and x_dec is not None:
            cov_start = self.target_dim
            cov_end = cov_start + self.known_future_num
            future_cov = x_dec[:, -self.pred_len :, cov_start:cov_end]
            dec_out = dec_out + self.future_cov_projection(future_cov)
        
        if self.output_attention:
            return dec_out, attns
        return dec_out

