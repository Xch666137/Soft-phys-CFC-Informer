import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.dec_out = configs.c_out
        self.d_model = configs.d_model
        self.n_layers = configs.e_layers
        self.dropout = configs.dropout
        self.target_dim = len(getattr(configs, "target_cols", [])) or self.dec_out
        self.known_future_num = len(
            getattr(configs, "known_future_covariate_cols", getattr(configs, "covariate_cols", [])) or []
        )

        self.gru = nn.GRU(
            input_size=self.enc_in,
            hidden_size=self.d_model,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=self.dropout if self.n_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(self.d_model, self.pred_len * self.dec_out)
        self.future_cov_projection = nn.Linear(self.known_future_num, self.dec_out) if self.known_future_num > 0 else None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        _, h_n = self.gru(x_enc)
        h_last = h_n[-1, :, :]
        output = self.projection(h_last).view(x_enc.shape[0], self.pred_len, self.dec_out)

        if self.future_cov_projection is not None and x_dec is not None:
            cov_start = self.target_dim
            cov_end = cov_start + self.known_future_num
            future_cov = x_dec[:, -self.pred_len :, cov_start:cov_end]
            output = output + self.future_cov_projection(future_cov)

        return output
