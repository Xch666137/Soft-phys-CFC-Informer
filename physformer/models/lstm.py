import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.dec_out = configs.c_out
        self.d_model = configs.d_model
        self.n_layers = getattr(configs, "e_layers", 1)
        self.dropout = getattr(configs, "dropout", 0.1)
        self.target_dim = len(getattr(configs, "target_cols", [])) or self.dec_out
        self.known_future_num = len(
            getattr(configs, "known_future_covariate_cols", getattr(configs, "covariate_cols", [])) or []
        )
        self.time_dim = getattr(configs, "time_feat_dim", 10)

        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.d_model,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=self.dropout if self.n_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(self.d_model, self.pred_len * self.dec_out)
        self.future_cov_projection = nn.Linear(self.known_future_num, self.dec_out) if self.known_future_num > 0 else None
        self.future_time_projection = nn.Linear(self.time_dim, self.dec_out) if self.time_dim > 0 else None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out, _ = self.lstm(x_enc)
        last_hidden = out[:, -1, :]
        preds = self.projection(last_hidden).reshape(-1, self.pred_len, self.dec_out)

        if self.future_cov_projection is not None and x_dec is not None:
            cov_start = self.target_dim
            cov_end = cov_start + self.known_future_num
            future_cov = x_dec[:, -self.pred_len :, cov_start:cov_end]
            preds = preds + self.future_cov_projection(future_cov)

        if self.future_time_projection is not None and x_mark_dec is not None:
            future_marks = x_mark_dec[:, -self.pred_len :, : self.time_dim]
            preds = preds + self.future_time_projection(future_marks)

        return preds
