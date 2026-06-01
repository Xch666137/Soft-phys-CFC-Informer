import torch.nn as nn
from ..layers.decomposition import series_decomp


class DLinear(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.target_dim = len(getattr(configs, "target_cols", [])) or self.c_out
        self.known_future_num = len(
            getattr(configs, "known_future_covariate_cols", getattr(configs, "covariate_cols", [])) or []
        )

        self.decomposition = series_decomp(kernel_size=25)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.channel_projection = nn.Linear(self.enc_in, self.c_out) if self.enc_in != self.c_out else None
        self.future_cov_projection = nn.Linear(self.known_future_num, self.c_out) if self.known_future_num > 0 else None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        seasonal_init, trend_init = self.decomposition(x_enc)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        seasonal_output = self.linear_seasonal(seasonal_init)
        trend_output = self.linear_trend(trend_init)

        x = (seasonal_output + trend_output).permute(0, 2, 1)
        if self.channel_projection is not None:
            x = self.channel_projection(x)

        if self.future_cov_projection is not None and x_dec is not None:
            cov_start = self.target_dim
            cov_end = cov_start + self.known_future_num
            future_cov = x_dec[:, -self.pred_len :, cov_start:cov_end]
            x = x + self.future_cov_projection(future_cov)

        return x
