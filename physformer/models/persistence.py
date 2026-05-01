"""Persistence forecast: repeats the last observed target value."""

import torch
import torch.nn as nn


class Persistence(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        c_out = getattr(configs, "c_out", 1)
        target_cols = getattr(configs, "target_cols", None)
        self.target_dim = len(target_cols) if target_cols else c_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        last_target = x_enc[:, -1:, : self.target_dim]
        return last_target.repeat(1, self.pred_len, 1)
