import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, configs):
        super(LSTM, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.dec_out = configs.c_out
        self.d_model = configs.d_model
        self.n_layers = configs.e_layers  # 复用参数 e_layers 作为 LSTM 层数

        # LSTM 编码器
        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.d_model,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=0.1 if self.n_layers > 1 else 0
        )

        # 投影层：直接从 LSTM 的最后状态映射到 (Pred_Len, Dec_Out)
        # 这种 Direct Strategy 在长序列预测中通常比自回归更稳健
        self.projection = nn.Linear(self.d_model, self.pred_len * self.dec_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, Seq_Len, Enc_In]

        # LSTM output: [Batch, Seq, Hidden], (h_n, c_n)
        out, _ = self.lstm(x_enc)

        # 取最后一个时间步的隐状态作为上下文
        last_hidden = out[:, -1, :]  # [Batch, Hidden]

        # 映射到未来序列
        preds = self.projection(last_hidden)  # [Batch, Pred_Len * Dec_Out]

        # 重塑形状
        preds = preds.reshape(-1, self.pred_len, self.dec_out)

        return preds