import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, configs):
        super(GRU, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.dec_out = configs.c_out
        self.d_model = configs.d_model
        self.n_layers = configs.e_layers
        self.dropout = configs.dropout

        # GRU 编码器
        self.gru = nn.GRU(
            input_size=self.enc_in,
            hidden_size=self.d_model,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=self.dropout if self.n_layers > 1 else 0
        )

        # 投影层：将 GRU 最后一步的隐状态映射到预测序列长度
        # 结构：Hidden_State -> Flatten -> Linear -> Reshape
        self.projection = nn.Linear(self.d_model, self.pred_len * self.dec_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, Seq_Len, Enc_In]

        # GRU 输出: out, h_n
        # out: [Batch, Seq, Hidden]
        # h_n: [Layers, Batch, Hidden]
        _, h_n = self.gru(x_enc)

        # 取最后一层的隐状态 [Batch, Hidden]
        h_last = h_n[-1, :, :]

        # 投影并重塑 [Batch, Pred_Len, C_Out]
        output = self.projection(h_last)
        output = output.view(x_enc.shape[0], self.pred_len, self.dec_out)

        return output