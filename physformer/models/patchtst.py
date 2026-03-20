import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        return x + self.pe[:x.size(0), :]


class PatchTST(nn.Module):
    """
    PatchTST 模型实现 (精简版)
    论文: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023)
    """

    def __init__(self, configs):
        super(PatchTST, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        # Patch 参数 (论文推荐配置)
        self.patch_len = 16
        self.stride = 8
        self.d_model = 128
        self.n_heads = 4
        self.e_layers = 3
        self.dropout = 0.2

        # 计算 Patch 数量
        self.patch_num = int((self.seq_len - self.patch_len) / self.stride + 1)

        # Patching 线性投影
        self.value_embedding = nn.Linear(self.patch_len, self.d_model)
        self.position_embedding = PositionalEncoding(self.d_model, max_len=self.patch_num)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.e_layers)

        # 预测头 (Flatten -> Linear)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * self.d_model, self.pred_len),
            nn.Dropout(self.dropout)
        )

        # 通道对齐层
        if self.enc_in != self.c_out:
            self.channel_projection = nn.Linear(self.enc_in, self.c_out)
        else:
            self.channel_projection = None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 采用 Channel Independence 策略
        # x_enc: [Batch, Seq_Len, Enc_In] -> [Batch * Enc_In, 1, Seq_Len]
        B, L, M = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1).reshape(B * M, 1, L)

        # 1. Patching
        # 从序列中截取 patch_num 个长度为 patch_len 的块
        patches = []
        for i in range(self.patch_num):
            start = i * self.stride
            end = start + self.patch_len
            patches.append(x_enc[:, 0, start:end])  # [Batch*Enc_In, patch_len]

        # Stack 起来 -> [Batch*Enc_In, patch_num, patch_len]
        x = torch.stack(patches, dim=1)

        # 2. Embedding
        x = self.value_embedding(x)  # [Batch*Enc_In, patch_num, d_model]

        # Transformer 需要的输入通常是 [Seq, Batch, Feature] (如果不设置 batch_first=True)
        # 这里我们设了 batch_first=True，所以直接加上位置编码
        x = x.transpose(0, 1)  # [patch_num, Batch*Enc_In, d_model]
        x = self.position_embedding(x)
        x = x.transpose(0, 1)  # [Batch*Enc_In, patch_num, d_model]

        # 3. Transformer Encoder
        x = self.transformer_encoder(x)  # [Batch*Enc_In, patch_num, d_model]

        # 4. Flatten and Predict
        x = self.head(x)  # [Batch*Enc_In, Pred_Len]

        # 5. Reshape back
        x = x.reshape(B, M, self.pred_len).permute(0, 2, 1)  # [Batch, Pred_Len, Enc_In]

        # 6. 通道对齐 (如果需要)
        if self.channel_projection is not None:
            x = self.channel_projection(x)  # [Batch, Pred_Len, C_Out]

        return x