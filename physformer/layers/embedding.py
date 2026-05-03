import torch
import torch.nn as nn
import math
from .positional import PositionalEncoding

class TokenEmbedding(nn.Module):
    """
    Value Embedding: 将输入的数值特征 (c_in) 映射到 d_model
    使用 Conv1d 通常比 Linear 效果更好，因为它能捕捉局部的时序上下文。
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x: [Batch, Seq_Len, Channel] -> permute -> [Batch, Channel, Seq_Len]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class FixedEmbedding(nn.Module):
    """
    基于正弦/余弦的固定位置编码 (可选，用于时间特征)
    """
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    """
    时间特征嵌入: 为月、日、周、时、分分别建立 Embeddings
    这是 Informer/Autoformer 的标准做法。
    """
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TemporalEmbedding, self).__init__()

        # 根据 freq 确定包含哪些时间特征
        # 15min 数据通常对应 't'，包含: Month, Day, Weekday, Hour, Minute
        # 这里为了稳健，我们定义通用的 Embeddings
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        # 对应 data_factory 中生成的 data_stamp 列顺序:
        # [Month, Day, Weekday, Hour, Minute]
        self.month_embed = Embed(month_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.minute_embed = Embed(minute_size, d_model) # 15min数据有4个刻度

    def forward(self, x):
        # x shape: [Batch, Seq_Len, 5]
        x = x.long()

        # 累加所有时间维度的 Embedding
        return (self.month_embed(x[:, :, 0]) +
                self.day_embed(x[:, :, 1]) +
                self.weekday_embed(x[:, :, 2]) +
                self.hour_embed(x[:, :, 3]) +
                self.minute_embed(x[:, :, 4]))

class TimeFeatureEmbedding(nn.Module):
    """
    专门的时间嵌入：使用于 Sin/Cos 浮点输入
    将 time_enc_in 维度的连续特征直接映射到 d_model
    """
    def __init__(self, d_model, time_enc_in=8, freq='t'):
        super(TimeFeatureEmbedding, self).__init__()
        # 简单的线性投影，将 8维 特征 -> 256/512维
        self.embed = nn.Linear(time_enc_in, d_model, bias=False)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, time_enc_in] (e.g., 8)
        # return shape: [Batch, Seq_Len, d_model]
        return self.embed(x)

class DataEmbedding(nn.Module):
    """
    综合嵌入层: 支持自动识别是“查表模式”还是“线性投影模式”
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, time_enc_in=8):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        self.position_embedding = PositionalEncoding(d_model=d_model)

        self.embed_type = embed_type

        if embed_type == 'timeF' or embed_type == 'fixed':
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, time_enc_in=time_enc_in, freq=freq)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        x: 输入数据 [Batch, Seq_Len, c_in]
        x_mark: 时间戳特征 [Batch, Seq_Len, 5] (Month, Day, Week, Hour, Min)
        """
        # Value Embedding
        x = self.value_embedding(x)

        x_pos = self.position_embedding(x)

        # Time Embedding
        # 如果是 Sin/Cos 编码，x_mark 是 float；如果是旧逻辑，x_mark 是 int/long
        # TimeFeatureEmbedding 接受 float，TemporalEmbedding 内部会转 long
        x_time = self.temporal_embedding(x_mark)

        # Sum
        return self.dropout(x + x_pos + x_time)