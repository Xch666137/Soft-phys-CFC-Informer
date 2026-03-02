import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    DLinear 模型实现
    论文: "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)
    """

    def __init__(self, configs):
        super(DLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # 输入特征数 (例如: 6)
        self.c_out = configs.c_out  # 输出特征数 (例如: 3)

        # 移动平均核大小 (通常设为一个周期长度，例如 24)
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

        # Trend 预测层
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        # Seasonal 预测层
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)

        # 如果输入维度和输出维度不一致，需要一个通道映射层
        if self.enc_in != self.c_out:
            self.channel_projection = nn.Linear(self.enc_in, self.c_out)
        else:
            self.channel_projection = None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, Seq_Len, Enc_In]

        # 1. 序列分解
        seasonal_init, trend_init = self.decompsition(x_enc)

        # 2. 独立线性映射 (需要将时间维度放到最后一维)
        seasonal_init = seasonal_init.permute(0, 2, 1)  # [Batch, Enc_In, Seq_Len]
        trend_init = trend_init.permute(0, 2, 1)  # [Batch, Enc_In, Seq_Len]

        seasonal_output = self.Linear_Seasonal(seasonal_init)  # [Batch, Enc_In, Pred_Len]
        trend_output = self.Linear_Trend(trend_init)  # [Batch, Enc_In, Pred_Len]

        # 3. 组合并转置回正常形状
        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)  # [Batch, Pred_Len, Enc_In]

        # 4. 通道对齐 (如果需要预测的变量比输入的少，比如只预测 Load, PV, Wind)
        if self.channel_projection is not None:
            x = self.channel_projection(x)  # [Batch, Pred_Len, C_Out]

        return x