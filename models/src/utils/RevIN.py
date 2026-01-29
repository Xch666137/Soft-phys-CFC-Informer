"""
RevIN: Reversible Instance Normalization, 是一种专门用于时间序列预测（Time Series Forecasting）的数据归一化技术。
旨在解决时间序列数据中常见的分布漂移（Distribution Shift）或非平稳性（Non-stationarity）问题。
"""
from typing_extensions import Self

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        # 初始化可学习的仿射变换参数
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        # 智能切片逻辑：如果输入的 x 维度小于记录的统计量维度，则截取统计量
        c_in = x.shape[-1]
        mean = self.mean[..., :c_in]
        stdev = self.stdev[..., :c_in]

        x = x - mean
        x = x / stdev

        if self.affine:
            # 同样截取仿射参数
            weight = self.affine_weight[:c_in]
            bias = self.affine_bias[:c_in]
            x = x * weight
            x = x + bias
        return x

    def _denormalize(self, x):
        # 智能切片逻辑：主要用于处理预测输出（3维）与输入统计量（6维）不匹配的情况
        c_in = x.shape[-1]

        # 截取对应的 mean 和 stdev (假设前 c_in 列就是对应的列)
        mean = self.mean[..., :c_in]
        stdev = self.stdev[..., :c_in]

        if self.affine:
            # 截取仿射参数
            weight = self.affine_weight[:c_in]
            bias = self.affine_bias[:c_in]
            x = x - bias
            x = x / (weight + self.eps * self.eps)

        x = x * stdev
        x = x + mean
        return x

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

class InformerWithRevIN(nn.Module):
    """
    这是一个包装类，将 RevIN 自动应用到 Informer 的输入和输出。
    这样你就不需要修改训练循环(train loop)中的任何代码。
    """
    def __init__(self, model, num_features, c_out):
        super().__init__()
        self.model = model
        self.revin = RevIN(num_features)
        self.c_out = c_out

    def forward(self, x, x_mark, dec_inp, y_mark):
        # 1. 归一化输入 x
        # 注意：我们使用 x 的统计量来归一化 x 和 dec_inp
        # 这对于处理非平稳数据至关重要
        x = self.revin(x, 'norm')

        # 同时也归一化 decoder input，因为它与 x 处于同一量级
        # 我们复用 x 的 mean/std (stored in self.revin.mean/stdev)
        dec_inp = self.revin._normalize(dec_inp)

        # 2. 模型前向传播
        outputs = self.model(x, x_mark, dec_inp, y_mark)

        # 3. 如果输出是 tuple (例如包含 attention 权重)，只处理预测部分
        if isinstance(outputs, tuple):
            pred = outputs[0]
            pred = self.revin(pred, 'denorm')
            return (pred, outputs[1])
        else:
            outputs = self.revin(outputs, 'denorm')
            return outputs


class CFCInformerWithRevIN(nn.Module):
    """
    专门适配 CFCInformer (Encoder-Only) 的 RevIN 包装器。
    特点：
    1. 忽略 dec_inp 的归一化 (因为它不需要)。
    2. 显式处理 Encoder-Only 的数据流。
    """
    def __init__(self, model, num_features, c_out):
        super().__init__()
        self.model = model
        self.revin = RevIN(num_features)
        self.c_out = c_out

    def forward(self, x, x_mark, y_mark_dec=None):
        # 1. 归一化输入
        # 只归一化 x (Encoder Input)
        x = self.revin(x, 'norm')

        # 2. 模型前向传播
        # 注意：这里我们明确传递 None 给 dec_inp，不管外部传入了什么
        # 这样确保 CFCInformer 接收到的始终是符合预期的参数
        outputs = self.model(x, x_mark, y_mark_dec)

        # 3. 反归一化输出
        if isinstance(outputs, tuple):
            pred = outputs[0]
            pred = self.revin(pred, 'denorm')
            return (pred, outputs[1])
        else:
            outputs = self.revin(outputs, 'denorm')
            return outputs