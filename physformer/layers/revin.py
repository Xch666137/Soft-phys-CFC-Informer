"""
RevIN: Reversible Instance Normalization.
"""
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

