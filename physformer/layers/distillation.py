"""
自注意力蒸馏模块

基于Informer论文的自注意力蒸馏机制，通过卷积和池化减少序列长度和特征维度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SelfAttentionDistillation(nn.Module):
    """
    自注意力蒸馏模块
    
    通过1D卷积和最大池化减少序列长度和特征维度。
    论文中在编码器层之间应用此模块来减少计算复杂度。
    
    Args:
        d_model: 模型维度
        kernel_size: 卷积核大小
        stride: 池化步长
    """
    
    def __init__(self, d_model: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=kernel_size, 
            padding=kernel_size//2
        )
        self.pool = nn.MaxPool1d(
            kernel_size=kernel_size, 
            stride=stride, 
            padding=kernel_size//2
        )
        self.activation = nn.ReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            蒸馏后的张量 [batch_size, seq_len//stride, d_model]
        """
        batch_size, seq_len, d_model = x.size()
        
        # 处理短序列情况
        if seq_len <= 2:
            # 如果序列太短，不进行池化，只进行卷积
            x = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
            x = self.conv(x)
            x = self.activation(x)
            return x.transpose(1, 2)
        
        # 转置以适应Conv1d的输入格式 [batch_size, d_model, seq_len]
        x = x.transpose(1, 2)
        
        # 特征压缩
        x = self.conv(x)
        x = self.activation(x)

        # 使用填充确保可被stride整除
        if seq_len % self.pool.stride != 0:
            pad_len = self.pool.stride - (seq_len % self.pool.stride)
            x = F.pad(x, (0, pad_len), mode='constant', value=0)

        # 池化
        x = self.pool(x)
        
        # 转置回原始格式 [batch_size, new_seq_len, d_model]
        return x.transpose(1, 2)


class DistilledEncoderLayer(nn.Module):
    """
    带蒸馏的编码器层
    
    包装编码器层和蒸馏模块。
    """
    
    def __init__(self, encoder_layer: nn.Module, d_model: int):
        super().__init__()
        self.encoder_layer = encoder_layer
        self.distillation = SelfAttentionDistillation(d_model)
        
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        前向传播：编码器层 + 蒸馏
        """
        x = self.encoder_layer(x, mask)
        x = self.distillation(x)
        return x