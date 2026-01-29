"""
Informer模型模块化实现

基于论文"Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"的完整实现
所有组件保持原始功能，仅进行模块化分解以提高可读性。

模块结构：
- core.py: 主Informer模型
- attention.py: ProbSparse注意力机制
- encoder.py: 编码器相关
- decoder.py: 解码器相关
- positional.py: 位置编码
- mask.py: 工具函数
"""

from .core import Informer
from .attention import ProbAttention
from .encoder import Encoder, EncoderLayer, AttentionLayer
from .decoder import Decoder, DecoderLayer
from .positional import PositionalEncoding

__all__ = [
    'Informer',
    'ProbAttention', 
    'Encoder',
    'EncoderLayer',
    'Decoder',
    'DecoderLayer',
    'PositionalEncoding',
    'AttentionLayer'
]