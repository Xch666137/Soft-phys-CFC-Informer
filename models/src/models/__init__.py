"""
Soft-phys-CFC-Informer 模型包

包含所有时间序列预测模型：
- PhysFormer: 物理引导的Transformer量化预测模型
- Informer: 基于ProbSparse注意力的Transformer模型
- Autoformer: 基于自相关机制的Transformer模型
- LSTM: 长短期记忆网络
- GRU: 门控循环单元
- PINN: 物理信息神经网络

所有模型都支持VPP负载预测任务。
"""

__version__ = "1.0.0"
__author__ = "XCH"

# 导入所有模型类
from .PhysFormer import PhysFormer, PhysicsGuidedCausalCoupling
from .informer import Informer, ProbAttention, Encoder, EncoderLayer, Decoder, DecoderLayer, PositionalEncoding, AttentionLayer
from .Autoformer import Autoformer
from .LSTM import LSTM
from .GRU import GRU
from .PINN import PINN
from .DLinear import DLinear
from .PatchTST import PatchTST

# 为方便使用，定义模型名称到类的映射
MODEL_REGISTRY = {
    'PhysFormer': PhysFormer,
    'Informer': Informer,
    'Autoformer': Autoformer,
    'LSTM': LSTM,
    'GRU': GRU,
    'PINN': PINN,
    'DLinear': DLinear,
    'PatchTST': PatchTST,
}

def get_model(model_name):
    """
    根据模型名称获取模型类

    Args:
        model_name (str): 模型名称，如 'PhysFormer', 'Informer' 等

    Returns:
        class: 对应的模型类

    Raises:
        ValueError: 如果模型名称不存在
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"未知模型 '{model_name}'。可用模型: {available}")
    return MODEL_REGISTRY[model_name]

__all__ = [
    # 主模型类
    'PhysFormer',
    'Informer',
    'Autoformer',
    'LSTM',
    'GRU',
    'PINN',
    'DLinear',
    'PatchTST',

    # PhysFormer 组件
    'PhysicsGuidedCausalCoupling',

    # Informer 组件
    'ProbAttention',
    'Encoder',
    'EncoderLayer',
    'Decoder',
    'DecoderLayer',
    'PositionalEncoding',
    'AttentionLayer',

    # 工具函数
    'get_model',
    'MODEL_REGISTRY',
]