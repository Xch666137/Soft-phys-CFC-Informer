"""
Soft-phys-CFC-Informer 模型包

顶层包初始化，重新导出 src 中的模型和工具。
"""

from .src import *

# 为了方便，直接重新导出常用模型
try:
    from .src.models import (
        PhysFormer, Informer, Autoformer, LSTM, GRU, PINN,
        get_model, MODEL_REGISTRY
    )
except ImportError as e:
    print(f"Warning: Could not import models from src: {e}")

__all__ = [
    'PhysFormer',
    'Informer',
    'Autoformer',
    'LSTM',
    'GRU',
    'PINN',
    'get_model',
    'MODEL_REGISTRY',
]
