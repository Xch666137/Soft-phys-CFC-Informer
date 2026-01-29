"""
模型模块初始化

提供所有模型的统一导入接口，保持向后兼容性。
"""

from .informer import Informer

__all__ = [
    'Informer',
]