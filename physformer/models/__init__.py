"""
PhysFormer model registry.

All time-series forecasting models:
- PhysFormer: Physics-guided Transformer
- Informer: ProbSparse attention Transformer
- Autoformer: Autocorrelation Transformer
- LSTM / GRU: Recurrent baselines
- PINN: Physics-Informed Neural Network
- DLinear: Linear baseline
- PatchTST: Patch-based Transformer
- iTransformer: Inverted Transformer
"""

__version__ = "1.0.0"
__author__ = "XCH"

from .physformer import PhysFormer
from .causal_coupling import PhysicsGuidedCausalCoupling
from .informer import Informer
from .autoformer import Autoformer
from .lstm import LSTM
from .gru import GRU
from .pinn import PINN
from .dlinear import DLinear
from .patchtst import PatchTST
from .itransformer import iTransformer

MODEL_REGISTRY = {
    'PhysFormer': PhysFormer,
    'Informer': Informer,
    'Autoformer': Autoformer,
    'LSTM': LSTM,
    'GRU': GRU,
    'PINN': PINN,
    'DLinear': DLinear,
    'PatchTST': PatchTST,
    'iTransformer': iTransformer,
}


def get_model(model_name):
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return MODEL_REGISTRY[model_name]


__all__ = [
    'PhysFormer', 'Informer', 'Autoformer', 'LSTM', 'GRU',
    'PINN', 'DLinear', 'PatchTST', 'iTransformer',
    'PhysicsGuidedCausalCoupling',
    'get_model', 'MODEL_REGISTRY',
]
