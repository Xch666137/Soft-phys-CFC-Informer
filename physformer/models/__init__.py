"""
PhysFormer model registry.

All time-series forecasting models:
- PhysFormer: portfolio-level gray-box Transformer
- Informer: ProbSparse attention Transformer
- Autoformer: Autocorrelation Transformer
- LSTM / GRU: Recurrent baselines
- PINN: Physics-Informed Neural Network
- DLinear: Linear baseline
- PatchTST: Patch-based Transformer
- iTransformer: Inverted Transformer
- TiDE: Dense encoder-decoder baseline
- TimeXer: Exogenous-aware Transformer baseline
- TFT: Temporal Fusion Transformer baseline
"""

__version__ = "1.0.0"
__author__ = "XCH"

from .physformer import PhysFormer, PhysFormeriGT
from .informer import Informer
from .autoformer import Autoformer
from .lstm import LSTM
from .gru import GRU
from .pinn import PINN
from .dlinear import DLinear
from .patchtst import PatchTST
from .itransformer import iTransformer
from .tide import TiDE
from .timexer import TimeXer
from .persistence import Persistence
from .tft import TFT

MODEL_REGISTRY = {
    'PhysFormer': PhysFormer,
    'PhysFormer-iGT': PhysFormeriGT,
    'Informer': Informer,
    'Autoformer': Autoformer,
    'LSTM': LSTM,
    'GRU': GRU,
    'PINN': PINN,
    'DLinear': DLinear,
    'PatchTST': PatchTST,
    'iTransformer': iTransformer,
    'TiDE': TiDE,
    'TimeXer': TimeXer,
    'TFT': TFT,
    'Persistence': Persistence,
}


def get_model(model_name):
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return MODEL_REGISTRY[model_name]


__all__ = [
    'PhysFormer', 'PhysFormeriGT', 'Informer', 'Autoformer', 'LSTM', 'GRU',
    'PINN', 'DLinear', 'PatchTST', 'iTransformer', 'TiDE', 'TimeXer', 'TFT',
    'get_model', 'MODEL_REGISTRY',
]


def __getattr__(name: str):
    _PHYSFORMER_MODULES = {"physical_layer", "conditioning", "temporal_decoder", "flatten_head"}
    if name in _PHYSFORMER_MODULES:
        from importlib import import_module
        return import_module(f".physformer.{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
