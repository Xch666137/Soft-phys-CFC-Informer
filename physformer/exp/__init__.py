from .base import BaseExperiment, EarlyStopping, ForecastExperiment
from .exp_baseline import BaselineExperiment, Exp_Baselines


def __getattr__(name):
    if name == 'Exp_PhysFormer':
        from .exp_physformer import Exp_PhysFormer
        return Exp_PhysFormer
    raise AttributeError(name)

__all__ = [
    'BaseExperiment',
    'ForecastExperiment',
    'EarlyStopping',
    'BaselineExperiment',
    'Exp_Baselines',
    'Exp_PhysFormer',
]
