"""
PhysFormer — Physics-guided Transformer for VPP net injection forecasting.
"""

__version__ = "1.0.0"
__author__ = "XCH"

from .config import load_config, config_to_args
from .data import data_provider
from .train import (
    BaseExperiment, EarlyStopping, set_seed,
    PhysFormerExperiment, PretrainExperiment, BaselineExperiment,
    create_experiment, create_pretrain_experiment,
)
from .loss import PhysLoss, PhysAwareBaseLoss, PretrainLoss
from .metrics import compute_forecast_metrics, per_channel_mae
from .models import MODEL_REGISTRY, get_model

__all__ = [
    # Config
    "load_config", "config_to_args",
    # Data
    "data_provider",
    # Training
    "BaseExperiment", "EarlyStopping", "set_seed",
    "PhysFormerExperiment", "BaselineExperiment",
    "create_experiment",
    # Loss & metrics
    "PhysLoss", "PhysAwareBaseLoss", "PretrainLoss",
    "compute_forecast_metrics", "per_channel_mae",
    # Models
    "MODEL_REGISTRY", "get_model",
]
