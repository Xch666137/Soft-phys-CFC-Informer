#!/usr/bin/env python3
"""Test script to verify B1 component metrics calculation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physformer.train.physformer_exp import PhysFormerExperiment
from physformer.config import load_config, config_to_args
import torch
import numpy as np

def main():
    # Load B1 config
    config_path = "configs/physformer_igt_b1_finetune_s2025.yaml"
    cfg = load_config(config_path)
    args = config_to_args(cfg)
    args.train_epochs = 0  # Skip training
    
    # Create experiment
    exp = PhysFormerExperiment(args)
    
    # Run test
    preds_real, trues_real, metrics = exp.test(return_preds=True)
    
    # Print component metrics
    print("\n=== Component Metrics ===")
    for key, value in metrics.items():
        if key.startswith('component_') and isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"{key}: {value:.6e}")
    
    # Compare with A1
    print("\n=== Comparison with A1 ===")
    a1_metrics = {
        'component_load_mae': 0.005521804094314575,
        'component_pv_mae': 0.00481983320787549,
        'component_wind_mae': 0.0008289760444313288,
        'component_battery_power_mae': 0.002247890457510948,
    }
    
    for key in a1_metrics:
        if key in metrics:
            b1_val = metrics[key]
            a1_val = a1_metrics[key]
            diff = (b1_val - a1_val) / a1_val * 100
            print(f"{key}: B1={b1_val:.6e}, A1={a1_val:.6e}, diff={diff:+.2f}%")

if __name__ == "__main__":
    main()
