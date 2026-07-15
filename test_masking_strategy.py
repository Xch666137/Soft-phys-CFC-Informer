#!/usr/bin/env python3
"""Test masking strategy to verify PV masking probability."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import random
from physformer.train.pretrain_exp import PretrainExperiment
from physformer.config import load_config, config_to_args

def test_masking_strategy():
    """Test the masking strategy to verify PV masking probability."""
    # Load a config to create the experiment
    config_path = "configs/physformer_igt_b1_pretrain.yaml"
    cfg = load_config(config_path)
    args = config_to_args(cfg)
    args.train_epochs = 0  # Skip training
    
    # Create experiment
    exp = PretrainExperiment(args)
    
    # Test masking strategy
    batch_size = 1000
    mask = exp._sample_mask_indices(batch_size)
    
    # Count PV masking
    pv_masked = mask[:, 1].sum().item()
    pv_mask_rate = pv_masked / batch_size
    
    # Count other components
    load_masked = mask[:, 0].sum().item()
    wind_masked = mask[:, 2].sum().item()
    batt_masked = mask[:, 3].sum().item()
    
    print("=== Masking Strategy Test ===")
    print(f"Batch size: {batch_size}")
    print(f"PV masked: {pv_masked} ({pv_mask_rate*100:.2f}%)")
    print(f"Load masked: {load_masked} ({load_masked/batch_size*100:.2f}%)")
    print(f"Wind masked: {wind_masked} ({wind_masked/batch_size*100:.2f}%)")
    print(f"Battery masked: {batt_masked} ({batt_masked/batch_size*100:.2f}%)")
    
    # Check if PV masking rate is ~50%
    expected_pv_rate = 0.5
    tolerance = 0.05
    if abs(pv_mask_rate - expected_pv_rate) < tolerance:
        print(f"\n✓ PV masking rate is approximately {expected_pv_rate*100:.0f}% (actual: {pv_mask_rate*100:.2f}%)")
    else:
        print(f"\n✗ PV masking rate is NOT {expected_pv_rate*100:.0f}% (actual: {pv_mask_rate*100:.2f}%)")
    
    # Check total masking rate per sample
    total_masked_per_sample = mask.sum(dim=1).float().mean().item()
    print(f"\nAverage masks per sample: {total_masked_per_sample:.2f}")
    
    return pv_mask_rate

if __name__ == "__main__":
    test_masking_strategy()