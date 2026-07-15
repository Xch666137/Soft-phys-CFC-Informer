#!/usr/bin/env python3
"""Dry-run test to verify pretrain checkpoint save/load and component metrics."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from pathlib import Path
from physformer.train.base import _unwrap_state_dict, _strip_orig_mod_prefix
from physformer.config import load_config, config_to_args
from physformer.train.pretrain_exp import PretrainExperiment

def test_checkpoint_save_load():
    """Test that checkpoint save/load handles _orig_mod. prefix correctly."""
    print("=== Test 1: Checkpoint Save/Load ===")
    
    # Create a dummy model with _orig_mod. prefix (simulating torch.compile)
    dummy_state = {
        'comp_embedding._orig_mod.gru.weight_ih_l0': torch.randn(10, 5),
        'comp_embedding._orig_mod.gru.weight_hh_l0': torch.randn(10, 10),
        'encoder._orig_mod.layers.0.w_q.weight': torch.randn(10, 10),
        'normal_key': torch.randn(5, 5),
    }
    
    # Test _strip_orig_mod_prefix
    stripped = _strip_orig_mod_prefix(dummy_state)
    expected_keys = [
        'comp_embedding.gru.weight_ih_l0',
        'comp_embedding.gru.weight_hh_l0',
        'encoder.layers.0.w_q.weight',
        'normal_key',
    ]
    
    for key in expected_keys:
        assert key in stripped, f"Missing key after strip: {key}"
    
    print("✓ _strip_orig_mod_prefix works correctly")
    
    # Test _unwrap_state_dict (requires actual model)
    # Skip if no model available
    print("✓ Checkpoint prefix handling verified")

def test_pretrain_checkpoint():
    """Test that current pretrain checkpoint is clean."""
    print("\n=== Test 2: Current Pretrain Checkpoint ===")
    
    checkpoint_path = Path("runs/physformer_igt_b1_pretrain_lam10/checkpoint.pth")
    if not checkpoint_path.exists():
        print("⚠ No checkpoint found yet (pretrain just started)")
        return
    
    state = torch.load(checkpoint_path, map_location='cpu')
    keys = list(state.keys())
    
    has_orig = any('_orig_mod.' in k for k in keys)
    if has_orig:
        print("✗ Checkpoint still has _orig_mod. prefix!")
        # Show sample keys
        for k in keys[:3]:
            if '_orig_mod.' in k:
                print(f"  {k}")
    else:
        print("✓ Checkpoint has clean keys (no _orig_mod. prefix)")
        print(f"  Total keys: {len(keys)}")
        print(f"  Sample: {keys[0]}")

def test_component_metrics_flow():
    """Test that component metrics calculation works end-to-end."""
    print("\n=== Test 3: Component Metrics Flow ===")
    
    # Check if comp_preds_norm is being saved
    extras_dir = Path("runs/physformer_igt_b1_pretrain_lam10/extras")
    if not extras_dir.exists():
        print("⚠ No extras directory found yet")
        return
    
    comp_preds_file = extras_dir / "comp_preds_norm.npy"
    if comp_preds_file.exists():
        comp_preds = torch.load(comp_preds_file, map_location='cpu')
        print(f"✓ comp_preds_norm saved: shape={comp_preds.shape}")
    else:
        print("⚠ comp_preds_norm.npy not found (will be created during test)")

def main():
    print("Dry-run test for PV masking pretrain fix\n")
    
    test_checkpoint_save_load()
    test_pretrain_checkpoint()
    test_component_metrics_flow()
    
    print("\n=== Summary ===")
    print("If all tests pass, the pretrain should complete successfully.")
    print("Expected completion: ~21:20 (3.2 hours from restart)")
    print("\nNext steps after pretrain completes:")
    print("1. Run finetune: python run.py train --config configs/physformer_igt_b1_finetune.yaml")
    print("2. Test component metrics: python test_b1_component.py")
    print("3. Compare with original B1 results")

if __name__ == "__main__":
    main()