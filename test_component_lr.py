#!/usr/bin/env python3
"""Test component-specific learning rate configuration."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physformer.train.physformer_exp import PhysFormerExperiment
from physformer.config import load_config, config_to_args
import json

def run_test(config_path):
    """Run test for a given config and return metrics."""
    cfg = load_config(config_path)
    args = config_to_args(cfg)
    args.train_epochs = 0  # Skip training
    
    exp = PhysFormerExperiment(args)
    preds_real, trues_real, metrics = exp.test(return_preds=True)
    return metrics

def main():
    config_path = "configs/physformer_igt_b1_finetune_component_lr.yaml"
    print(f"\n=== Testing component LR config ===")
    
    try:
        metrics = run_test(config_path)
        
        # Print component metrics
        print("Component metrics:")
        for key, value in metrics.items():
            if 'component' in key:
                print(f"  {key}: {value:.6e}")
        
        # Compare with B1 and A1
        print("\n=== Comparison with B1 and A1 ===")
        # B1 component metrics
        b1_metrics = {
            'component_load_mae': 2.995e-3,
            'component_pv_mae': 5.911e-3,
            'component_wind_mae': 5.604e-4,
            'component_battery_power_mae': 2.423e-3,
            'component_battery_soc_mae': 2.055e-2,
        }
        
        # A1 component metrics
        a1_metrics = {
            'component_load_mae': 5.522e-3,
            'component_pv_mae': 4.820e-3,
            'component_wind_mae': 8.290e-4,
            'component_battery_power_mae': 2.248e-3,
            'component_battery_soc_mae': 2.055e-2,
        }
        
        component_keys = [
            'component_load_mae', 'component_pv_mae', 'component_wind_mae',
            'component_battery_power_mae', 'component_battery_soc_mae'
        ]
        
        for key in component_keys:
            if key in metrics:
                comp_val = metrics[key]
                b1_val = b1_metrics.get(key, 0)
                a1_val = a1_metrics.get(key, 0)
                
                print(f"{key}:")
                print(f"  component_lr: {comp_val:.6e}")
                print(f"  B1: {b1_val:.6e}")
                print(f"  A1: {a1_val:.6e}")
                
                if b1_val > 0:
                    diff_b1 = (comp_val - b1_val) / b1_val * 100
                    print(f"  vs B1: {diff_b1:+.2f}%")
                if a1_val > 0:
                    diff_a1 = (comp_val - a1_val) / a1_val * 100
                    print(f"  vs A1: {diff_a1:+.2f}%")
        
        # Save results
        output_path = "results/component_lr_metrics.json"
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {output_path}")
        
    except Exception as e:
        print(f"Error testing component LR config: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()