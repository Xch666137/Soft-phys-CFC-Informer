#!/usr/bin/env python3
"""Run B1 tests for all seeds and collect component metrics."""
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
    seeds = [2025, 2026, 2027]
    results = {}
    
    for seed in seeds:
        config_path = f"configs/physformer_igt_b1_finetune_s{seed}.yaml"
        print(f"\n=== Testing B1 seed {seed} ===")
        
        try:
            metrics = run_test(config_path)
            results[seed] = metrics
            
            # Print component metrics
            print("Component metrics:")
            for key, value in metrics.items():
                if key.startswith('component_') and isinstance(value, (int, float)) and not isinstance(value, bool):
                    print(f"  {key}: {value:.6e}")
                    
        except Exception as e:
            print(f"Error testing seed {seed}: {e}")
            results[seed] = {"error": str(e)}
    
    # Save results
    output_path = "results/b1_component_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n=== Summary ===")
    component_keys = [
        'component_load_mae', 'component_pv_mae', 'component_wind_mae',
        'component_battery_power_mae'
    ]
    
    for key in component_keys:
        values = []
        for seed in seeds:
            if seed in results and key in results[seed]:
                values.append(results[seed][key])
        if values:
            mean_val = sum(values) / len(values)
            print(f"{key}: {mean_val:.6e} (mean of {len(values)} seeds)")

if __name__ == "__main__":
    main()
