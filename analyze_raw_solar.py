#!/usr/bin/env python3
"""Analyze raw solar data sign convention."""
import pandas as pd
import numpy as np
from pathlib import Path

# Check a few raw files
raw_dir = Path("data_raw/nextgen")
files = list(raw_dir.glob("*.csv"))[:3]  # Check first 3 files

for file in files:
    print(f"\n=== {file.name} ===")
    df = pd.read_csv(file)
    
    # Check solar power column
    solar_col = "solar power (kW)"
    if solar_col in df.columns:
        solar = pd.to_numeric(df[solar_col], errors="coerce").dropna()
        
        print(f"Total samples: {len(solar)}")
        print(f"Positive: {(solar > 0).sum()} ({(solar > 0).mean()*100:.2f}%)")
        print(f"Negative: {(solar < 0).sum()} ({(solar < 0).mean()*100:.2f}%)")
        print(f"Zeros: {(solar == 0).sum()} ({(solar == 0).mean()*100:.2f}%)")
        print(f"Mean: {solar.mean():.6f} kW")
        
        # Check diurnal pattern
        df['timestamp'] = pd.to_datetime(df['original index'], unit='s')
        df['hour'] = df['timestamp'].dt.hour
        
        print("\nDiurnal pattern (mean by hour):")
        hourly = df.groupby('hour')[solar_col].mean()
        for hour, mean_val in hourly.items():
            print(f"  Hour {hour:2d}: {mean_val:.6f} kW")
        
        # Determine sign convention
        positive_share = (solar > 0).mean()
        negative_share = (solar < 0).mean()
        convention = "positive_is_generation" if positive_share >= negative_share else "negative_is_generation"
        print(f"\nSign convention: {convention}")
        print(f"Positive share: {positive_share:.4f}")
        print(f"Negative share: {negative_share:.4f}")