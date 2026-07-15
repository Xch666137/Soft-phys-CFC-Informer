#!/usr/bin/env python3
"""Check timestamp timezone in raw data."""
import pandas as pd
import numpy as np
from pathlib import Path

# Check a raw file
raw_dir = Path("data_raw/nextgen")
file = list(raw_dir.glob("*.csv"))[0]

print(f"=== {file.name} ===")
df = pd.read_csv(file)

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['original index'], unit='s')
df['hour'] = df['timestamp'].dt.hour
df['date'] = df['timestamp'].dt.date

# Check first few rows
print("\nFirst 5 rows:")
print(df[['original index', 'timestamp', 'hour', 'solar power (kW)']].head())

# Check diurnal pattern
print("\nDiurnal pattern (mean by hour):")
hourly = df.groupby('hour')['solar power (kW)'].mean()
for hour, mean_val in hourly.items():
    print(f"  Hour {hour:2d}: {mean_val:.6f} kW")

# Check date range
print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Timezone: {df['timestamp'].dt.tz}")

# Check if timestamps are UTC
print("\nSample timestamps:")
for i in range(5):
    ts = df['timestamp'].iloc[i]
    print(f"  {ts} (hour={ts.hour})")

# Check solar radiation pattern
# Solar should be highest at noon, lowest at midnight
# If timestamps are UTC, local time might be different
print("\n=== Interpretation ===")
print("If timestamps are UTC:")
print("  Hour 0 UTC = local time depends on timezone")
print("  For China (UTC+8): hour 0 UTC = 8am local")
print("  For US (UTC-5): hour 0 UTC = 7pm previous day local")
print("\nIf solar is highest at hour 0-5 UTC:")
print("  This could be noon in UTC+8 (China)")