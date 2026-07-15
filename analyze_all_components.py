#!/usr/bin/env python3
"""Analyze all component data statistics."""
import pandas as pd
import numpy as np

df = pd.read_csv('data_processed/multi_portfolio/portfolio_dataset_for_training.csv')

components = ['p_load_mw', 'p_pv_mw', 'p_wind_mw', 'p_battery_mw', 'e_battery_soc_mwh']

print('=== Component Data Statistics ===')
for comp in components:
    data = df[comp]
    print(f'\n{comp}:')
    print(f'  Mean: {data.mean():.6f} MW')
    print(f'  Std: {data.std():.6f} MW')
    print(f'  Min: {data.min():.6f} MW')
    print(f'  Max: {data.max():.6f} MW')
    print(f'  Zeros: {(data == 0).sum()} ({(data == 0).mean()*100:.2f}%)')

# Check net injection
df['net_injection'] = df['p_load_mw'] - df['p_pv_mw'] - df['p_wind_mw'] + df['p_battery_mw']
print('\n=== Net Injection ===')
print(f'Mean: {df["net_injection"].mean():.6f} MW')
print(f'Std: {df["net_injection"].std():.6f} MW')

# Check diurnal pattern for all components
df['hour'] = pd.to_datetime(df['date']).dt.hour
print('\n=== Diurnal Pattern (mean by hour) ===')
for comp in components:
    hourly = df.groupby('hour')[comp].mean()
    print(f'\n{comp}:')
    for hour, mean_val in hourly.items():
        print(f'  Hour {hour:2d}: {mean_val:.6f} MW')

# Check correlation between PV and irradiance
if 'irradiance' in df.columns:
    corr = df['p_pv_mw'].corr(df['irradiance'])
    print(f'\n=== PV-Irradiance Correlation: {corr:.4f} ===')
    
    # Check by hour
    print('\nPV-Irradiance Correlation by Hour:')
    for hour in range(24):
        hour_data = df[df['hour'] == hour]
        if len(hour_data) > 10:
            corr_hour = hour_data['p_pv_mw'].corr(hour_data['irradiance'])
            print(f'  Hour {hour:2d}: {corr_hour:.4f}')