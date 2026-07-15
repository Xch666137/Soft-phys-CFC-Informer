#!/usr/bin/env python3
"""Analyze PV data statistics."""
import pandas as pd
import numpy as np

df = pd.read_csv('data_processed/multi_portfolio/portfolio_dataset_for_training.csv')
pv = df['p_pv_mw']

print('=== PV Data Statistics ===')
print(f'Total samples: {len(pv)}')
print(f'Mean: {pv.mean():.6f} MW')
print(f'Std: {pv.std():.6f} MW')
print(f'Min: {pv.min():.6f} MW')
print(f'Max: {pv.max():.6f} MW')
print(f'Zeros: {(pv == 0).sum()} ({(pv == 0).mean()*100:.2f}%)')
print(f'Negative: {(pv < 0).sum()} ({(pv < 0).mean()*100:.2f}%)')
print(f'Positive: {(pv > 0).sum()} ({(pv > 0).mean()*100:.2f}%)')

# Check by split
for split in ['train', 'val', 'test']:
    split_data = df[df['split'] == split]['p_pv_mw']
    print(f'\n{split}: n={len(split_data)}, mean={split_data.mean():.6f}, std={split_data.std():.6f}')

# Check by portfolio
print('\n=== PV by Portfolio (top 5) ===')
portfolio_stats = df.groupby('portfolio_id')['p_pv_mw'].agg(['mean', 'std', 'count'])
portfolio_stats = portfolio_stats.sort_values('count', ascending=False).head(5)
for pid, row in portfolio_stats.iterrows():
    print(f'Portfolio {pid}: mean={row["mean"]:.6f}, std={row["std"]:.6f}, n={int(row["count"])}')

# Check diurnal pattern
df['hour'] = pd.to_datetime(df['date']).dt.hour
hourly_pv = df.groupby('hour')['p_pv_mw'].mean()
print('\n=== PV Diurnal Pattern (mean by hour) ===')
for hour, mean_pv in hourly_pv.items():
    print(f'Hour {hour:2d}: {mean_pv:.6f} MW')