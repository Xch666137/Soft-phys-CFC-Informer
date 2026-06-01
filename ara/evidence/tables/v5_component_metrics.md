# V5 Per-Component Metrics

- **Source**: `docs/research_directions.md` V5 component table
- **Claims**: C03, C06
- **Notes**: All MAE values in kW. V5 uses component-consistent residual (5 independent per-component corrections). The Battery Power improvement is dramatic because V4's scalar residual had pathological cross-contamination from Load errors.

| Component | V4 MAE (kW) | V4.1 MAE (kW) | V4.2 MAE (kW) | V5 MAE (kW) | Best Version | V5 vs V4 Δ |
|-----------|-------------|---------------|---------------|-------------|--------------|------------|
| Load | 14.707 | 2.044 | 2.093 | 2.069 | V4.1 | -85.9% |
| PV | 3.998 | 3.795 | 2.449 | 1.892 | **V5** | -52.7% |
| Wind | 0.825 | 0.459 | 0.355 | 0.313 | **V5** | -62.1% |
| Battery Power | 21.345 | 1.532 | 1.686 | 1.340 | **V5** | -93.7% |
| Battery SOC | 20.111 | 4.579 | 6.190 | 4.422 | V4.1 | -78.0% |

## Key Observations

1. **V5 achieves best-ever component metrics for PV, Wind, and Battery Power** — confirming component-consistent residual works.
2. **V4 Battery Power = 21.345 kW was pathological** — scalar residual could not disentangle battery from load errors.
3. **Load improvement from V4→V4.1/V5 is largely from architecture (not component-consistent residual)** — the calendar + load proxy in V4.1 already captured most of the gain.
4. **Battery SOC is the one component where V5 does not lead** — V4.1 (4.579) < V5 (4.422), though V5 SOC constraint is perfectly satisfied.
