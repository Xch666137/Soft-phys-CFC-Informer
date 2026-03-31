# Forecasting Layer Schema

This project now supports a forecasting corpus that is driven by configuration
instead of a fixed column order.

## Supported task modes

- `component_multitask`
  - Targets: component-level power series such as `load_mw`, `pv_mw`, `wind_mw`
  - Intended for the current PhysFormer architecture
- `net_injection`
  - Target: portfolio-level power such as `p_vpp_mw`
  - Intended for thesis-scale VPP forecasting experiments and network mapping

## Canonical forecasting table

Minimum columns for the thesis mainline:

- `date`
- `portfolio_id`
- `region_id`
- `p_vpp_mw`
- `temperature`
- `irradiance`
- `wind_speed`

Optional auxiliary columns:

- `p_load_mw`
- `p_pv_mw`
- `p_wind_mw`
- `soc`
- `calendar_flag`
- `holiday_flag`

## Sign convention

- `p_vpp_mw` is defined as the portfolio-level **net load**.
- `p_vpp_mw > 0`: the VPP is importing power from the grid.
- `p_vpp_mw < 0`: the VPP is exporting net power to the grid.
- Network validation should map this signed portfolio quantity into load/sgen injections.

## Current model support

- `PhysFormer`
  - Supports only `component_multitask`
  - Requires exactly 3 targets and 3 weather covariates
- Baseline models
  - Can use either `component_multitask` or `net_injection`
  - Input/output dimensions are derived from `target_cols` and `covariate_cols`

## Mapping to network validation

Forecasting outputs should be exported at portfolio granularity:

- `date`
- `portfolio_id`
- `pred_p_vpp_mw`

These outputs are then joined with `templates/network_mapping.csv` to inject
portfolio forecasts into the chosen SimBench/pandapower network.
