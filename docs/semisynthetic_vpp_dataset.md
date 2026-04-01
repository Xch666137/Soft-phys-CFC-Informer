# Semi-Synthetic VPP Dataset

This thesis branch uses a dedicated data-construction workflow for a **semi-synthetic VPP research dataset**.

It is not a same-origin operational VPP dataset. The field origins are intentionally mixed and must be documented as such:

- `load_kw`, `solar_kw`, `battery_power_kw`, `battery_soc_kwh`: real household traces from NextGen
- `air_temperature_2m`, `wind_speed_10m`, `cloud_cover`, `surface_solar_radiation`: external reconstructed weather from ERA5 hourly
- `synthetic_wind_cf`, `synthetic_wind_kw`: synthetic wind variables generated from a Rye-derived wind template

## Output Products

The build chain writes these files under `data_processed/` by default:

- `nextgen_vpp_household_15min.csv`
- `nextgen_vpp_aggregate_15min.csv`
- `nextgen_vpp_metadata.json`
- `wind_template_lookup.csv`

The aggregate table is designed for the thesis forecasting task:

- `agg_load_kw`
- `agg_solar_kw`
- `agg_battery_power_kw`
- `agg_battery_soc_kwh`
- `synthetic_wind_kw`
- `agg_net_load_kw`

## Data Sources

### NextGen

Public Zenodo record used as the real household backbone:

- Record: [NextGen on Zenodo](https://zenodo.org/records/14885589)

Download command:

```bash
python tools/fetch_nextgen.py --output-dir data_raw/nextgen
```

Windows batch:

```bat
set FETCH_RYE=0
set FETCH_ERA5_ACT=0
set FETCH_ERA5_RYE=0
set BUILD_DATASET=0
scripts\build_semisynthetic_vpp_dataset.bat
```

### Rye

Public Zenodo record used only as the wind-generation template source:

- Record: [Rye microgrid on Zenodo](https://zenodo.org/records/4448894)

Download command:

```bash
python tools/fetch_rye.py --output-dir data_raw/rye
```

The default workflow only needs `rye_generation_and_load.csv`. `met_data.h5` is optional and is not required by the current builder.

### ERA5

ERA5 hourly weather is required for:

- `act_canberra`: the shared ACT weather used in the final dataset
- `rye_template`: the weather series paired with Rye wind output during template fitting

This workflow uses the Copernicus CDS API and prefers the dedicated single-point time-series catalogue entry:

- [ERA5 hourly time-series data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries?tab=overview)

CDS credentials must be configured on the machine that performs the download.

ACT weather:

```bash
python tools/fetch_era5.py \
  --site-key act_canberra \
  --start-date 2018-01-01 \
  --end-date 2018-12-31 \
  --output-csv data_raw/era5/act_canberra_hourly.csv
```

Rye template weather:

```bash
python tools/fetch_era5.py \
  --site-key rye_template \
  --start-date 2020-01-01 \
  --end-date 2020-12-31 \
  --output-csv data_raw/era5/rye_template_hourly.csv
```

Notes:

- Weather remains hourly and is broadcast to the 15-minute grid during dataset construction.
- Do not interpolate ERA5 to create artificial high-frequency weather variation.
- CDS credentials or tokens must not be committed to the repository.
- The fetcher targets the single-point time-series ERA5 product first and falls back across known request shapes if CDS changes the request schema.

## Build Command

Once the raw files exist, build the semi-synthetic dataset:

```bash
python tools/build_semisynthetic_vpp.py \
  --nextgen-dir data_raw/nextgen \
  --act-weather-csv data_raw/era5/act_canberra_hourly.csv \
  --rye-generation-csv data_raw/rye/rye_generation_and_load.csv \
  --rye-weather-csv data_raw/era5/rye_template_hourly.csv \
  --output-dir data_processed
```

Windows one-click wrapper:

```bat
scripts\build_semisynthetic_vpp_dataset.bat
```

Useful Windows overrides:

```bat
set ACT_START_DATE=2018-01-01
set ACT_END_DATE=2018-12-31
set RYE_START_DATE=2020-01-01
set RYE_END_DATE=2020-12-31
set WIND_PENETRATION_TARGET=0.15
scripts\build_semisynthetic_vpp_dataset.bat
```

Preview commands without downloading or building:

```bat
set DRY_RUN=1
scripts\build_semisynthetic_vpp_dataset.bat
```

## Modeling Assumptions

- The dataset classification is `semi-synthetic VPP research dataset`.
- Shared ACT weather is used for all households because NextGen does not expose precise household coordinates.
- Rye is used as a wind-output template only. Its original timestamps and seasonal labels are not transferred into the ACT dataset.
- The wind template uses `wind_speed_10m` and `hour` only. `month` is intentionally excluded to avoid wrong-season transfer across hemispheres.
- Synthetic wind is scaled to a fixed annual energy penetration target of `0.15` relative to aggregate load energy.

## Validation Checklist

- Time axis is strictly monotonic in UTC.
- No duplicate timestamps or daylight-saving discontinuities remain after processing.
- `battery_soc_kwh` stays non-negative.
- Metadata records the inferred `battery_power_kw` sign convention.
- `synthetic_wind_cf` remains within the documented range.
- `nextgen_vpp_metadata.json` must explicitly state that this is not a real same-origin VPP operational dataset.
