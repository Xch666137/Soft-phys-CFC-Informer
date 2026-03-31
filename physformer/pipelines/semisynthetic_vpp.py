import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


NEXTGEN_RECORD_ID = "14885589"
RYE_RECORD_ID = "4448894"


@dataclass(frozen=True)
class SiteSpec:
    name: str
    lat: float
    lon: float
    timezone: str


SITE_REGISTRY = {
    "act_canberra": SiteSpec(
        name="ACT / Canberra",
        lat=-35.2809,
        lon=149.1300,
        timezone="Australia/Sydney",
    ),
    "rye_template": SiteSpec(
        name="Rye microgrid",
        lat=63.41,
        lon=10.11,
        timezone="Europe/Oslo",
    ),
}


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _download_file(url: str, output_path: Path, chunk_size: int = 1024 * 1024):
    import requests

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def fetch_zenodo_record(record_id: str, output_dir: str | Path, include_names: Iterable[str] | None = None):
    import requests

    output_dir = _ensure_dir(output_dir)
    api_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(api_url, timeout=120)
    response.raise_for_status()
    payload = response.json()

    include_set = set(include_names or [])
    downloaded = []

    for file_info in payload.get("files", []):
        file_name = file_info.get("key")
        if include_set and file_name not in include_set:
            continue

        links = file_info.get("links", {})
        file_url = links.get("self") or links.get("content")
        if not file_url:
            continue

        destination = output_dir / file_name
        if not destination.exists():
            _download_file(file_url, destination)
        downloaded.append(str(destination))

    return downloaded


def fetch_nextgen(output_dir: str | Path):
    return fetch_zenodo_record(NEXTGEN_RECORD_ID, output_dir)


def fetch_rye(output_dir: str | Path, include_weather_h5: bool = False):
    include = ["rye_generation_and_load.csv"]
    if include_weather_h5:
        include.append("met_data.h5")
    return fetch_zenodo_record(RYE_RECORD_ID, output_dir, include_names=include)


def fetch_era5_cds(
    output_csv: str | Path,
    start_date: str,
    end_date: str,
    site_key: str,
    raw_download_path: str | Path | None = None,
):
    try:
        import cdsapi
        import xarray as xr
    except Exception as exc:
        raise RuntimeError(
            "ERA5 fetching requires cdsapi and xarray. Install the project requirements first."
        ) from exc

    if site_key not in SITE_REGISTRY:
        raise ValueError(f"Unknown site_key '{site_key}'. Available: {sorted(SITE_REGISTRY)}")

    site = SITE_REGISTRY[site_key]
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    raw_path = Path(raw_download_path) if raw_download_path else output_csv.with_suffix(".csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    variables = [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "total_cloud_cover",
        "surface_solar_radiation_downwards",
    ]
    client = cdsapi.Client()

    request_variants = [
        (
            "reanalysis-era5-single-levels-timeseries",
            {
                "variable": variables,
                "location": {"latitude": site.lat, "longitude": site.lon},
                "date": f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
                "data_format": "csv",
            },
        ),
        (
            "reanalysis-era5-single-levels-timeseries",
            {
                "variable": variables,
                "latitude": site.lat,
                "longitude": site.lon,
                "date": f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
                "data_format": "csv",
            },
        ),
        (
            "reanalysis-era5-single-levels-timeseries",
            {
                "variable": variables,
                "location": {"latitude": site.lat, "longitude": site.lon},
                "date": f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
                "data_format": "netcdf",
            },
        ),
        (
            "reanalysis-era5-single-levels-timeseries",
            {
                "variable": variables,
                "latitude": site.lat,
                "longitude": site.lon,
                "date": f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
                "data_format": "netcdf",
            },
        ),
        (
            "reanalysis-era5-single-levels-timeseries",
            {
                "variable": variables,
                "location": {"latitude": site.lat, "longitude": site.lon},
                "year": [f"{y:04d}" for y in range(start.year, end.year + 1)],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "data_format": "netcdf",
            },
        ),
    ]

    last_error = None
    for dataset_name, request in request_variants:
        try:
            client.retrieve(dataset_name, request, str(raw_path))
            last_error = None
            break
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise RuntimeError(
            "ERA5 time-series retrieval failed for all known request schemas. "
            "The CDS time-series API shape may have changed."
        ) from last_error

    if raw_path.suffix.lower() == ".csv":
        if zipfile.is_zipfile(raw_path):
            with zipfile.ZipFile(raw_path) as zf:
                members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
                if not members:
                    raise ValueError("ERA5 archive did not contain a CSV member.")
                with zf.open(members[0]) as member:
                    raw_csv = pd.read_csv(member)
        else:
            raw_csv = None
            last_csv_error = None
            for encoding in (None, "utf-8", "utf-8-sig", "utf-16", "latin-1"):
                try:
                    raw_csv = pd.read_csv(raw_path, encoding=encoding) if encoding else pd.read_csv(raw_path)
                    break
                except Exception as exc:
                    last_csv_error = exc
            if raw_csv is None:
                raise RuntimeError("Unable to parse ERA5 CSV output with known encodings.") from last_csv_error

        rename_candidates = {
            "date": "date",
            "time": "date",
            "valid_time": "date",
            "2m_temperature": "air_temperature_2m",
            "temperature_2m": "air_temperature_2m",
            "t2m": "air_temperature_2m",
            "10m_u_component_of_wind": "u10",
            "u10": "u10",
            "10m_v_component_of_wind": "v10",
            "v10": "v10",
            "total_cloud_cover": "cloud_cover",
            "tcc": "cloud_cover",
            "surface_solar_radiation_downwards": "surface_solar_radiation",
            "ssrd": "surface_solar_radiation",
        }
        normalized = {}
        for col in raw_csv.columns:
            key = col.strip().lower()
            if key in rename_candidates:
                normalized[col] = rename_candidates[key]
        raw_csv = raw_csv.rename(columns=normalized)

        if "date" not in raw_csv.columns:
            raise ValueError(f"ERA5 CSV output did not contain a recognizable time column: {list(raw_csv.columns)}")

        raw_csv["date"] = pd.to_datetime(raw_csv["date"], utc=True)
        if "u10" in raw_csv.columns and "v10" in raw_csv.columns:
            raw_csv["wind_speed_10m"] = np.sqrt(
                raw_csv["u10"].astype(float) ** 2 + raw_csv["v10"].astype(float) ** 2
            )
        elif "wind_speed_10m" not in raw_csv.columns:
            raise ValueError("ERA5 CSV output did not contain u10/v10 or wind_speed_10m.")

        if "cloud_cover" not in raw_csv.columns:
            raw_csv["cloud_cover"] = np.nan

        if "air_temperature_2m" not in raw_csv.columns:
            raise ValueError(
                f"ERA5 CSV output did not contain a recognizable temperature column. Columns: {list(raw_csv.columns)}"
            )

        weather = pd.DataFrame({
            "date": raw_csv["date"],
            "air_temperature_2m": raw_csv["air_temperature_2m"].astype(float) - 273.15,
            "wind_speed_10m": raw_csv["wind_speed_10m"].astype(float),
            "cloud_cover": np.clip(raw_csv["cloud_cover"].astype(float), 0.0, 1.0),
            "surface_solar_radiation": np.maximum(raw_csv["surface_solar_radiation"].astype(float) / 3600.0, 0.0),
        })
    else:
        dataset = xr.open_dataset(raw_path)
        if "latitude" in dataset.dims:
            dataset = dataset.isel(latitude=0)
        if "longitude" in dataset.dims:
            dataset = dataset.isel(longitude=0)

        time_index = pd.to_datetime(
            dataset["valid_time"].values if "valid_time" in dataset else dataset["time"].values,
            utc=True,
        )
        weather = pd.DataFrame({
            "date": time_index,
            "air_temperature_2m": np.asarray(dataset["t2m"].values, dtype=np.float64) - 273.15,
            "wind_speed_10m": np.sqrt(
                np.asarray(dataset["u10"].values, dtype=np.float64) ** 2
                + np.asarray(dataset["v10"].values, dtype=np.float64) ** 2
            ),
            "cloud_cover": np.clip(np.asarray(dataset["tcc"].values, dtype=np.float64), 0.0, 1.0),
            "surface_solar_radiation": np.maximum(
                np.asarray(dataset["ssrd"].values, dtype=np.float64) / 3600.0,
                0.0,
            ),
        })
        dataset.close()

    weather = weather[(weather["date"] >= start.tz_localize("UTC")) & (weather["date"] < (end + pd.Timedelta(days=1)).tz_localize("UTC"))]
    weather = weather.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    weather.to_csv(output_csv, index=False)
    return output_csv


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _parse_timestamp_series(series: pd.Series, timezone: str):
    timestamps = pd.to_datetime(series)
    if getattr(timestamps.dt, "tz", None) is None:
        timestamps = timestamps.dt.tz_localize(timezone, ambiguous="infer", nonexistent="shift_forward")
    else:
        timestamps = timestamps.dt.tz_convert(timezone)
    return timestamps.dt.tz_convert("UTC")


def _standardize_nextgen_columns(df: pd.DataFrame):
    normalized = {_normalize_name(c): c for c in df.columns}
    required_map = {
        "index": "date",
        "loadpowerkw": "load_kw",
        "solarpowerkw": "solar_kw",
        "batterypowerkw": "battery_power_kw",
        "batterysockwh": "battery_soc_kwh",
        "solarcapacitykw": "solar_capacity_kw",
        "batterycapacitykwh": "battery_capacity_kwh",
        "batterypeakpowerkw": "battery_peak_power_kw",
    }

    missing = [raw for raw in required_map if raw not in normalized]
    if missing:
        raise ValueError(f"NextGen CSV missing expected columns: {missing}")

    out = df[[normalized[k] for k in required_map]].rename(columns={normalized[k]: v for k, v in required_map.items()})
    return out


def load_nextgen_households(input_dir: str | Path, source_timezone: str = "Australia/Sydney"):
    input_dir = Path(input_dir)
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    frames = []
    for csv_path in csv_files:
        raw = pd.read_csv(csv_path)
        df = _standardize_nextgen_columns(raw)
        df["date"] = _parse_timestamp_series(df["date"], source_timezone)
        df["household_id"] = csv_path.stem
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["household_id", "date"]).reset_index(drop=True)


def resample_nextgen_to_15min(nextgen_df: pd.DataFrame):
    frames = []
    power_cols = ["load_kw", "solar_kw", "battery_power_kw"]
    state_cols = ["battery_soc_kwh"]
    static_cols = ["solar_capacity_kw", "battery_capacity_kwh", "battery_peak_power_kw"]

    for household_id, group in nextgen_df.groupby("household_id", sort=False):
        g = group.set_index("date").sort_index()
        power = g[power_cols].resample("15min").mean()
        state = g[state_cols].resample("15min").last().ffill()
        static = g[static_cols].resample("15min").last().ffill()
        merged = pd.concat([power, state, static], axis=1).reset_index()
        merged["household_id"] = household_id
        frames.append(merged)

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["household_id", "date"]).reset_index(drop=True)


def audit_battery_power_sign(household_df: pd.DataFrame):
    working = household_df[["date", "household_id", "battery_power_kw", "battery_soc_kwh"]].copy()
    working = working.sort_values(["household_id", "date"])
    working["soc_delta_kwh"] = working.groupby("household_id")["battery_soc_kwh"].diff()
    working = working.dropna(subset=["soc_delta_kwh", "battery_power_kw"])

    if working.empty:
        return {
            "battery_power_sign_convention": "unknown",
            "correlation_with_soc_delta": None,
            "inference_rule": "Not enough valid samples to audit sign.",
        }

    corr = float(working["battery_power_kw"].corr(working["soc_delta_kwh"]))
    convention = "positive_is_charging" if corr >= 0 else "positive_is_discharging"
    return {
        "battery_power_sign_convention": convention,
        "correlation_with_soc_delta": corr,
        "inference_rule": "Sign inferred from correlation between battery_power_kw and successive battery_soc_kwh differences.",
    }


def load_standardized_weather_csv(weather_csv: str | Path):
    weather = pd.read_csv(weather_csv)
    required = [
        "date",
        "air_temperature_2m",
        "wind_speed_10m",
        "cloud_cover",
        "surface_solar_radiation",
    ]
    missing = [c for c in required if c not in weather.columns]
    if missing:
        raise ValueError(f"Weather CSV missing columns: {missing}")

    weather["date"] = pd.to_datetime(weather["date"], utc=True)
    weather = weather.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return weather


def broadcast_weather_to_15min(weather_hourly: pd.DataFrame, full_index: pd.DatetimeIndex):
    weather = weather_hourly.set_index("date").reindex(full_index.floor("H"), method="ffill")
    weather = weather.reset_index().rename(columns={"index": "date"})
    weather["date"] = pd.to_datetime(weather["date"], utc=True)
    return weather


def _standardize_rye_generation_columns(df: pd.DataFrame):
    normalized = {_normalize_name(c): c for c in df.columns}
    time_col = None
    for candidate in ["time", "timestamp", "date", "datetime", "utc"]:
        if candidate in normalized:
            time_col = normalized[candidate]
            break
    if time_col is None:
        first_col = df.columns[0]
        time_col = first_col

    candidates = {
        "consumption": "consumption_kw",
        "solar": "solar_kw",
        "wind": "wind_kw",
    }
    missing = [raw for raw in candidates if raw not in normalized]
    if missing:
        raise ValueError(f"Rye generation CSV missing expected columns: {missing}")

    out = df[[time_col, normalized["consumption"], normalized["solar"], normalized["wind"]]].rename(columns={
        time_col: "date",
        normalized["consumption"]: "consumption_kw",
        normalized["solar"]: "solar_kw",
        normalized["wind"]: "wind_kw",
    })
    out["date"] = pd.to_datetime(out["date"], utc=True)
    return out.sort_values("date").reset_index(drop=True)


def load_rye_generation_csv(csv_path: str | Path):
    return _standardize_rye_generation_columns(pd.read_csv(csv_path))


def fit_hourly_wind_template(rye_generation: pd.DataFrame, rye_weather_hourly: pd.DataFrame):
    rye = rye_generation.copy()
    weather = rye_weather_hourly.copy()
    rye["date"] = pd.to_datetime(rye["date"], utc=True).dt.floor("H")
    weather["date"] = pd.to_datetime(weather["date"], utc=True).dt.floor("H")

    merged = rye.merge(weather[["date", "wind_speed_10m"]], on="date", how="inner")
    if merged.empty:
        raise ValueError("Rye generation and hourly weather do not overlap.")

    rated_kw = float(np.quantile(np.maximum(merged["wind_kw"].to_numpy(dtype=float), 0.0), 0.995))
    rated_kw = rated_kw if rated_kw > 0 else float(np.maximum(merged["wind_kw"].max(), 1.0))
    merged["wind_cf"] = np.clip(merged["wind_kw"] / rated_kw, 0.0, 1.0)
    merged["hour"] = pd.to_datetime(merged["date"], utc=True).dt.hour

    global_model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    global_model.fit(merged["wind_speed_10m"].to_numpy(dtype=float), merged["wind_cf"].to_numpy(dtype=float))

    models = {}
    hourly_sample_counts = {}
    for hour in range(24):
        subset = merged.loc[merged["hour"] == hour]
        hourly_sample_counts[str(hour)] = int(len(subset))
        if len(subset) >= 10 and subset["wind_speed_10m"].nunique() >= 3:
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(subset["wind_speed_10m"].to_numpy(dtype=float), subset["wind_cf"].to_numpy(dtype=float))
            models[hour] = model
        else:
            models[hour] = global_model

    report = {
        "rated_wind_kw_proxy": rated_kw,
        "hourly_sample_counts": hourly_sample_counts,
        "fit_method": "hourly isotonic regression on wind_speed_10m -> wind_cf with global fallback",
    }
    return models, report


def apply_hourly_wind_template(weather_hourly: pd.DataFrame, models: dict[int, IsotonicRegression]):
    weather = weather_hourly.copy()
    weather["date"] = pd.to_datetime(weather["date"], utc=True)
    weather["hour"] = weather["date"].dt.hour
    weather["synthetic_wind_cf"] = 0.0

    for hour in range(24):
        mask = weather["hour"] == hour
        if not mask.any():
            continue
        speeds = weather.loc[mask, "wind_speed_10m"].to_numpy(dtype=float)
        weather.loc[mask, "synthetic_wind_cf"] = np.clip(models[hour].predict(speeds), 0.0, 1.0)

    return weather.drop(columns=["hour"])


def scale_synthetic_wind_to_penetration(
    weather_15min: pd.DataFrame,
    aggregate_load_15min: pd.DataFrame,
    target_penetration: float = 0.15,
):
    out = weather_15min.copy()
    load_energy_kwh = float(aggregate_load_15min["agg_load_kw"].sum() * 0.25)
    cf_energy = float(out["synthetic_wind_cf"].sum() * 0.25)
    if cf_energy <= 0:
        raise ValueError("Synthetic wind template produced zero yearly energy; cannot scale to penetration target.")

    installed_capacity_kw = target_penetration * load_energy_kwh / cf_energy
    out["synthetic_wind_kw"] = out["synthetic_wind_cf"] * installed_capacity_kw

    scaling = {
        "wind_penetration_target": target_penetration,
        "aggregate_load_energy_kwh": load_energy_kwh,
        "synthetic_wind_cf_energy_proxy_kwh": cf_energy,
        "synthetic_wind_capacity_kw": installed_capacity_kw,
        "synthetic_wind_energy_kwh": float(out["synthetic_wind_kw"].sum() * 0.25),
    }
    return out, scaling


def build_aggregate_table(
    household_df_15min: pd.DataFrame,
    shared_weather_15min: pd.DataFrame,
    battery_sign_audit: dict,
):
    agg = household_df_15min.groupby("date", as_index=False).agg({
        "load_kw": "sum",
        "solar_kw": "sum",
        "battery_power_kw": "sum",
        "battery_soc_kwh": "sum",
    }).rename(columns={
        "load_kw": "agg_load_kw",
        "solar_kw": "agg_solar_kw",
        "battery_power_kw": "agg_battery_power_kw",
        "battery_soc_kwh": "agg_battery_soc_kwh",
    })

    merged = agg.merge(shared_weather_15min, on="date", how="left")

    sign_convention = battery_sign_audit["battery_power_sign_convention"]
    if sign_convention == "positive_is_discharging":
        merged["agg_net_load_kw"] = (
            merged["agg_load_kw"] - merged["agg_solar_kw"] - merged["synthetic_wind_kw"] - merged["agg_battery_power_kw"]
        )
        formula = "agg_load_kw - agg_solar_kw - synthetic_wind_kw - agg_battery_power_kw"
    else:
        merged["agg_net_load_kw"] = (
            merged["agg_load_kw"] - merged["agg_solar_kw"] - merged["synthetic_wind_kw"] + merged["agg_battery_power_kw"]
        )
        formula = "agg_load_kw - agg_solar_kw - synthetic_wind_kw + agg_battery_power_kw"

    return merged, formula


def build_semisynthetic_vpp_dataset(
    nextgen_dir: str | Path,
    act_weather_csv: str | Path,
    rye_generation_csv: str | Path,
    rye_weather_csv: str | Path,
    output_dir: str | Path,
    target_penetration: float = 0.15,
    source_timezone: str = "Australia/Sydney",
):
    output_dir = _ensure_dir(output_dir)
    nextgen = load_nextgen_households(nextgen_dir, source_timezone=source_timezone)
    household_15 = resample_nextgen_to_15min(nextgen)
    battery_audit = audit_battery_power_sign(household_15)

    act_weather_hourly = load_standardized_weather_csv(act_weather_csv)
    rye_generation = load_rye_generation_csv(rye_generation_csv)
    rye_weather_hourly = load_standardized_weather_csv(rye_weather_csv)

    full_index = pd.date_range(
        start=household_15["date"].min(),
        end=household_15["date"].max(),
        freq="15min",
        tz="UTC",
    )

    household_15 = household_15.set_index("date").groupby("household_id", group_keys=False).apply(
        lambda df: df.reindex(full_index).assign(household_id=df["household_id"].iloc[0]).ffill()
    ).reset_index().rename(columns={"index": "date"})

    for static_col in ["solar_capacity_kw", "battery_capacity_kwh", "battery_peak_power_kw"]:
        household_15[static_col] = household_15.groupby("household_id")[static_col].ffill().bfill()

    household_15[["load_kw", "solar_kw", "battery_power_kw"]] = household_15[
        ["load_kw", "solar_kw", "battery_power_kw"]
    ].fillna(0.0)
    household_15["battery_soc_kwh"] = household_15.groupby("household_id")["battery_soc_kwh"].ffill().bfill()

    base_agg = household_15.groupby("date", as_index=False)["load_kw"].sum().rename(columns={"load_kw": "agg_load_kw"})
    act_weather_15 = broadcast_weather_to_15min(act_weather_hourly, full_index)

    wind_models, wind_report = fit_hourly_wind_template(rye_generation, rye_weather_hourly)
    act_weather_hourly_cf = apply_hourly_wind_template(act_weather_hourly, wind_models)
    act_weather_15_cf = broadcast_weather_to_15min(act_weather_hourly_cf, full_index)
    act_weather_15_cf, scaling = scale_synthetic_wind_to_penetration(
        act_weather_15_cf,
        base_agg,
        target_penetration=target_penetration,
    )

    household_out = household_15.merge(
        act_weather_15_cf[[
            "date",
            "air_temperature_2m",
            "wind_speed_10m",
            "cloud_cover",
            "surface_solar_radiation",
            "synthetic_wind_cf",
        ]],
        on="date",
        how="left",
    ).sort_values(["household_id", "date"]).reset_index(drop=True)

    aggregate_out, net_formula = build_aggregate_table(household_15, act_weather_15_cf, battery_audit)
    aggregate_out = aggregate_out.sort_values("date").reset_index(drop=True)

    household_path = output_dir / "nextgen_vpp_household_15min.csv"
    aggregate_path = output_dir / "nextgen_vpp_aggregate_15min.csv"
    metadata_path = output_dir / "nextgen_vpp_metadata.json"
    wind_lookup_path = output_dir / "wind_template_lookup.csv"

    household_out.to_csv(household_path, index=False)
    aggregate_out.to_csv(aggregate_path, index=False)

    lookup_rows = []
    speed_grid = np.arange(0.0, 30.5, 0.5)
    for hour in range(24):
        pred = np.clip(wind_models[hour].predict(speed_grid), 0.0, 1.0)
        for speed, cf in zip(speed_grid, pred):
            lookup_rows.append({"hour": hour, "wind_speed_10m": float(speed), "predicted_wind_cf": float(cf)})
    pd.DataFrame(lookup_rows).to_csv(wind_lookup_path, index=False)

    metadata = {
        "dataset_name": "NextGen-based semi-synthetic VPP research dataset",
        "classification": "semi-synthetic VPP research dataset",
        "time_axis_timezone": "UTC",
        "source_site_timezone": source_timezone,
        "weather_site": SITE_REGISTRY["act_canberra"].name,
        "weather_site_coordinates": {
            "lat": SITE_REGISTRY["act_canberra"].lat,
            "lon": SITE_REGISTRY["act_canberra"].lon,
        },
        "wind_template_site": SITE_REGISTRY["rye_template"].name,
        "wind_template_coordinates": {
            "lat": SITE_REGISTRY["rye_template"].lat,
            "lon": SITE_REGISTRY["rye_template"].lon,
        },
        "sources": {
            "nextgen": {
                "record_id": NEXTGEN_RECORD_ID,
                "fields": ["load_kw", "solar_kw", "battery_power_kw", "battery_soc_kwh", "solar_capacity_kw", "battery_capacity_kwh", "battery_peak_power_kw"],
            },
            "era5": {
                "fields": ["air_temperature_2m", "wind_speed_10m", "cloud_cover", "surface_solar_radiation"],
                "broadcast_to_15min": True,
                "broadcast_rule": "hourly values held constant within each hour",
            },
            "rye_template": {
                "record_id": RYE_RECORD_ID,
                "fit_method": wind_report["fit_method"],
            },
        },
        "battery_power_audit": battery_audit,
        "aggregate_net_load_formula": net_formula,
        "wind_scaling": scaling,
        "wind_template_report": wind_report,
        "output_files": {
            "household_table": str(household_path),
            "aggregate_table": str(aggregate_path),
            "metadata": str(metadata_path),
            "wind_template_lookup": str(wind_lookup_path),
        },
        "field_origins": {
            "real_nextgen_fields": ["load_kw", "solar_kw", "battery_power_kw", "battery_soc_kwh"],
            "external_weather_fields": ["air_temperature_2m", "wind_speed_10m", "cloud_cover", "surface_solar_radiation"],
            "synthetic_fields": ["synthetic_wind_cf", "synthetic_wind_kw", "agg_net_load_kw"],
        },
        "notes": [
            "This is a semi-synthetic research dataset for method validation, not a same-origin operational VPP dataset.",
            "Weather is represented by a shared ACT / Canberra ERA5 grid point.",
            "Wind is generated from a Rye-derived hourly wind-speed template and scaled to a fixed 15% annual energy penetration target.",
        ],
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return {
        "household_csv": str(household_path),
        "aggregate_csv": str(aggregate_path),
        "metadata_json": str(metadata_path),
        "wind_template_lookup_csv": str(wind_lookup_path),
    }
