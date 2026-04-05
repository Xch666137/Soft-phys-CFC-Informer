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


def _round_to_grid(value: float, step: float = 0.25) -> float:
    return float(np.round(value / step) * step)


def fetch_era5_cloud_cover_cds(
    output_csv: str | Path,
    start_date: str,
    end_date: str,
    site_key: str,
    raw_download_dir: str | Path | None = None,
):
    try:
        import cdsapi
        import xarray as xr
    except Exception as exc:
        raise RuntimeError(
            "ERA5 cloud-cover fetching requires cdsapi and xarray. Install the project requirements first."
        ) from exc

    if site_key not in SITE_REGISTRY:
        raise ValueError(f"Unknown site_key '{site_key}'. Available: {sorted(SITE_REGISTRY)}")

    site = SITE_REGISTRY[site_key]
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(raw_download_dir) if raw_download_dir else output_csv.parent / "_cloud_cover_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    client = cdsapi.Client()

    grid_lat = _round_to_grid(site.lat, 0.25)
    grid_lon = _round_to_grid(site.lon, 0.25)
    epsilon = 0.01
    area = [grid_lat + epsilon, grid_lon - epsilon, grid_lat - epsilon, grid_lon + epsilon]

    monthly_frames = []
    month_starts = pd.period_range(start=start, end=end, freq="M")
    for month_period in month_starts:
        month_start = max(start, month_period.start_time.normalize())
        month_end = min(end, month_period.end_time.normalize())
        days = [f"{d.day:02d}" for d in pd.date_range(month_start, month_end, freq="D")]
        raw_path = raw_dir / f"{site_key}_cloud_cover_{month_period.strftime('%Y%m')}.nc"

        request = {
            "product_type": "reanalysis",
            "variable": ["total_cloud_cover"],
            "year": [f"{month_period.year:04d}"],
            "month": [f"{month_period.month:02d}"],
            "day": days,
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        client.retrieve("reanalysis-era5-single-levels", request, str(raw_path))

        dataset = xr.open_dataset(raw_path)
        if "latitude" in dataset.dims:
            dataset = dataset.isel(latitude=0)
        if "longitude" in dataset.dims:
            dataset = dataset.isel(longitude=0)

        time_index = pd.to_datetime(
            dataset["valid_time"].values if "valid_time" in dataset else dataset["time"].values,
            utc=True,
        )
        tcc_values = np.asarray(dataset["tcc"].values, dtype=np.float64)
        monthly_frames.append(pd.DataFrame({
            "date": time_index,
            "cloud_cover": np.clip(tcc_values, 0.0, 1.0),
        }))
        dataset.close()

    cloud_cover = pd.concat(monthly_frames, ignore_index=True)
    cloud_cover = cloud_cover[
        (cloud_cover["date"] >= start.tz_localize("UTC"))
        & (cloud_cover["date"] < (end + pd.Timedelta(days=1)).tz_localize("UTC"))
    ]
    cloud_cover = cloud_cover.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    cloud_cover.to_csv(output_csv, index=False)
    return output_csv


def enrich_weather_with_cloud_cover(
    weather_csv: str | Path,
    cloud_cover_csv: str | Path,
    output_csv: str | Path | None = None,
):
    weather_path = Path(weather_csv)
    cloud_path = Path(cloud_cover_csv)
    out_path = Path(output_csv) if output_csv else weather_path

    weather = pd.read_csv(weather_path)
    cloud = pd.read_csv(cloud_path)
    weather["date"] = pd.to_datetime(weather["date"], utc=True)
    cloud["date"] = pd.to_datetime(cloud["date"], utc=True)

    if "cloud_cover" in weather.columns:
        weather = weather.drop(columns=["cloud_cover"])

    merged = weather.merge(cloud[["date", "cloud_cover"]], on="date", how="left")
    merged = merged.sort_values("date").reset_index(drop=True)
    merged.to_csv(out_path, index=False)
    return out_path


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _parse_timestamp_series(series: pd.Series, timezone: str):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        timestamps = pd.to_datetime(numeric.astype("int64"), unit="s", utc=True)
        return timestamps.dt.tz_convert("UTC")

    timestamps = pd.to_datetime(series)
    if getattr(timestamps.dt, "tz", None) is None:
        timestamps = timestamps.dt.tz_localize(timezone, ambiguous="infer", nonexistent="shift_forward")
    else:
        timestamps = timestamps.dt.tz_convert(timezone)
    return timestamps.dt.tz_convert("UTC")


def _standardize_nextgen_columns(df: pd.DataFrame):
    normalized = {_normalize_name(c): c for c in df.columns}
    alias_groups = {
        "date": ["originalindex", "index"],
        "load_kw": ["loadpowerkw"],
        "solar_kw": ["solarpowerkw"],
        "battery_power_kw": ["batterypowerkw"],
        "battery_soc_kwh": ["batterysockwh"],
        "solar_capacity_kw": ["solarcapacitykw"],
        "battery_capacity_kwh": ["batterycapacitykwh"],
        "battery_peak_power_kw": ["batterypeakpowerkw"],
    }

    resolved = {}
    missing = []
    for target_col, aliases in alias_groups.items():
        source = next((normalized[a] for a in aliases if a in normalized), None)
        if source is None:
            missing.append("/".join(aliases))
        else:
            resolved[source] = target_col

    if missing:
        raise ValueError(f"NextGen CSV missing expected columns: {missing}")

    out = df[list(resolved.keys())].rename(columns=resolved)
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


def audit_generation_sign(series: pd.Series, positive_label: str, negative_label: str):
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "sign_convention": "unknown",
            "positive_share": None,
            "negative_share": None,
        }

    positive_share = float((clean > 0).mean())
    negative_share = float((clean < 0).mean())
    convention = positive_label if positive_share >= negative_share else negative_label
    return {
        "sign_convention": convention,
        "positive_share": positive_share,
        "negative_share": negative_share,
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
    weather = weather_hourly.copy()
    weather["date"] = pd.to_datetime(weather["date"], utc=True)
    weather = weather.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")

    mapped = weather.reindex(full_index.floor("h"))
    mapped = mapped.ffill().bfill()
    mapped.index = full_index
    mapped = mapped.reset_index().rename(columns={"index": "date"})
    mapped["date"] = pd.to_datetime(mapped["date"], utc=True)
    return mapped


def reindex_households_to_full_index(household_df: pd.DataFrame, full_index: pd.DatetimeIndex):
    frames = []
    for household_id, group in household_df.groupby("household_id", sort=False):
        aligned = group.set_index("date").reindex(full_index)
        aligned["household_id"] = household_id
        aligned = aligned.ffill()
        frames.append(aligned.reset_index().rename(columns={"index": "date"}))
    return pd.concat(frames, ignore_index=True)


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
    rye["date"] = pd.to_datetime(rye["date"], utc=True).dt.floor("h")
    weather["date"] = pd.to_datetime(weather["date"], utc=True).dt.floor("h")

    merged = rye.merge(weather[["date", "wind_speed_10m"]], on="date", how="inner")
    if merged.empty:
        raise ValueError("Rye generation and hourly weather do not overlap.")

    rated_kw = float(np.quantile(np.maximum(merged["wind_kw"].to_numpy(dtype=float), 0.0), 0.995))
    rated_kw = rated_kw if rated_kw > 0 else float(np.maximum(merged["wind_kw"].max(), 1.0))
    merged["wind_cf"] = np.clip(merged["wind_kw"] / rated_kw, 0.0, 1.0)
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=["wind_speed_10m", "wind_cf"])
    if merged.empty:
        raise ValueError("Wind template fitting data is empty after dropping NaN/Inf rows.")
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
    solar_sign_audit: dict,
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

    solar_sign = solar_sign_audit["sign_convention"]
    if solar_sign == "negative_is_generation":
        solar_term = merged["agg_solar_kw"]
        solar_formula = "+ agg_solar_kw"
    else:
        solar_term = -merged["agg_solar_kw"]
        solar_formula = "- agg_solar_kw"

    sign_convention = battery_sign_audit["battery_power_sign_convention"]
    if sign_convention == "positive_is_discharging":
        merged["agg_net_load_kw"] = (
            merged["agg_load_kw"] + solar_term - merged["synthetic_wind_kw"] - merged["agg_battery_power_kw"]
        )
        formula = f"agg_load_kw {solar_formula} - synthetic_wind_kw - agg_battery_power_kw"
    else:
        merged["agg_net_load_kw"] = (
            merged["agg_load_kw"] + solar_term - merged["synthetic_wind_kw"] + merged["agg_battery_power_kw"]
        )
        formula = f"agg_load_kw {solar_formula} - synthetic_wind_kw + agg_battery_power_kw"

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
    solar_audit = audit_generation_sign(
        household_15["solar_kw"],
        positive_label="positive_is_generation",
        negative_label="negative_is_generation",
    )

    act_weather_hourly = load_standardized_weather_csv(act_weather_csv)
    rye_generation = load_rye_generation_csv(rye_generation_csv)
    rye_weather_hourly = load_standardized_weather_csv(rye_weather_csv)

    full_index = pd.date_range(
        start=household_15["date"].min(),
        end=household_15["date"].max(),
        freq="15min",
        tz="UTC",
    )

    household_15 = reindex_households_to_full_index(household_15, full_index)

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

    aggregate_out, net_formula = build_aggregate_table(household_15, act_weather_15_cf, battery_audit, solar_audit)
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
        "solar_sign_audit": solar_audit,
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


def build_reference_15min_index(
    audit_year: int = 2018,
    source_timezone: str = "Australia/Sydney",
):
    start_local = pd.Timestamp(f"{audit_year}-01-01 00:00:00", tz=source_timezone)
    end_local = pd.Timestamp(f"{audit_year}-12-31 23:45:00", tz=source_timezone)
    return pd.date_range(start_local, end_local, freq="15min").tz_convert("UTC")


def _safe_zscore(series: pd.Series):
    clean = pd.to_numeric(series, errors="coerce")
    std = float(clean.std(ddof=0)) if len(clean) else 0.0
    if std == 0.0 or math.isnan(std):
        return pd.Series(np.zeros(len(clean)), index=clean.index, dtype=float)
    return ((clean - float(clean.mean())) / std).astype(float)


def _first_timestamp_from_group(group: pd.DataFrame):
    dates = pd.to_datetime(group["date"], utc=True)
    return pd.Timestamp(dates.min())


def _last_timestamp_from_group(group: pd.DataFrame):
    dates = pd.to_datetime(group["date"], utc=True)
    return pd.Timestamp(dates.max())


def audit_nextgen_household_eligibility(
    nextgen_dir: str | Path,
    output_dir: str | Path,
    audit_year: int = 2018,
    source_timezone: str = "Australia/Sydney",
    min_coverage_ratio: float = 1.0,
    min_feature_availability: float = 0.99,
):
    output_dir = _ensure_dir(output_dir)
    full_index = build_reference_15min_index(audit_year=audit_year, source_timezone=source_timezone)
    expected_rows = len(full_index)

    nextgen = load_nextgen_households(nextgen_dir, source_timezone=source_timezone)
    household_15 = resample_nextgen_to_15min(nextgen)

    records = []
    exclusion_counts = {}

    for household_id, raw_group in nextgen.groupby("household_id", sort=True):
        resampled_group = household_15.loc[household_15["household_id"] == household_id].copy()
        resampled_group["date"] = pd.to_datetime(resampled_group["date"], utc=True)
        resampled_group = resampled_group.sort_values("date")
        resampled_index = pd.DatetimeIndex(resampled_group["date"])
        aligned = resampled_group.set_index("date").reindex(full_index)

        raw_start = _first_timestamp_from_group(raw_group)
        raw_end = _last_timestamp_from_group(raw_group)
        resampled_start = pd.Timestamp(resampled_index.min()) if len(resampled_index) else pd.NaT
        resampled_end = pd.Timestamp(resampled_index.max()) if len(resampled_index) else pd.NaT

        full_year_start_ok = bool(pd.notna(raw_start) and raw_start <= full_index[0])
        full_year_end_ok = bool(pd.notna(raw_end) and raw_end >= full_index[-1])
        present_rows = int(resampled_index.intersection(full_index).size)
        coverage_ratio = float(present_rows / expected_rows) if expected_rows else 0.0

        load_ratio = float(aligned["load_kw"].notna().mean())
        solar_ratio = float(aligned["solar_kw"].notna().mean())
        battery_power_ratio = float(aligned["battery_power_kw"].notna().mean())
        battery_soc_ratio = float(aligned["battery_soc_kwh"].notna().mean())

        exclusion_reasons = []
        if not full_year_start_ok:
            exclusion_reasons.append("starts_after_audit_window")
        if not full_year_end_ok:
            exclusion_reasons.append("ends_before_audit_window")
        if coverage_ratio < min_coverage_ratio:
            exclusion_reasons.append("missing_15min_bins")
        if load_ratio < min_feature_availability:
            exclusion_reasons.append("missing_load_kw")
        if solar_ratio < min_feature_availability:
            exclusion_reasons.append("missing_solar_kw")
        if battery_power_ratio < min_feature_availability:
            exclusion_reasons.append("missing_battery_power_kw")
        if battery_soc_ratio < min_feature_availability:
            exclusion_reasons.append("missing_battery_soc_kwh")

        eligible = len(exclusion_reasons) == 0
        for reason in exclusion_reasons:
            exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1

        records.append({
            "household_id": household_id,
            "raw_rows": int(len(raw_group)),
            "raw_start_utc": raw_start.isoformat() if pd.notna(raw_start) else None,
            "raw_end_utc": raw_end.isoformat() if pd.notna(raw_end) else None,
            "resampled_rows_in_window": present_rows,
            "resampled_start_utc": resampled_start.isoformat() if pd.notna(resampled_start) else None,
            "resampled_end_utc": resampled_end.isoformat() if pd.notna(resampled_end) else None,
            "expected_rows": int(expected_rows),
            "coverage_ratio": coverage_ratio,
            "load_available_ratio": load_ratio,
            "solar_available_ratio": solar_ratio,
            "battery_power_available_ratio": battery_power_ratio,
            "battery_soc_available_ratio": battery_soc_ratio,
            "full_year_start_ok": full_year_start_ok,
            "full_year_end_ok": full_year_end_ok,
            "eligible": eligible,
            "exclusion_reason": ";".join(exclusion_reasons) if exclusion_reasons else "",
        })

    audit_df = pd.DataFrame(records).sort_values(["eligible", "household_id"], ascending=[False, True]).reset_index(drop=True)
    eligible_ids = audit_df.loc[audit_df["eligible"], "household_id"].tolist()

    exclusion_manifest = {
        "audit_year": audit_year,
        "source_timezone": source_timezone,
        "expected_rows": int(expected_rows),
        "min_coverage_ratio": float(min_coverage_ratio),
        "min_feature_availability": float(min_feature_availability),
        "raw_household_files": int(nextgen["household_id"].nunique()),
        "eligible_households": len(eligible_ids),
        "excluded_households": int(len(audit_df) - len(eligible_ids)),
        "exclusion_counts": exclusion_counts,
        "eligible_household_ids": eligible_ids,
    }

    audit_csv = output_dir / "household_eligibility.csv"
    exclusion_json = output_dir / "household_exclusion_report.json"
    audit_df.to_csv(audit_csv, index=False)
    with open(exclusion_json, "w", encoding="utf-8") as f:
        json.dump(exclusion_manifest, f, indent=2, ensure_ascii=False)

    return audit_df, exclusion_manifest, full_index, audit_csv, exclusion_json


def _prepare_filled_household_frame(
    nextgen_dir: str | Path,
    household_ids: list[str],
    full_index: pd.DatetimeIndex,
    source_timezone: str = "Australia/Sydney",
):
    nextgen = load_nextgen_households(nextgen_dir, source_timezone=source_timezone)
    nextgen = nextgen.loc[nextgen["household_id"].isin(household_ids)].copy()
    household_15 = resample_nextgen_to_15min(nextgen)

    household_15 = reindex_households_to_full_index(household_15, full_index)

    for static_col in ["solar_capacity_kw", "battery_capacity_kwh", "battery_peak_power_kw"]:
        household_15[static_col] = household_15.groupby("household_id")[static_col].ffill().bfill()

    household_15[["load_kw", "solar_kw", "battery_power_kw"]] = household_15[
        ["load_kw", "solar_kw", "battery_power_kw"]
    ].fillna(0.0)
    household_15["battery_soc_kwh"] = household_15.groupby("household_id")["battery_soc_kwh"].ffill().bfill()
    household_15["date"] = pd.to_datetime(household_15["date"], utc=True)

    return household_15.sort_values(["household_id", "date"]).reset_index(drop=True)


def compute_household_descriptors(
    household_df_15min: pd.DataFrame,
    solar_sign_audit: dict,
    source_timezone: str = "Australia/Sydney",
):
    working = household_df_15min.copy()
    working["local_date"] = pd.to_datetime(working["date"], utc=True).dt.tz_convert(source_timezone)
    working["local_hour"] = working["local_date"].dt.hour

    if solar_sign_audit.get("sign_convention") == "negative_is_generation":
        working["solar_generation_kw"] = (-working["solar_kw"]).clip(lower=0.0)
    else:
        working["solar_generation_kw"] = working["solar_kw"].clip(lower=0.0)

    grouped = working.groupby("household_id", sort=True)
    descriptors = grouped.agg(
        load_mean=("load_kw", "mean"),
        load_std=("load_kw", "std"),
        solar_capacity_kw=("solar_capacity_kw", "max"),
        battery_capacity_kwh=("battery_capacity_kwh", "max"),
        battery_peak_power_kw=("battery_peak_power_kw", "max"),
        battery_abs_power_mean=("battery_power_kw", lambda s: float(np.mean(np.abs(s.to_numpy(dtype=float))))),
        battery_soc_range=("battery_soc_kwh", lambda s: float(s.max() - s.min())),
        total_load_kwh=("load_kw", lambda s: float(s.sum() * 0.25)),
        total_solar_generation_kwh=("solar_generation_kw", lambda s: float(s.sum() * 0.25)),
    ).reset_index()

    evening = working.loc[(working["local_hour"] >= 17) & (working["local_hour"] < 21)]
    evening_mean = evening.groupby("household_id")["load_kw"].mean().rename("evening_load_mean")
    descriptors = descriptors.merge(evening_mean, on="household_id", how="left")
    descriptors["evening_load_mean"] = descriptors["evening_load_mean"].fillna(descriptors["load_mean"])
    descriptors["evening_peak_ratio"] = descriptors["evening_load_mean"] / descriptors["load_mean"].replace(0.0, np.nan)
    descriptors["evening_peak_ratio"] = descriptors["evening_peak_ratio"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    descriptors["solar_generation_ratio"] = descriptors["total_solar_generation_kwh"] / descriptors["total_load_kwh"].replace(0.0, np.nan)
    descriptors["solar_generation_ratio"] = descriptors["solar_generation_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    descriptors["load_score"] = (
        _safe_zscore(descriptors["load_mean"])
        + 0.5 * _safe_zscore(descriptors["load_std"])
        + 0.5 * _safe_zscore(descriptors["evening_peak_ratio"])
    )
    descriptors["der_score"] = (
        _safe_zscore(descriptors["solar_capacity_kw"])
        + _safe_zscore(descriptors["solar_generation_ratio"])
    )
    descriptors["flex_score"] = (
        _safe_zscore(descriptors["battery_capacity_kwh"])
        + 0.5 * _safe_zscore(descriptors["battery_peak_power_kw"])
        + 0.5 * _safe_zscore(descriptors["battery_abs_power_mean"])
        + 0.5 * _safe_zscore(descriptors["battery_soc_range"])
    )

    return descriptors.sort_values(["load_score", "household_id"], ascending=[False, True]).reset_index(drop=True)


def _derive_split_portfolio_counts(num_portfolios: int):
    if num_portfolios < 3:
        raise ValueError("At least 3 portfolios are required to produce train/val/test splits.")
    if num_portfolios == 5:
        return {"train": 3, "val": 1, "test": 1}

    train = max(1, int(round(num_portfolios * 0.6)))
    val = max(1, int(round(num_portfolios * 0.2)))
    test = max(1, num_portfolios - train - val)

    while train + val + test > num_portfolios:
        if train > val and train > 1:
            train -= 1
        elif val > 1:
            val -= 1
        elif test > 1:
            test -= 1
        else:
            break

    while train + val + test < num_portfolios:
        train += 1

    if val < 1 or test < 1:
        raise ValueError("Portfolio split must retain at least one validation and one test portfolio.")
    return {"train": train, "val": val, "test": test}


def assign_households_to_splits(
    descriptors: pd.DataFrame,
    portfolio_size: int = 5,
):
    num_portfolios = len(descriptors) // portfolio_size
    used_count = num_portfolios * portfolio_size
    usable = descriptors.sort_values(["load_score", "household_id"], ascending=[False, True]).head(used_count).copy()
    leftover = descriptors.iloc[used_count:].copy()

    split_portfolio_counts = _derive_split_portfolio_counts(num_portfolios)
    target_households = {k: v * portfolio_size for k, v in split_portfolio_counts.items()}
    split_state = {
        name: {
            "rows": [],
            "current_count": 0,
            "current_load": 0.0,
        }
        for name in ["train", "val", "test"]
    }

    for row in usable.to_dict("records"):
        candidates = [
            name for name in ["train", "val", "test"]
            if split_state[name]["current_count"] < target_households[name]
        ]
        chosen = min(
            candidates,
            key=lambda name: (
                split_state[name]["current_load"],
                split_state[name]["current_count"] / max(target_households[name], 1),
                ["train", "val", "test"].index(name),
            ),
        )
        split_state[chosen]["rows"].append(row)
        split_state[chosen]["current_count"] += 1
        split_state[chosen]["current_load"] += float(row["load_mean"])

    split_frames = {}
    for split_name, state in split_state.items():
        split_frames[split_name] = pd.DataFrame(state["rows"]).reset_index(drop=True)
        split_frames[split_name]["split"] = split_name

    return split_frames, leftover, split_portfolio_counts


def _feature_distance(row: pd.Series, members_df: pd.DataFrame):
    if members_df.empty:
        return 0.0
    features = ["load_score", "der_score", "flex_score", "load_mean", "solar_generation_ratio", "battery_capacity_kwh"]
    centroid = members_df[features].mean().to_numpy(dtype=float)
    return float(np.linalg.norm(row[features].to_numpy(dtype=float) - centroid))


def build_portfolios_for_split(
    split_df: pd.DataFrame,
    split_name: str,
    num_portfolios: int,
    portfolio_size: int = 5,
):
    if len(split_df) != num_portfolios * portfolio_size:
        raise ValueError(f"Split '{split_name}' does not match the expected household count for {num_portfolios} portfolios.")

    ordered = split_df.sort_values(["load_score", "household_id"], ascending=[False, True]).reset_index(drop=True)
    portfolio_ids = [f"{split_name}_portfolio_{idx + 1:02d}" for idx in range(num_portfolios)]
    buckets = {
        pid: {
            "rows": [],
            "load_total": 0.0,
        }
        for pid in portfolio_ids
    }

    seed_rows = ordered.head(num_portfolios).to_dict("records")
    for pid, row in zip(portfolio_ids, seed_rows):
        buckets[pid]["rows"].append(row)
        buckets[pid]["load_total"] += float(row["load_mean"])

    for row in ordered.iloc[num_portfolios:].to_dict("records"):
        candidates = [pid for pid in portfolio_ids if len(buckets[pid]["rows"]) < portfolio_size]
        min_load_total = min(buckets[pid]["load_total"] for pid in candidates)
        lightest = [pid for pid in candidates if buckets[pid]["load_total"] == min_load_total]

        if len(lightest) == 1:
            chosen = lightest[0]
        else:
            chosen = max(
                lightest,
                key=lambda pid: (
                    _feature_distance(pd.Series(row), pd.DataFrame(buckets[pid]["rows"])),
                    -portfolio_ids.index(pid),
                ),
            )

        buckets[chosen]["rows"].append(row)
        buckets[chosen]["load_total"] += float(row["load_mean"])

    membership_rows = []
    for pid in portfolio_ids:
        members = pd.DataFrame(buckets[pid]["rows"]).sort_values("household_id").reset_index(drop=True)
        members["portfolio_id"] = pid
        members["split"] = split_name
        membership_rows.append(members)

    return pd.concat(membership_rows, ignore_index=True)


def label_portfolios(portfolio_summary: pd.DataFrame):
    out = portfolio_summary.copy()
    load_z = _safe_zscore(out["load_score_mean"])
    der_z = _safe_zscore(out["der_score_mean"])
    flex_z = _safe_zscore(out["flex_score_mean"])
    labels = []
    for idx in out.index:
        vals = {
            "load_heavy": float(load_z.loc[idx]),
            "der_heavy": float(der_z.loc[idx]),
            "flex_heavy": float(flex_z.loc[idx]),
        }
        if max(abs(v) for v in vals.values()) <= 0.5:
            labels.append("balanced")
            continue
        top_label, top_value = max(vals.items(), key=lambda kv: kv[1])
        labels.append(top_label if top_value >= 0.75 else "mixed")
    out["portfolio_label"] = labels
    return out


def build_time_generalization_training_table(training_df: pd.DataFrame, train_portfolio_ids: list[str]):
    """
    Build a second benchmark table that measures time generalization using only the
    portfolios that belong to the composition benchmark train split.
    """
    df = training_df.loc[training_df["portfolio_id"].isin(train_portfolio_ids)].copy()
    if df.empty:
        raise ValueError("Cannot build time-generalization table without train portfolios.")

    split_frames = []
    for portfolio_id, portfolio_df in df.groupby("portfolio_id", sort=True):
        portfolio_df = portfolio_df.sort_values("date").reset_index(drop=True)
        n = len(portfolio_df)
        n_train = int(n * 0.7)
        n_test = int(n * 0.2)
        n_val = n - n_train - n_test

        split_labels = np.empty(n, dtype=object)
        split_labels[:n_train] = "train"
        split_labels[n_train:n_train + n_val] = "val"
        split_labels[n_train + n_val:] = "test"

        split_df = portfolio_df.copy()
        split_df["split"] = split_labels
        split_frames.append(split_df)

    return pd.concat(split_frames, ignore_index=True).sort_values(["portfolio_id", "date"]).reset_index(drop=True)


def build_multi_portfolio_dataset(
    nextgen_dir: str | Path,
    act_weather_csv: str | Path,
    rye_generation_csv: str | Path,
    rye_weather_csv: str | Path,
    output_dir: str | Path,
    portfolio_size: int = 5,
    target_penetration: float = 0.15,
    audit_year: int = 2018,
    source_timezone: str = "Australia/Sydney",
    region_id: str = "act_canberra",
    min_feature_availability: float = 0.99,
):
    output_dir = _ensure_dir(output_dir)
    audit_df, exclusion_manifest, full_index, audit_csv, exclusion_json = audit_nextgen_household_eligibility(
        nextgen_dir=nextgen_dir,
        output_dir=output_dir,
        audit_year=audit_year,
        source_timezone=source_timezone,
        min_feature_availability=min_feature_availability,
    )
    eligible_ids = audit_df.loc[audit_df["eligible"], "household_id"].tolist()
    if len(eligible_ids) < portfolio_size * 3:
        raise ValueError(
            f"Need at least {portfolio_size * 3} eligible households to create train/val/test portfolios. "
            f"Got {len(eligible_ids)}."
        )

    household_15 = _prepare_filled_household_frame(
        nextgen_dir=nextgen_dir,
        household_ids=eligible_ids,
        full_index=full_index,
        source_timezone=source_timezone,
    )
    battery_audit = audit_battery_power_sign(household_15)
    solar_audit = audit_generation_sign(
        household_15["solar_kw"],
        positive_label="positive_is_generation",
        negative_label="negative_is_generation",
    )

    descriptors = compute_household_descriptors(
        household_df_15min=household_15,
        solar_sign_audit=solar_audit,
        source_timezone=source_timezone,
    )
    split_frames, leftover_df, split_portfolio_counts = assign_households_to_splits(
        descriptors=descriptors,
        portfolio_size=portfolio_size,
    )

    membership_frames = []
    for split_name, split_df in split_frames.items():
        membership_frames.append(
            build_portfolios_for_split(
                split_df=split_df,
                split_name=split_name,
                num_portfolios=split_portfolio_counts[split_name],
                portfolio_size=portfolio_size,
            )
        )
    membership = pd.concat(membership_frames, ignore_index=True)
    membership = membership[["portfolio_id", "split", "household_id"]].sort_values(
        ["split", "portfolio_id", "household_id"]
    ).reset_index(drop=True)

    eligible_members = household_15.merge(membership, on="household_id", how="inner")
    if eligible_members["household_id"].nunique() != len(membership["household_id"].unique()):
        raise ValueError("Membership merge dropped some eligible households unexpectedly.")

    act_weather_hourly = load_standardized_weather_csv(act_weather_csv)
    rye_generation = load_rye_generation_csv(rye_generation_csv)
    rye_weather_hourly = load_standardized_weather_csv(rye_weather_csv)
    wind_models, wind_report = fit_hourly_wind_template(rye_generation, rye_weather_hourly)
    act_weather_hourly_cf = apply_hourly_wind_template(act_weather_hourly, wind_models)
    act_weather_15_cf = broadcast_weather_to_15min(act_weather_hourly_cf, full_index)

    portfolio_ts_frames = []
    summary_rows = []
    descriptor_lookup = descriptors.set_index("household_id")

    for portfolio_id, member_ts in eligible_members.groupby("portfolio_id", sort=True):
        split_name = str(member_ts["split"].iloc[0])
        aggregate_input = member_ts.drop(columns=["split"]).copy()
        base_agg = aggregate_input.groupby("date", as_index=False)["load_kw"].sum().rename(columns={"load_kw": "agg_load_kw"})
        portfolio_weather, scaling = scale_synthetic_wind_to_penetration(
            act_weather_15_cf.copy(),
            base_agg,
            target_penetration=target_penetration,
        )
        aggregate_out, net_formula = build_aggregate_table(
            household_df_15min=aggregate_input,
            shared_weather_15min=portfolio_weather,
            battery_sign_audit=battery_audit,
            solar_sign_audit=solar_audit,
        )
        aggregate_out["portfolio_id"] = portfolio_id
        aggregate_out["split"] = split_name
        aggregate_out["region_id"] = region_id
        aggregate_out["p_vpp_kw"] = aggregate_out["agg_net_load_kw"]
        portfolio_ts_frames.append(aggregate_out)

        member_ids = sorted(member_ts["household_id"].unique().tolist())
        member_desc = descriptor_lookup.loc[member_ids]
        summary_rows.append({
            "portfolio_id": portfolio_id,
            "split": split_name,
            "region_id": region_id,
            "household_count": int(len(member_ids)),
            "member_households": ",".join(member_ids),
            "load_score_mean": float(member_desc["load_score"].mean()),
            "der_score_mean": float(member_desc["der_score"].mean()),
            "flex_score_mean": float(member_desc["flex_score"].mean()),
            "agg_load_mean_kw": float(aggregate_out["agg_load_kw"].mean()),
            "agg_load_std_kw": float(aggregate_out["agg_load_kw"].std()),
            "agg_net_load_mean_kw": float(aggregate_out["agg_net_load_kw"].mean()),
            "agg_net_load_std_kw": float(aggregate_out["agg_net_load_kw"].std()),
            "agg_solar_generation_mean_kw": float((-aggregate_out["agg_solar_kw"]).clip(lower=0.0).mean())
            if solar_audit["sign_convention"] == "negative_is_generation"
            else float(aggregate_out["agg_solar_kw"].clip(lower=0.0).mean()),
            "agg_battery_abs_power_mean_kw": float(np.mean(np.abs(aggregate_out["agg_battery_power_kw"].to_numpy(dtype=float)))),
            "wind_load_energy_ratio": float(scaling["synthetic_wind_energy_kwh"] / scaling["aggregate_load_energy_kwh"]),
            "wind_penetration_target": float(scaling["wind_penetration_target"]),
        })

    multi_portfolio_ts = pd.concat(portfolio_ts_frames, ignore_index=True).sort_values(
        ["portfolio_id", "date"]
    ).reset_index(drop=True)
    portfolio_summary = label_portfolios(pd.DataFrame(summary_rows).sort_values(["split", "portfolio_id"]).reset_index(drop=True))

    training_df = pd.DataFrame({
        "date": pd.to_datetime(multi_portfolio_ts["date"], utc=True),
        "portfolio_id": multi_portfolio_ts["portfolio_id"],
        "region_id": multi_portfolio_ts["region_id"],
        "split": multi_portfolio_ts["split"],
        "p_vpp_mw": multi_portfolio_ts["p_vpp_kw"] / 1000.0,
        "temperature": multi_portfolio_ts["air_temperature_2m"],
        "irradiance": multi_portfolio_ts["surface_solar_radiation"],
        "wind_speed": multi_portfolio_ts["wind_speed_10m"],
        "p_load_mw": multi_portfolio_ts["agg_load_kw"] / 1000.0,
        "p_pv_mw": (
            (-multi_portfolio_ts["agg_solar_kw"]).clip(lower=0.0)
            if solar_audit["sign_convention"] == "negative_is_generation"
            else multi_portfolio_ts["agg_solar_kw"].clip(lower=0.0)
        ) / 1000.0,
        "p_wind_mw": multi_portfolio_ts["synthetic_wind_kw"] / 1000.0,
        "p_battery_mw": multi_portfolio_ts["agg_battery_power_kw"] / 1000.0,
        "e_battery_soc_mwh": multi_portfolio_ts["agg_battery_soc_kwh"] / 1000.0,
    }).sort_values(["portfolio_id", "date"]).reset_index(drop=True)

    time_generalization_df = build_time_generalization_training_table(
        training_df,
        sorted(portfolio_summary.loc[portfolio_summary["split"] == "train", "portfolio_id"].tolist()),
    )

    timeseries_path = output_dir / "multi_portfolio_timeseries.csv"
    membership_path = output_dir / "portfolio_membership.csv"
    summary_path = output_dir / "portfolio_summary.csv"
    training_path = output_dir / "portfolio_dataset_for_training.csv"
    time_generalization_path = output_dir / "portfolio_dataset_for_time_generalization.csv"
    metadata_path = output_dir / "multi_portfolio_metadata.json"

    multi_portfolio_ts.to_csv(timeseries_path, index=False)
    membership.to_csv(membership_path, index=False)
    portfolio_summary.to_csv(summary_path, index=False)
    training_df.to_csv(training_path, index=False)
    time_generalization_df.to_csv(time_generalization_path, index=False)

    metadata = {
        "dataset_name": "NextGen multi-portfolio semi-synthetic VPP benchmark",
        "classification": "single-climate multi-portfolio semi-synthetic VPP benchmark",
        "audit_year": audit_year,
        "source_site_timezone": source_timezone,
        "region_id": region_id,
        "portfolio_size": portfolio_size,
        "min_feature_availability": float(min_feature_availability),
        "split_portfolio_counts": split_portfolio_counts,
        "eligible_households": int(len(eligible_ids)),
        "leftover_eligible_households": int(len(leftover_df)),
        "leftover_household_ids": leftover_df["household_id"].tolist(),
        "assumptions": {
            "shared_weather": SITE_REGISTRY["act_canberra"].name,
            "cloud_cover_enabled": False,
            "strict_household_disjoint": True,
        },
        "audits": {
            "battery_power_audit": battery_audit,
            "solar_sign_audit": solar_audit,
        },
        "wind_template_report": wind_report,
        "aggregate_net_load_formula": net_formula,
        "output_files": {
            "household_eligibility_csv": str(audit_csv),
            "household_exclusion_report_json": str(exclusion_json),
            "multi_portfolio_timeseries_csv": str(timeseries_path),
            "portfolio_membership_csv": str(membership_path),
            "portfolio_summary_csv": str(summary_path),
            "portfolio_dataset_for_training_csv": str(training_path),
            "portfolio_dataset_for_time_generalization_csv": str(time_generalization_path),
        },
        "notes": [
            "This benchmark evaluates composition-level generalization under a shared ACT weather driver.",
            "Households are split into train/val/test before training use and are not reused across portfolios.",
            "Cloud cover is retained in raw weather files but is not enabled as a v1 training covariate.",
            "Time-generalization data is built only from train portfolios and split chronologically within each portfolio.",
        ],
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return {
        "household_eligibility_csv": str(audit_csv),
        "household_exclusion_report_json": str(exclusion_json),
        "multi_portfolio_timeseries_csv": str(timeseries_path),
        "portfolio_membership_csv": str(membership_path),
        "portfolio_summary_csv": str(summary_path),
        "portfolio_dataset_for_training_csv": str(training_path),
        "portfolio_dataset_for_time_generalization_csv": str(time_generalization_path),
        "multi_portfolio_metadata_json": str(metadata_path),
    }
