"""Data quality, feature construction, and CAMS sampling for PM2.5 forecasts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
AQ_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path = EXPERIMENT_ROOT

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def station_metadata(self) -> Path:
        return self.root / "station_metadata.csv"

    @property
    def derived(self) -> Path:
        return self.root / "data" / "derived"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def provenance(self) -> Path:
        return self.root / "provenance"

    @property
    def cams_netcdf(self) -> Path:
        return self.root / "data" / "external" / "cams_global_forecasts" / "netcdf"

    @property
    def cams_earth_engine(self) -> Path:
        return self.root / "data" / "external" / "cams_earth_engine"


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or ExperimentPaths().config
    return json.loads(config_path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _validate_station_metadata(metadata: pd.DataFrame, observation_paths: list[Path]) -> None:
    required = {
        "source_file",
        "station_code",
        "station_name",
        "province",
        "region",
        "latitude",
        "longitude",
        "timezone",
        "utc_offset_hours",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Station metadata columns are missing: {sorted(missing)}")
    if metadata.station_code.duplicated().any():
        raise ValueError("Station metadata contains duplicate station codes")
    if metadata.source_file.duplicated().any():
        raise ValueError("Station metadata contains duplicate source files")
    configured = set(metadata.source_file)
    discovered = {path.name for path in observation_paths}
    if configured != discovered:
        raise ValueError(
            "Station metadata and observation files differ: "
            f"metadata_only={sorted(configured - discovered)}, "
            f"files_only={sorted(discovered - configured)}"
        )
    if not metadata.latitude.astype(float).between(-90, 90).all():
        raise ValueError("Station latitude is outside -90 to 90 degrees")
    if not metadata.longitude.astype(float).between(-180, 180).all():
        raise ValueError("Station longitude is outside -180 to 180 degrees")


def _parse_timestamp(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    date_value = pd.to_datetime(frame["TANGGAL"], errors="coerce")
    hour_value = pd.to_numeric(frame["JAM_UTC"], errors="coerce")
    valid_hour = hour_value.between(0, 23) & np.isclose(hour_value % 1, 0)
    timestamp = date_value + pd.to_timedelta(hour_value.where(valid_hour), unit="h")
    return timestamp.dt.tz_localize("UTC"), valid_hour


def load_and_quality_control_observations(
    config: dict[str, Any], paths: ExperimentPaths
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    configured_glob = os.environ.get(
        "PM25_OBSERVATION_GLOB", config["observation_glob"]
    )
    configured_path = Path(configured_glob).expanduser()
    observation_pattern = (
        configured_path if configured_path.is_absolute() else paths.root / configured_path
    )
    observation_paths = sorted(
        observation_pattern.parent.resolve().glob(observation_pattern.name)
    )
    if not observation_paths:
        raise FileNotFoundError("No observation files matched the configured glob")
    metadata = pd.read_csv(paths.station_metadata)
    _validate_station_metadata(metadata, observation_paths)
    metadata_lookup = metadata.set_index("source_file")

    required_columns = {"STASIUN", "TANGGAL", "JAM_UTC", "PM25", "RH", "TEMP"}
    frames: list[pd.DataFrame] = []
    source_manifest: list[dict[str, Any]] = []
    for source_path in observation_paths:
        frame = pd.read_csv(
            source_path,
            dtype={"STASIUN": "string", "TANGGAL": "string", "JAM_UTC": "string"},
        )
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"{source_path.name} is missing columns: {sorted(missing)}")
        expected_code = str(metadata_lookup.loc[source_path.name, "station_code"])
        observed_codes = set(frame["STASIUN"].dropna().str.strip().unique())
        if observed_codes != {expected_code}:
            raise ValueError(
                f"{source_path.name} station identity mismatch: "
                f"expected {expected_code}, found {sorted(observed_codes)}"
            )
        frame = frame.loc[:, ["STASIUN", "TANGGAL", "JAM_UTC", "PM25", "RH", "TEMP"]]
        frame["source_file"] = source_path.name
        frames.append(frame)
        try:
            manifest_path = str(source_path.relative_to(AQ_ROOT))
        except ValueError:
            manifest_path = str(source_path)
        source_manifest.append(
            {
                "path": manifest_path,
                "bytes": source_path.stat().st_size,
                "sha256": file_sha256(source_path),
                "rows": len(frame),
            }
        )

    raw = pd.concat(frames, ignore_index=True)
    raw["station_code"] = raw["STASIUN"].str.strip()
    raw["timestamp_utc"], valid_hour = _parse_timestamp(raw)
    for column in ("PM25", "RH", "TEMP"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    duplicate_mask = raw.timestamp_utc.notna() & raw.duplicated(
        ["station_code", "timestamp_utc"], keep=False
    )
    duplicate_removed_mask = raw.timestamp_utc.notna() & raw.duplicated(
        ["station_code", "timestamp_utc"], keep="first"
    )
    duplicate_rows = raw.loc[duplicate_mask]
    if len(duplicate_rows):
        conflicting = (
            duplicate_rows.groupby(["station_code", "timestamp_utc"], dropna=False)[
                ["PM25", "RH", "TEMP"]
            ]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        conflicting_duplicate_keys = int(conflicting.sum())
    else:
        conflicting_duplicate_keys = 0
    if conflicting_duplicate_keys:
        raise ValueError(
            f"Found {conflicting_duplicate_keys} conflicting duplicate station-hours"
        )

    qc = config["quality_control"]
    raw["pm25_qc"] = np.select(
        [
            raw["PM25"].isna(),
            raw["PM25"].lt(qc["pm25_min_inclusive_ug_m3"]),
            raw["PM25"].ge(qc["pm25_max_exclusive_ug_m3"]),
        ],
        ["missing_or_nonnumeric", "negative", "at_or_above_985"],
        default="valid",
    )
    raw["rh_qc"] = np.select(
        [
            raw["RH"].isna(),
            raw["RH"].le(qc["relative_humidity_min_exclusive_pct"])
            | raw["RH"].gt(qc["relative_humidity_max_inclusive_pct"]),
        ],
        ["missing_or_nonnumeric", "outside_valid_range"],
        default="valid",
    )
    raw["temperature_qc"] = np.select(
        [
            raw["TEMP"].isna(),
            raw["TEMP"].lt(qc["temperature_min_inclusive_c"])
            | raw["TEMP"].gt(qc["temperature_max_inclusive_c"]),
        ],
        ["missing_or_nonnumeric", "outside_valid_range"],
        default="valid",
    )
    raw["pm25_ug_m3"] = raw["PM25"].where(raw.pm25_qc.eq("valid"))
    raw["relative_humidity_pct"] = raw["RH"].where(raw.rh_qc.eq("valid"))
    raw["temperature_c"] = raw["TEMP"].where(raw.temperature_qc.eq("valid"))

    clean = (
        raw.dropna(subset=["timestamp_utc"])
        .sort_values(["station_code", "timestamp_utc", "source_file"])
        .drop_duplicates(["station_code", "timestamp_utc"], keep="first")
        .merge(
            metadata.drop(columns=["source_file"]),
            on="station_code",
            how="left",
            validate="many_to_one",
        )
    )
    clean["timestamp_local"] = clean["timestamp_utc"] + pd.to_timedelta(
        clean["utc_offset_hours"], unit="h"
    )

    quality_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    for source_file, group_raw in raw.groupby("source_file", sort=True):
        station_code = str(metadata_lookup.loc[source_file, "station_code"])
        group = clean.loc[clean.station_code.eq(station_code)].copy()
        start = group.timestamp_utc.min()
        end = group.timestamp_utc.max()
        expected = int((end - start) / pd.Timedelta(hours=1)) + 1
        observed = int(group.timestamp_utc.nunique())
        gaps = group.timestamp_utc.sort_values().diff().dropna() / pd.Timedelta(hours=1)
        original = group_raw
        quality_rows.append(
            {
                "station_code": station_code,
                "station_name": metadata_lookup.loc[source_file, "station_name"],
                "source_file": source_file,
                "start_utc": start.isoformat(),
                "end_utc": end.isoformat(),
                "source_rows": len(original),
                "unique_station_hours": observed,
                "expected_hours_within_span": expected,
                "absent_hours_within_span": expected - observed,
                "temporal_coverage_pct": 100.0 * observed / expected,
                "maximum_gap_hours": float(gaps.max()) if len(gaps) else 0.0,
                "timestamp_parse_failures": int(original.timestamp_utc.isna().sum()),
                "invalid_hour_rows": int((~valid_hour.loc[original.index]).sum()),
                "duplicate_rows_affected": int(
                    duplicate_mask.loc[original.index].sum()
                ),
                "duplicate_rows_removed": int(
                    duplicate_removed_mask.loc[original.index].sum()
                ),
                "valid_pm25": int(original.pm25_qc.eq("valid").sum()),
                "valid_pm25_pct_of_source_rows": 100.0
                * original.pm25_qc.eq("valid").mean(),
                "negative_pm25": int(original.pm25_qc.eq("negative").sum()),
                "pm25_at_or_above_985": int(
                    original.pm25_qc.eq("at_or_above_985").sum()
                ),
                "invalid_relative_humidity": int(original.rh_qc.ne("valid").sum()),
                "invalid_temperature": int(original.temperature_qc.ne("valid").sum()),
                "valid_pm25_pct_of_expected_hours": 100.0
                * group.pm25_ug_m3.notna().sum()
                / expected,
            }
        )
        local = group.set_index("timestamp_utc")
        month_starts = pd.date_range(start.floor("D").replace(day=1), end, freq="MS", tz="UTC")
        for month_start in month_starts:
            month_end = month_start + pd.offsets.MonthBegin(1)
            expected_month = int(
                (min(end + pd.Timedelta(hours=1), month_end) - max(start, month_start))
                / pd.Timedelta(hours=1)
            )
            subset = local.loc[
                (local.index >= month_start) & (local.index < month_end)
            ]
            monthly_rows.append(
                {
                    "station_code": station_code,
                    "station_name": metadata_lookup.loc[source_file, "station_name"],
                    "month_utc": month_start.strftime("%Y-%m"),
                    "expected_hours_within_station_span": expected_month,
                    "observed_hours": int(subset.index.nunique()),
                    "valid_pm25_hours": int(subset.pm25_ug_m3.notna().sum()),
                    "valid_pm25_coverage_pct": (
                        100.0 * subset.pm25_ug_m3.notna().sum() / expected_month
                        if expected_month
                        else np.nan
                    ),
                }
            )

    quality = pd.DataFrame(quality_rows)
    monthly_quality = pd.DataFrame(monthly_rows)
    overall = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": len(observation_paths),
        "station_codes": int(clean.station_code.nunique()),
        "source_rows": len(raw),
        "unique_station_hours": len(clean),
        "duplicate_rows_affected": int(duplicate_mask.sum()),
        "duplicate_rows_removed": int(duplicate_removed_mask.sum()),
        "conflicting_duplicate_keys": conflicting_duplicate_keys,
        "timestamp_parse_failures": int(raw.timestamp_utc.isna().sum()),
        "valid_pm25": int(raw.pm25_qc.eq("valid").sum()),
        "valid_pm25_pct": 100.0 * raw.pm25_qc.eq("valid").mean(),
        "negative_pm25": int(raw.pm25_qc.eq("negative").sum()),
        "pm25_at_or_above_985": int(raw.pm25_qc.eq("at_or_above_985").sum()),
        "invalid_relative_humidity": int(raw.rh_qc.ne("valid").sum()),
        "invalid_temperature": int(raw.temperature_qc.ne("valid").sum()),
        "start_utc": clean.timestamp_utc.min().isoformat(),
        "end_utc": clean.timestamp_utc.max().isoformat(),
        "source_manifest": source_manifest,
    }
    return clean, quality, monthly_quality, overall


def _full_hourly_grid(observations: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    value_columns = ["pm25_ug_m3", "relative_humidity_pct", "temperature_c"]
    for station_code, group in observations.groupby("station_code", sort=True):
        index = pd.date_range(
            group.timestamp_utc.min(), group.timestamp_utc.max(), freq="h", tz="UTC"
        )
        values = (
            group.set_index("timestamp_utc")[value_columns]
            .sort_index()
            .reindex(index)
        )
        values.index.name = "timestamp_utc"
        values["station_code"] = station_code
        pieces.append(values.reset_index())
    return pd.concat(pieces, ignore_index=True)


def haversine_distance_km(
    latitude_a: np.ndarray,
    longitude_a: np.ndarray,
    latitude_b: np.ndarray,
    longitude_b: np.ndarray,
) -> np.ndarray:
    radius_km = 6371.0088
    lat_a = np.radians(latitude_a)
    lat_b = np.radians(latitude_b)
    delta_lat = lat_b - lat_a
    delta_lon = np.radians(longitude_b - longitude_a)
    value = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * radius_km * np.arcsin(np.sqrt(value))


def _add_network_features(
    issues: pd.DataFrame, metadata: pd.DataFrame, radius_km: float
) -> pd.DataFrame:
    station_order = metadata.station_code.tolist()
    station_index = {station: index for index, station in enumerate(station_order)}
    latitude = metadata.set_index("station_code").loc[station_order, "latitude"].to_numpy(float)
    longitude = metadata.set_index("station_code").loc[station_order, "longitude"].to_numpy(float)
    distances = haversine_distance_km(
        latitude[:, None], longitude[:, None], latitude[None, :], longitude[None, :]
    )
    weights = np.where(
        (distances > 0) & (distances <= radius_km), 1.0 / np.maximum(distances, 25.0), 0.0
    )
    issues["network_pm25_mean_ug_m3"] = np.nan
    issues["nearby_pm25_weighted_mean_ug_m3"] = np.nan
    issues["nearby_station_count"] = 0
    for _, positions in issues.groupby("timestamp_utc", sort=False).groups.items():
        positions = np.asarray(list(positions), dtype=int)
        station_positions = np.array(
            [station_index[value] for value in issues.loc[positions, "station_code"]]
        )
        values = np.full(len(station_order), np.nan)
        values[station_positions] = issues.loc[positions, "pm25_lag_0h"].to_numpy(float)
        valid = np.isfinite(values)
        if valid.sum() > 1:
            total = np.nansum(values)
            count = int(valid.sum())
            other_count = count - valid.astype(int)
            own_contribution = np.where(valid, values, 0.0)
            other_mean = (total - own_contribution) / np.maximum(other_count, 1)
            other_mean[other_count == 0] = np.nan
            issues.loc[positions, "network_pm25_mean_ug_m3"] = other_mean[
                station_positions
            ]
        for row_position, station_position in zip(positions, station_positions, strict=True):
            usable = valid & (weights[station_position] > 0)
            if usable.any():
                local_weights = weights[station_position, usable]
                issues.loc[row_position, "nearby_pm25_weighted_mean_ug_m3"] = float(
                    np.average(values[usable], weights=local_weights)
                )
                issues.loc[row_position, "nearby_station_count"] = int(usable.sum())
    return issues


def build_issue_features(
    observations: pd.DataFrame, metadata: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    grid = _full_hourly_grid(observations)
    grid = grid.sort_values(["station_code", "timestamp_utc"]).reset_index(drop=True)
    grouped = grid.groupby("station_code", sort=False)
    feature_config = config["features"]
    for lag in feature_config["pm25_lags_hours"]:
        grid[f"pm25_lag_{lag}h"] = grouped.pm25_ug_m3.shift(lag)
    for lag in feature_config["meteorology_lags_hours"]:
        grid[f"rh_lag_{lag}h"] = grouped.relative_humidity_pct.shift(lag)
        grid[f"temperature_lag_{lag}h"] = grouped.temperature_c.shift(lag)

    for window in feature_config["rolling_windows_hours"]:
        rolling = grouped.pm25_ug_m3.rolling(window=window, min_periods=1)
        grid[f"pm25_mean_{window}h"] = rolling.mean().reset_index(level=0, drop=True)
        grid[f"pm25_std_{window}h"] = rolling.std(ddof=0).reset_index(level=0, drop=True)
        grid[f"pm25_max_{window}h"] = rolling.max().reset_index(level=0, drop=True)
        count = rolling.count().reset_index(level=0, drop=True)
        grid[f"pm25_available_fraction_{window}h"] = count / float(window)

    valid_timestamp = grid.timestamp_utc.where(grid.pm25_ug_m3.notna())
    last_valid = valid_timestamp.groupby(grid.station_code).ffill()
    grid["latest_pm25_age_hours"] = (
        grid.timestamp_utc - last_valid
    ) / pd.Timedelta(hours=1)
    for horizon in config["forecast_hours"]:
        grid[f"target_pm25_{horizon}h"] = grouped.pm25_ug_m3.shift(-horizon)

    issues = grid.loc[
        grid.timestamp_utc.dt.hour.isin(config["forecast_cycle_hours_utc"])
    ].copy()
    issues = issues.merge(metadata, on="station_code", how="left", validate="many_to_one")
    local_issue = issues.timestamp_utc + pd.to_timedelta(
        issues.utc_offset_hours, unit="h"
    )
    issues["issue_hour_local_sin"] = np.sin(2 * np.pi * local_issue.dt.hour / 24.0)
    issues["issue_hour_local_cos"] = np.cos(2 * np.pi * local_issue.dt.hour / 24.0)
    issue_day = local_issue.dt.dayofyear
    issues["issue_day_of_year_sin"] = np.sin(2 * np.pi * issue_day / 365.25)
    issues["issue_day_of_year_cos"] = np.cos(2 * np.pi * issue_day / 365.25)
    issues["issue_year"] = local_issue.dt.year
    issues["issue_month"] = local_issue.dt.month
    issues = _add_network_features(
        issues.reset_index(drop=True),
        metadata,
        float(feature_config["network_radius_km"]),
    )
    return issues


def _select_variable(dataset: xr.Dataset, candidates: Iterable[str]) -> str:
    for candidate in candidates:
        if candidate in dataset.data_vars:
            return candidate
    normalized = {name.lower().replace("_", ""): name for name in dataset.data_vars}
    for candidate in candidates:
        key = candidate.lower().replace("_", "")
        if key in normalized:
            return normalized[key]
    raise KeyError(
        f"None of the CAMS variables {list(candidates)} are present; "
        f"available={list(dataset.data_vars)}"
    )


def _relative_humidity_from_temperature_and_dewpoint(
    temperature_k: pd.Series, dewpoint_k: pd.Series
) -> pd.Series:
    temperature_c = temperature_k - 273.15
    dewpoint_c = dewpoint_k - 273.15
    exponent = 17.625 * dewpoint_c / (243.04 + dewpoint_c) - (
        17.625 * temperature_c / (243.04 + temperature_c)
    )
    return (100.0 * np.exp(exponent)).clip(0, 100)


def sample_cams_at_stations(
    metadata: pd.DataFrame, paths: ExperimentPaths
) -> tuple[pd.DataFrame, dict[str, Any]]:
    earth_engine_paths = sorted(
        paths.cams_earth_engine.glob("cams_station_forecasts_*.csv.gz")
    )
    if earth_engine_paths:
        if len(earth_engine_paths) > 1:
            raise ValueError(
                "Multiple combined CAMS Earth Engine extracts are present; retain one "
                "explicit experiment period or select it in configuration"
            )
        source_path = earth_engine_paths[0]
        cams = pd.read_csv(
            source_path,
            parse_dates=["issue_time_utc", "valid_time_utc"],
            low_memory=False,
        )
        cams["issue_time_utc"] = pd.to_datetime(cams.issue_time_utc, utc=True)
        cams["valid_time_utc"] = pd.to_datetime(cams.valid_time_utc, utc=True)
        required = {
            "station_code",
            "sample_latitude",
            "sample_longitude",
            "issue_time_utc",
            "valid_time_utc",
            "forecast_hour",
            "cams_pm25_kg_m3",
            "cams_pm25_ug_m3",
            "source_image_id",
        }
        missing = required.difference(cams.columns)
        if missing:
            raise ValueError(
                f"CAMS Earth Engine station extract lacks columns: {sorted(missing)}"
            )
        if set(cams.station_code.unique()) != set(metadata.station_code.astype(str)):
            raise ValueError("CAMS Earth Engine station identifiers differ from metadata")
        coordinate_check = (
            cams[["station_code", "sample_latitude", "sample_longitude"]]
            .drop_duplicates()
            .merge(
                metadata[["station_code", "latitude", "longitude"]],
                on="station_code",
                how="outer",
                validate="one_to_one",
            )
        )
        coordinate_difference = np.maximum(
            (coordinate_check.sample_latitude - coordinate_check.latitude).abs(),
            (coordinate_check.sample_longitude - coordinate_check.longitude).abs(),
        )
        if coordinate_difference.max() > 1.0e-8:
            raise ValueError("CAMS Earth Engine samples use stale station coordinates")
        expected_valid = cams.issue_time_utc + pd.to_timedelta(
            cams.forecast_hour, unit="h"
        )
        if not expected_valid.eq(cams.valid_time_utc).all():
            raise ValueError("CAMS Earth Engine issue/lead/valid-time mismatch")
        conversion_error = (
            cams.cams_pm25_ug_m3 - cams.cams_pm25_kg_m3 * 1.0e9
        ).abs()
        if conversion_error.max() > 1.0e-8:
            raise ValueError("CAMS Earth Engine PM2.5 unit conversion mismatch")
        optional_columns = [
            "cams_temperature_c",
            "cams_dewpoint_c",
            "cams_relative_humidity_pct",
            "cams_u10_m_s",
            "cams_v10_m_s",
            "cams_wind_speed_10m_m_s",
            "cams_boundary_layer_height_m",
            "cams_total_precipitation_mm",
        ]
        for column in optional_columns:
            cams[column] = np.nan
        cams["source_netcdf"] = cams.source_image_id
        keep = [
            "station_code",
            "issue_time_utc",
            "valid_time_utc",
            "forecast_hour",
            "cams_pm25_ug_m3",
            *optional_columns,
            "source_netcdf",
        ]
        cams = cams[keep].sort_values(
            ["station_code", "issue_time_utc", "forecast_hour"]
        )
        duplicate_keys = cams.duplicated(
            ["station_code", "issue_time_utc", "forecast_hour"]
        )
        if duplicate_keys.any():
            raise ValueError(
                f"CAMS Earth Engine extract has {int(duplicate_keys.sum())} duplicate keys"
            )
        external_manifest_path = paths.cams_earth_engine / "manifest.json"
        external_manifest = (
            json.loads(external_manifest_path.read_text(encoding="utf-8"))
            if external_manifest_path.exists()
            else None
        )
        manifest = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "provider": "official Earth Engine mirror of ECMWF/CAMS/NRT",
            "interpolation": "bilinear at supplied station coordinates",
            "conversion": {
                "PM2.5": "kg m-3 multiplied by 1e9 to micrograms m-3"
            },
            "rows": len(cams),
            "stations": int(cams.station_code.nunique()),
            "issue_start_utc": cams.issue_time_utc.min().isoformat(),
            "issue_end_utc": cams.issue_time_utc.max().isoformat(),
            "forecast_hours": sorted(cams.forecast_hour.unique().tolist()),
            "source_files": [
                {
                    "path": str(source_path.relative_to(paths.root)),
                    "bytes": source_path.stat().st_size,
                    "sha256": file_sha256(source_path),
                }
            ],
            "external_acquisition_manifest": external_manifest,
        }
        return cams.reset_index(drop=True), manifest

    netcdf_paths = sorted(paths.cams_netcdf.glob("*.nc"))
    if not netcdf_paths:
        raise FileNotFoundError(
            "No CAMS NetCDF archives were found. Run scripts/acquire_external_data.py first."
        )
    frames: list[pd.DataFrame] = []
    file_metadata: list[dict[str, Any]] = []
    for netcdf_path in netcdf_paths:
        matching_codes = [
            code
            for code in metadata.station_code.astype(str)
            if f"cams_{code}_" in netcdf_path.name
        ]
        if len(matching_codes) > 1:
            raise ValueError(f"Ambiguous station code in CAMS filename: {netcdf_path.name}")
        file_metadata_rows = (
            metadata.loc[metadata.station_code.eq(matching_codes[0])]
            if matching_codes
            else metadata
        )
        file_station_coordinate = file_metadata_rows.station_code.to_numpy(str)
        file_latitude = xr.DataArray(
            file_metadata_rows.latitude.to_numpy(float),
            dims="station_code",
            coords={"station_code": file_station_coordinate},
        )
        file_longitude = xr.DataArray(
            file_metadata_rows.longitude.to_numpy(float),
            dims="station_code",
            coords={"station_code": file_station_coordinate},
        )
        with xr.open_dataset(netcdf_path) as dataset:
            required_coordinates = {
                "forecast_period",
                "forecast_reference_time",
                "latitude",
                "longitude",
            }
            missing_coordinates = required_coordinates.difference(dataset.coords)
            if missing_coordinates:
                raise ValueError(
                    f"{netcdf_path.name} lacks CAMS coordinates: {sorted(missing_coordinates)}"
                )
            selected = dataset.interp(
                latitude=file_latitude, longitude=file_longitude, method="linear"
            ).load()
            candidates = {
                "cams_pm25_kg_m3": ["pm2p5"],
                "cams_temperature_k": ["t2m", "2t"],
                "cams_dewpoint_k": ["d2m", "2d"],
                "cams_u10_m_s": ["u10", "10u"],
                "cams_v10_m_s": ["v10", "10v"],
                "cams_boundary_layer_height_m": ["blh"],
                "cams_total_precipitation_m": ["tp"],
            }
            names: dict[str, str] = {}
            for target, aliases in candidates.items():
                try:
                    names[target] = _select_variable(selected, aliases)
                except KeyError:
                    if target == "cams_pm25_kg_m3":
                        raise
            pm_units = str(
                selected[names["cams_pm25_kg_m3"]].attrs.get("units", "")
            )
            normalized_pm_units = (
                pm_units.lower()
                .replace(" ", "")
                .replace("**", "^")
                .replace("−", "-")
            )
            if normalized_pm_units not in {
                "kgm^-3",
                "kgm-3",
                "kg/m^3",
                "kg/m3",
            }:
                raise ValueError(
                    f"Unexpected CAMS PM2.5 units in {netcdf_path.name}: {pm_units!r}"
                )
            subset = selected[list(names.values())].rename(
                {source: target for target, source in names.items()}
            )
            frame = subset.to_dataframe().reset_index()
            for optional in set(candidates).difference(names):
                frame[optional] = np.nan
            frame["source_netcdf"] = netcdf_path.name
            frames.append(frame)
            file_metadata.append(
                {
                    "path": str(netcdf_path.relative_to(paths.root)),
                    "bytes": netcdf_path.stat().st_size,
                    "sha256": file_sha256(netcdf_path),
                    "dimensions": {name: int(size) for name, size in dataset.sizes.items()},
                    "variables": {
                        target: {
                            "source_name": source,
                            "source_units": str(dataset[source].attrs.get("units", "")),
                        }
                        for target, source in names.items()
                    },
                }
            )
    cams = pd.concat(frames, ignore_index=True)
    cams["issue_time_utc"] = pd.to_datetime(
        cams["forecast_reference_time"], utc=True
    )
    cams["forecast_hour"] = (
        pd.to_timedelta(cams["forecast_period"]) / pd.Timedelta(hours=1)
    ).astype(int)
    cams["valid_time_utc"] = cams.issue_time_utc + pd.to_timedelta(
        cams.forecast_hour, unit="h"
    )
    cams["cams_pm25_ug_m3"] = cams.cams_pm25_kg_m3 * 1.0e9
    cams["cams_temperature_c"] = cams.cams_temperature_k - 273.15
    cams["cams_dewpoint_c"] = cams.cams_dewpoint_k - 273.15
    cams["cams_relative_humidity_pct"] = _relative_humidity_from_temperature_and_dewpoint(
        cams.cams_temperature_k, cams.cams_dewpoint_k
    )
    cams["cams_wind_speed_10m_m_s"] = np.hypot(
        cams.cams_u10_m_s, cams.cams_v10_m_s
    )
    cams["cams_total_precipitation_mm"] = cams.cams_total_precipitation_m * 1000.0
    keep = [
        "station_code",
        "issue_time_utc",
        "valid_time_utc",
        "forecast_hour",
        "cams_pm25_ug_m3",
        "cams_temperature_c",
        "cams_dewpoint_c",
        "cams_relative_humidity_pct",
        "cams_u10_m_s",
        "cams_v10_m_s",
        "cams_wind_speed_10m_m_s",
        "cams_boundary_layer_height_m",
        "cams_total_precipitation_mm",
        "source_netcdf",
    ]
    cams = cams[keep].sort_values(
        ["station_code", "issue_time_utc", "forecast_hour"]
    )
    duplicated = cams.duplicated(
        ["station_code", "issue_time_utc", "forecast_hour"], keep=False
    )
    if duplicated.any():
        duplicate_groups = cams.loc[duplicated].groupby(
            ["station_code", "issue_time_utc", "forecast_hour"]
        )
        value_columns = [column for column in keep if column.startswith("cams_")]
        conflicts = duplicate_groups[value_columns].nunique(dropna=False).gt(1).any(axis=1)
        if conflicts.any():
            raise ValueError(f"CAMS files contain {int(conflicts.sum())} conflicting keys")
        cams = cams.drop_duplicates(
            ["station_code", "issue_time_utc", "forecast_hour"], keep="last"
        )
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "interpolation": "bilinear in latitude-longitude coordinates",
        "conversion": {
            "PM2.5": "kg m-3 multiplied by 1e9 to micrograms m-3",
            "temperature_and_dewpoint": "K minus 273.15 to degrees Celsius",
            "relative_humidity": "Magnus approximation from 2 m temperature and dew point",
            "wind_speed": "Euclidean magnitude of 10 m u and v components",
            "precipitation": "m multiplied by 1000 to mm; source accumulation semantics retained",
        },
        "rows": len(cams),
        "stations": int(cams.station_code.nunique()),
        "issue_start_utc": cams.issue_time_utc.min().isoformat(),
        "issue_end_utc": cams.issue_time_utc.max().isoformat(),
        "forecast_hours": sorted(cams.forecast_hour.unique().tolist()),
        "source_files": file_metadata,
    }
    return cams, manifest


def assign_split(target_time: pd.Series, config: dict[str, Any]) -> pd.Series:
    split = pd.Series("outside", index=target_time.index, dtype="string")
    periods = config["splits"]
    ranges = {
        "train": (
            pd.Timestamp(periods["training_target_start_utc"]),
            pd.Timestamp(periods["training_target_end_utc"]),
        ),
        "validation": (
            pd.Timestamp(periods["validation_target_start_utc"]),
            pd.Timestamp(periods["validation_target_end_utc"]),
        ),
        "test": (
            pd.Timestamp(periods["test_target_start_utc"]),
            pd.Timestamp(periods["test_target_end_utc"]),
        ),
    }
    for name, (start, end) in ranges.items():
        split.loc[target_time.between(start, end, inclusive="both")] = name
    return split


def build_modeling_table(
    issues: pd.DataFrame,
    cams: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    target_columns = [
        f"target_pm25_{horizon}h" for horizon in config["forecast_hours"]
    ]
    base_columns = [column for column in issues.columns if column not in target_columns]
    for horizon in config["forecast_hours"]:
        frame = issues[base_columns + [f"target_pm25_{horizon}h"]].copy()
        frame = frame.rename(columns={f"target_pm25_{horizon}h": "target_pm25_ug_m3"})
        frame["forecast_hour"] = int(horizon)
        frame["target_time_utc"] = frame.timestamp_utc + pd.to_timedelta(
            horizon, unit="h"
        )
        frames.append(frame)
    modeling = pd.concat(frames, ignore_index=True)
    modeling = modeling.rename(columns={"timestamp_utc": "issue_time_utc"})
    modeling = modeling.merge(
        cams,
        on=["station_code", "issue_time_utc", "forecast_hour"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_cams"),
    )
    valid_time_mismatch = modeling.valid_time_utc.notna() & modeling.valid_time_utc.ne(
        modeling.target_time_utc
    )
    if valid_time_mismatch.any():
        raise ValueError(
            f"CAMS valid time differs from target time for {int(valid_time_mismatch.sum())} rows"
        )
    local_target = modeling.target_time_utc + pd.to_timedelta(
        modeling.utc_offset_hours, unit="h"
    )
    modeling["target_hour_local"] = local_target.dt.hour
    modeling["target_hour_local_sin"] = np.sin(2 * np.pi * local_target.dt.hour / 24.0)
    modeling["target_hour_local_cos"] = np.cos(2 * np.pi * local_target.dt.hour / 24.0)
    target_day = local_target.dt.dayofyear
    modeling["target_day_of_year_sin"] = np.sin(2 * np.pi * target_day / 365.25)
    modeling["target_day_of_year_cos"] = np.cos(2 * np.pi * target_day / 365.25)
    modeling["target_month"] = local_target.dt.month
    modeling["split"] = assign_split(modeling.target_time_utc, config)
    modeling = modeling.loc[modeling.split.ne("outside")].copy()
    return modeling.sort_values(
        ["target_time_utc", "station_code", "forecast_hour"]
    ).reset_index(drop=True)


def validate_modeling_table(
    modeling: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    key = ["station_code", "issue_time_utc", "forecast_hour"]
    duplicate_keys = int(modeling.duplicated(key).sum())
    future_feature_columns = [
        column
        for column in modeling.columns
        if column.startswith("pm25_lag_") and not column.endswith("0h")
    ]
    lag_violations = 0
    for column in future_feature_columns:
        lag = int(column.removeprefix("pm25_lag_").removesuffix("h"))
        if lag < 0:
            lag_violations += 1
    overlap = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left_times = set(modeling.loc[modeling.split.eq(left), "target_time_utc"])
        right_times = set(modeling.loc[modeling.split.eq(right), "target_time_utc"])
        overlap[f"{left}_{right}_target_time_overlap"] = len(left_times & right_times)
    required_cams_columns = ["cams_pm25_ug_m3"]
    report = {
        "rows": len(modeling),
        "stations": int(modeling.station_code.nunique()),
        "forecast_hours": sorted(modeling.forecast_hour.unique().tolist()),
        "duplicate_keys": duplicate_keys,
        "negative_lag_feature_definitions": lag_violations,
        "target_before_or_at_issue": int(
            modeling.target_time_utc.le(modeling.issue_time_utc).sum()
        ),
        "valid_target_rows": int(modeling.target_pm25_ug_m3.notna().sum()),
        "valid_target_pct": 100.0 * modeling.target_pm25_ug_m3.notna().mean(),
        "cams_complete_rows": int(
            modeling[required_cams_columns].notna().all(axis=1).sum()
        ),
        "cams_complete_pct": 100.0
        * modeling[required_cams_columns].notna().all(axis=1).mean(),
        "cams_complete_by_split_pct": (
            modeling.assign(
                cams_complete=modeling[required_cams_columns].notna().all(axis=1)
            )
            .groupby("split", observed=True)
            .cams_complete.mean()
            .mul(100.0)
            .to_dict()
        ),
        "split_rows": modeling.groupby("split", observed=True).size().to_dict(),
        "split_valid_targets": modeling.loc[modeling.target_pm25_ug_m3.notna()]
        .groupby("split", observed=True)
        .size()
        .to_dict(),
        **overlap,
    }
    if duplicate_keys or lag_violations or report["target_before_or_at_issue"]:
        raise ValueError(f"Modeling-table leakage/key audit failed: {report}")
    return report


def prepare_all(paths: ExperimentPaths | None = None) -> dict[str, Any]:
    paths = paths or ExperimentPaths()
    config = load_config(paths.config)
    for directory in (paths.derived, paths.tables, paths.provenance):
        directory.mkdir(parents=True, exist_ok=True)

    observations, quality, monthly_quality, observation_manifest = (
        load_and_quality_control_observations(config, paths)
    )
    observation_columns = [
        "source_file",
        "station_code",
        "station_name",
        "province",
        "region",
        "latitude",
        "longitude",
        "timezone",
        "utc_offset_hours",
        "timestamp_utc",
        "timestamp_local",
        "PM25",
        "pm25_qc",
        "pm25_ug_m3",
        "RH",
        "rh_qc",
        "relative_humidity_pct",
        "TEMP",
        "temperature_qc",
        "temperature_c",
    ]
    observations[observation_columns].to_csv(
        paths.derived / "observations_quality_controlled.csv.gz",
        index=False,
        compression="gzip",
    )
    quality.to_csv(paths.tables / "data_quality_by_station.csv", index=False)
    monthly_quality.to_csv(paths.tables / "monthly_station_coverage.csv", index=False)
    station_distribution = (
        observations.loc[observations.pm25_ug_m3.notna()]
        .groupby(["station_code", "station_name"], as_index=False)
        .pm25_ug_m3.agg(
            n="count",
            mean_ug_m3="mean",
            median_ug_m3="median",
            q90_ug_m3=lambda value: value.quantile(0.90),
            q95_ug_m3=lambda value: value.quantile(0.95),
            q99_ug_m3=lambda value: value.quantile(0.99),
            maximum_ug_m3="max",
        )
    )
    station_distribution.to_csv(
        paths.tables / "station_pm25_distribution.csv", index=False
    )
    write_json(paths.provenance / "observation_manifest.json", observation_manifest)

    metadata = pd.read_csv(paths.station_metadata)
    issues = build_issue_features(observations, metadata, config)
    issues.to_csv(
        paths.derived / "issue_time_observation_features.csv.gz",
        index=False,
        compression="gzip",
    )
    cams, cams_manifest = sample_cams_at_stations(metadata, paths)
    cams.to_csv(
        paths.derived / "cams_station_forecasts.csv.gz",
        index=False,
        compression="gzip",
    )
    write_json(paths.provenance / "cams_sampling_manifest.json", cams_manifest)
    modeling = build_modeling_table(issues, cams, config)
    audit = validate_modeling_table(modeling, config)
    modeling_coverage = (
        modeling.assign(
            target_available=modeling.target_pm25_ug_m3.notna(),
            cams_available=modeling.cams_pm25_ug_m3.notna(),
        )
        .groupby(["split", "forecast_hour", "station_code"], as_index=False)
        .agg(
            issue_rows=("issue_time_utc", "size"),
            valid_targets=("target_available", "sum"),
            cams_rows=("cams_available", "sum"),
        )
    )
    modeling_coverage["valid_target_pct"] = (
        100.0 * modeling_coverage.valid_targets / modeling_coverage.issue_rows
    )
    modeling_coverage["cams_coverage_pct"] = (
        100.0 * modeling_coverage.cams_rows / modeling_coverage.issue_rows
    )
    modeling_coverage.to_csv(
        paths.tables / "modeling_coverage_by_station_split_lead.csv", index=False
    )
    modeling.to_csv(
        paths.derived / "modeling_table.csv.gz", index=False, compression="gzip"
    )
    write_json(paths.provenance / "modeling_table_audit.json", audit)
    environment = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "xarray": xr.__version__,
    }
    write_json(paths.provenance / "preparation_environment.json", environment)
    return {
        "observations": observation_manifest,
        "cams": cams_manifest,
        "modeling_table": audit,
    }
