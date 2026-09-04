"""Prospective, non-public shadow forecasting and delayed verification."""

from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

from .data import (
    AQ_ROOT,
    ExperimentPaths,
    build_issue_features,
    file_sha256,
    load_config,
    write_json,
)
from .deployment import run_operational_forecast
from .modeling import metric_values


DASHBOARD_URL = "https://cews.bmkg.go.id/tempatirk/TEMPORARY/dashboard_pm2p5.html"
CAMS_DATASET = "cams-global-atmospheric-composition-forecasts"
CAMS_SOURCE_URL = (
    "https://ads.atmosphere.copernicus.eu/datasets/"
    "cams-global-atmospheric-composition-forecasts?tab=overview"
)


def _atomic_csv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def _load_local_environment() -> None:
    env_path = AQ_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def fetch_dashboard_payload(url: str = DASHBOARD_URL) -> dict[str, Any]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "BMKG-PM25-ML-shadow/1.0"}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        html = response.read().decode("utf-8")
    marker = "const dashboardData = "
    start = html.find(marker)
    if start < 0:
        raise ValueError("Official dashboard does not contain dashboardData")
    payload, _ = json.JSONDecoder().raw_decode(html[start + len(marker) :])
    if not isinstance(payload.get("locations"), dict):
        raise ValueError("Official dashboard payload lacks a locations object")
    return payload


def _normalise_station_name(value: str) -> str:
    return "".join(character for character in value.upper() if character.isalnum())


def parse_dashboard_observations(
    payload: dict[str, Any],
    metadata: pd.DataFrame,
    retrieved_utc: pd.Timestamp,
) -> pd.DataFrame:
    """Parse local dashboard timestamps to UTC with explicit station offsets."""

    location_lookup = {
        _normalise_station_name(name): (name, location)
        for name, location in payload["locations"].items()
    }
    rows: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for station in metadata.itertuples(index=False):
        candidates = {
            _normalise_station_name(Path(station.source_file).stem),
            _normalise_station_name(station.station_name),
        }
        matches = [location_lookup[candidate] for candidate in candidates if candidate in location_lookup]
        if len({match[0] for match in matches}) != 1:
            unmatched.append(str(station.station_code))
            continue
        dashboard_name, location = matches[0]
        reported_timezone = str(location.get("latest", {}).get("timezone", ""))
        if reported_timezone and reported_timezone != str(station.timezone):
            raise ValueError(
                f"Dashboard timezone changed for {station.station_code}: "
                f"{reported_timezone} != {station.timezone}"
            )
        labels = location.get("timeseries", {}).get("labels", [])
        values = location.get("timeseries", {}).get("values", [])
        if len(labels) != len(values):
            raise ValueError(f"Dashboard time/value length mismatch for {dashboard_name}")
        local_times = pd.to_datetime(pd.Series(labels), errors="coerce")
        for local_time, value in zip(local_times, values, strict=True):
            if pd.isna(local_time):
                continue
            numeric = pd.to_numeric(value, errors="coerce")
            qc = "valid"
            if pd.isna(numeric):
                qc = "missing_or_nonnumeric"
            elif float(numeric) < 0:
                qc = "negative"
            elif float(numeric) >= 985:
                qc = "at_or_above_985"
            timestamp_utc = (
                pd.Timestamp(local_time)
                - pd.Timedelta(hours=float(station.utc_offset_hours))
            ).tz_localize("UTC")
            rows.append(
                {
                    "station_code": str(station.station_code),
                    "station_name": str(station.station_name),
                    "dashboard_name": dashboard_name,
                    "timestamp_utc": timestamp_utc,
                    "pm25_ug_m3": float(numeric) if qc == "valid" else np.nan,
                    "pm25_qc": qc,
                    "timezone": str(station.timezone),
                    "utc_offset_hours": float(station.utc_offset_hours),
                    "source_retrieved_utc": retrieved_utc,
                    "source_url": DASHBOARD_URL,
                }
            )
    if unmatched:
        raise ValueError(f"Dashboard station reconciliation failed: {unmatched}")
    result = pd.DataFrame(rows)
    if result.station_code.nunique() != len(metadata):
        raise ValueError("Dashboard observations do not cover all configured stations")
    return result.sort_values(["station_code", "timestamp_utc"]).reset_index(drop=True)


def acquire_dashboard_observations(
    paths: ExperimentPaths,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    retrieved = pd.Timestamp.now(tz="UTC")
    payload = fetch_dashboard_payload()
    snapshot_dir = paths.root / "shadow" / "inputs" / "observations" / "raw"
    snapshot = snapshot_dir / f"dashboard_{retrieved:%Y%m%dT%H%M%SZ}.json"
    write_json(snapshot, payload)
    metadata = pd.read_csv(paths.station_metadata)
    parsed = parse_dashboard_observations(payload, metadata, retrieved)
    archive_path = paths.root / "shadow" / "inputs" / "observations" / "dashboard_hourly.csv.gz"
    if archive_path.exists():
        existing = pd.read_csv(
            archive_path,
            parse_dates=["timestamp_utc", "source_retrieved_utc"],
            low_memory=False,
        )
        combined = pd.concat([existing, parsed], ignore_index=True)
    else:
        combined = parsed
    key = ["station_code", "timestamp_utc"]
    conflicting = (
        combined.groupby(key, observed=True).pm25_ug_m3.nunique(dropna=False).gt(1)
    )
    revisions = int(conflicting.sum())
    first_arrival = (
        combined.sort_values("source_retrieved_utc")
        .drop_duplicates(key, keep="first")
        .sort_values(key)
        .reset_index(drop=True)
    )
    _atomic_csv(first_arrival, archive_path, compression="gzip")
    manifest = {
        "source_name": "BMKG CEWS PM2.5 dashboard",
        "source_url": DASHBOARD_URL,
        "retrieved_utc": retrieved.isoformat(),
        "snapshot": str(snapshot.relative_to(paths.root)),
        "snapshot_sha256": file_sha256(snapshot),
        "parsed_rows_this_run": len(parsed),
        "archive_rows": len(first_arrival),
        "stations": int(first_arrival.station_code.nunique()),
        "first_timestamp_utc": first_arrival.timestamp_utc.min().isoformat(),
        "latest_timestamp_utc": first_arrival.timestamp_utc.max().isoformat(),
        "conflicting_station_hours_seen": revisions,
        "revision_policy": "retain first observed value; preserve every raw snapshot",
        "archive_sha256": file_sha256(archive_path),
    }
    write_json(archive_path.parent / "manifest.json", manifest)
    return first_arrival, manifest


def build_shadow_issue_features(
    paths: ExperimentPaths,
    dashboard_observations: pd.DataFrame,
    issue_time: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = load_config(paths.config)
    historical_path = paths.derived / "observations_quality_controlled.csv.gz"
    history = pd.read_csv(
        historical_path,
        usecols=[
            "station_code",
            "timestamp_utc",
            "pm25_ug_m3",
            "relative_humidity_pct",
            "temperature_c",
        ],
        parse_dates=["timestamp_utc"],
        low_memory=False,
    )
    history["timestamp_utc"] = pd.to_datetime(history.timestamp_utc, utc=True)
    window_start = issue_time - pd.Timedelta(days=10)
    history = history.loc[
        history.timestamp_utc.between(window_start, issue_time, inclusive="both")
    ].copy()
    dashboard = dashboard_observations.loc[
        dashboard_observations.timestamp_utc.between(
            window_start, issue_time, inclusive="both"
        )
    ][["station_code", "timestamp_utc", "pm25_ug_m3"]].copy()
    dashboard["relative_humidity_pct"] = np.nan
    dashboard["temperature_c"] = np.nan
    combined = pd.concat([history, dashboard], ignore_index=True)
    combined = (
        combined.sort_values(["station_code", "timestamp_utc"])
        .drop_duplicates(["station_code", "timestamp_utc"], keep="last")
        .reset_index(drop=True)
    )
    metadata = pd.read_csv(paths.station_metadata)
    present = set(
        combined.loc[combined.timestamp_utc.eq(issue_time), "station_code"].astype(str)
    )
    missing_issue_rows = metadata.loc[~metadata.station_code.astype(str).isin(present)]
    if len(missing_issue_rows):
        placeholders = pd.DataFrame(
            {
                "station_code": missing_issue_rows.station_code.astype(str),
                "timestamp_utc": issue_time,
                "pm25_ug_m3": np.nan,
                "relative_humidity_pct": np.nan,
                "temperature_c": np.nan,
            }
        )
        combined = pd.concat([combined, placeholders], ignore_index=True)
    features = build_issue_features(combined, metadata, config)
    selected = features.loc[features.timestamp_utc.eq(issue_time)].copy()
    if len(selected) != len(metadata) or selected.station_code.nunique() != len(metadata):
        raise ValueError("Shadow feature construction did not produce one row per station")
    output = (
        paths.root
        / "shadow"
        / "inputs"
        / "features"
        / f"issue_features_{issue_time:%Y%m%dT%H%MZ}.csv.gz"
    )
    _atomic_csv(selected, output, compression="gzip")
    manifest = {
        "issue_time_utc": issue_time.isoformat(),
        "observation_snapshot_cutoff_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "rows": len(selected),
        "stations": int(selected.station_code.nunique()),
        "latest_pm25_age_hours_median": float(selected.latest_pm25_age_hours.median()),
        "latest_pm25_age_hours_maximum": float(selected.latest_pm25_age_hours.max()),
        "stale_stations_over_6_hours": int(selected.latest_pm25_age_hours.gt(6).sum()),
        "historical_source_sha256": file_sha256(historical_path),
        "output": str(output.relative_to(paths.root)),
        "output_sha256": file_sha256(output),
        "meteorology_note": (
            "The current dashboard contributes PM2.5 only; temperature and humidity "
            "remain missing after the historical archive endpoint."
        ),
    }
    write_json(output.with_suffix(".json"), manifest)
    return selected, manifest


def _sample_direct_cams_archive(
    archive: Path,
    metadata: pd.DataFrame,
    issue_time: pd.Timestamp,
    forecast_hours: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pm25-shadow-cams-") as temporary:
        with zipfile.ZipFile(archive) as bundle:
            bad_member = bundle.testzip()
            if bad_member is not None:
                raise ValueError(f"CAMS archive member is corrupt: {bad_member}")
            bundle.extractall(temporary)
        netcdf_paths = sorted(Path(temporary).glob("*.nc"))
        if not netcdf_paths:
            raise ValueError("CAMS archive contains no NetCDF file")
        for netcdf_path in netcdf_paths:
            with xr.open_dataset(netcdf_path) as dataset:
                if "pm2p5" not in dataset:
                    continue
                units = str(dataset.pm2p5.attrs.get("units", "")).replace(" ", "")
                if units not in {"kgm**-3", "kgm-3"}:
                    raise ValueError(f"Unexpected CAMS PM2.5 units: {units}")
                reference_values = pd.to_datetime(dataset.forecast_reference_time.values)
                matching_reference = [
                    value
                    for value in reference_values
                    if pd.Timestamp(value, tz="UTC") == issue_time
                ]
                if len(matching_reference) != 1:
                    continue
                reference_value = matching_reference[0].to_datetime64()
                for period_value in dataset.forecast_period.values:
                    horizon = int(pd.Timedelta(period_value) / pd.Timedelta(hours=1))
                    if horizon not in forecast_hours:
                        continue
                    field = dataset.pm2p5.sel(
                        forecast_reference_time=reference_value,
                        forecast_period=period_value,
                    )
                    for station in metadata.itertuples(index=False):
                        value = float(
                            field.interp(
                                latitude=float(station.latitude),
                                longitude=float(station.longitude),
                                method="linear",
                            ).item()
                        )
                        rows.append(
                            {
                                "station_code": str(station.station_code),
                                "issue_time_utc": issue_time,
                                "valid_time_utc": issue_time + pd.Timedelta(hours=horizon),
                                "forecast_hour": horizon,
                                "cams_pm25_ug_m3": value * 1.0e9,
                                "source_archive": archive.name,
                            }
                        )
    frame = pd.DataFrame(rows)
    expected = len(metadata) * len(forecast_hours)
    if len(frame) != expected or frame.duplicated(
        ["station_code", "issue_time_utc", "forecast_hour"]
    ).any():
        raise ValueError(f"CAMS station sampling returned {len(frame)} of {expected} rows")
    if frame.cams_pm25_ug_m3.isna().any() or frame.cams_pm25_ug_m3.lt(0).any():
        raise ValueError("CAMS station sampling contains missing or negative values")
    return frame.sort_values(["station_code", "forecast_hour"]).reset_index(drop=True)


def acquire_direct_cams(
    paths: ExperimentPaths,
    issue_time: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _load_local_environment()
    key = os.environ.get("AQ_ADS_KEY")
    if not key:
        raise RuntimeError("AQ_ADS_KEY is not configured")
    config = load_config(paths.config)
    forecast_hours = [int(value) for value in config["forecast_hours"]]
    output_dir = paths.root / "shadow" / "inputs" / "cams"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"cams_direct_{issue_time:%Y%m%dT%H%MZ}.zip"
    request = {
        "variable": ["particulate_matter_2.5um"],
        "date": issue_time.strftime("%Y-%m-%d"),
        "time": [issue_time.strftime("%H:00")],
        "leadtime_hour": [str(value) for value in forecast_hours],
        "type": "forecast",
        "area": config["cams"]["area_north_west_south_east"],
        "data_format": "netcdf_zip",
    }
    if not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        client = cdsapi.Client(
            url=os.environ.get(
                "AQ_ADS_URL", "https://ads.atmosphere.copernicus.eu/api"
            ),
            key=key,
            quiet=True,
        )
        client.retrieve(CAMS_DATASET, request, str(temporary))
        if not zipfile.is_zipfile(temporary):
            raise ValueError("CAMS response is not a valid ZIP archive")
        os.replace(temporary, archive)
    metadata = pd.read_csv(paths.station_metadata)
    sampled = _sample_direct_cams_archive(
        archive, metadata, issue_time, forecast_hours
    )
    sample_path = output_dir / f"cams_station_{issue_time:%Y%m%dT%H%MZ}.csv.gz"
    _atomic_csv(sampled, sample_path, compression="gzip")
    manifest = {
        "source_name": "CAMS global atmospheric composition forecasts",
        "source_url": CAMS_SOURCE_URL,
        "dataset": CAMS_DATASET,
        "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "issue_time_utc": issue_time.isoformat(),
        "request_without_credentials": request,
        "archive": str(archive.relative_to(paths.root)),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": file_sha256(archive),
        "station_rows": len(sampled),
        "stations": int(sampled.station_code.nunique()),
        "forecast_hours": forecast_hours,
        "sampling": "linear interpolation on the direct CAMS latitude-longitude grid",
        "unit_conversion": "kg m-3 multiplied by 1e9 to micrograms m-3",
        "station_output": str(sample_path.relative_to(paths.root)),
        "station_output_sha256": file_sha256(sample_path),
    }
    write_json(sample_path.with_suffix(".json"), manifest)
    return sampled, manifest


def verify_shadow_forecasts(
    paths: ExperimentPaths,
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast_paths = sorted((paths.root / "shadow" / "forecasts").glob("*.csv"))
    if not forecast_paths:
        return pd.DataFrame(), pd.DataFrame()
    forecasts = pd.concat(
        [
            pd.read_csv(
                path,
                parse_dates=["issue_time_utc", "target_time_utc", "generated_utc"],
                low_memory=False,
            )
            for path in forecast_paths
        ],
        ignore_index=True,
    )
    observations = observations[["station_code", "timestamp_utc", "pm25_ug_m3"]].rename(
        columns={"timestamp_utc": "target_time_utc", "pm25_ug_m3": "observed_pm25_ug_m3"}
    )
    matched = forecasts.merge(
        observations,
        on=["station_code", "target_time_utc"],
        how="inner",
        validate="many_to_one",
    )
    matched = matched.loc[matched.observed_pm25_ug_m3.notna()].copy()
    if matched.empty:
        return matched, pd.DataFrame()
    matched["forecast_error_ug_m3"] = (
        matched.forecast_pm25_ug_m3 - matched.observed_pm25_ug_m3
    )
    matched["forecast_absolute_error_ug_m3"] = matched.forecast_error_ug_m3.abs()
    matched["persistence_absolute_error_ug_m3"] = (
        matched.pm25_lag_0h - matched.observed_pm25_ug_m3
    ).abs()
    matched["raw_cams_absolute_error_ug_m3"] = (
        matched.cams_pm25_ug_m3 - matched.observed_pm25_ug_m3
    ).abs()
    matched["interval_covered"] = (
        matched.observed_pm25_ug_m3.ge(matched.prediction_q10_ug_m3)
        & matched.observed_pm25_ug_m3.le(matched.prediction_q90_ug_m3)
    ).where(matched.prediction_q10_ug_m3.notna())
    verification_dir = paths.root / "shadow" / "verification"
    matched_path = verification_dir / "matched_forecasts.csv.gz"
    _atomic_csv(matched.sort_values(["target_time_utc", "station_code", "forecast_hour"]), matched_path, compression="gzip")
    rows: list[dict[str, Any]] = []
    for horizon, group in matched.groupby("forecast_hour", observed=True):
        forecast_metrics = metric_values(group.observed_pm25_ug_m3, group.forecast_pm25_ug_m3)
        persistence_metrics = metric_values(group.observed_pm25_ug_m3, group.pm25_lag_0h)
        prospective = group.generation_status.eq("prospective")
        rows.append(
            {
                "forecast_hour": int(horizon),
                "n": len(group),
                "prospective_n": int(prospective.sum()),
                "stations": int(group.station_code.nunique()),
                "forecast_mae_ug_m3": forecast_metrics["mae_ug_m3"],
                "persistence_mae_ug_m3": persistence_metrics["mae_ug_m3"],
                "skill_vs_persistence_pct": (
                    100.0 * (1.0 - forecast_metrics["mae_ug_m3"] / persistence_metrics["mae_ug_m3"])
                    if persistence_metrics["mae_ug_m3"]
                    else np.nan
                ),
                "forecast_bias_ug_m3": forecast_metrics["bias_ug_m3"],
                "interval_coverage_pct": 100.0 * group.interval_covered.mean(),
                "primary_fraction_pct": 100.0 * group.forecast_status.str.startswith("primary").mean(),
            }
        )
    summary = pd.DataFrame(rows).sort_values("forecast_hour")
    _atomic_csv(summary, verification_dir / "scorecard_by_lead.csv")
    return matched, summary


def run_daily_shadow(
    issue_date: str | None = None,
    paths: ExperimentPaths | None = None,
) -> dict[str, Any]:
    paths = paths or ExperimentPaths()
    started = time.perf_counter()
    generated = pd.Timestamp.now(tz="UTC")
    issue_time = (
        pd.Timestamp(issue_date, tz="UTC")
        if issue_date is not None
        else generated.floor("D")
    )
    if issue_time.hour != 0:
        raise ValueError("Shadow workflow currently supports the validated 00 UTC cycle only")
    shadow = paths.root / "shadow"
    (shadow / "logs").mkdir(parents=True, exist_ok=True)
    observations, observation_manifest = acquire_dashboard_observations(paths)
    forecast_path = shadow / "forecasts" / f"pm25_shadow_{issue_time:%Y%m%dT%H%MZ}.csv"
    warnings: list[str] = []
    if forecast_path.exists():
        forecast = pd.read_csv(
            forecast_path,
            parse_dates=["issue_time_utc", "target_time_utc", "generated_utc"],
            low_memory=False,
        )
        status = "forecast_already_exists"
    else:
        features, feature_manifest = build_shadow_issue_features(
            paths, observations, issue_time
        )
        cams_manifest: dict[str, Any] = {}
        cams_error: str | None = None
        cams = pd.DataFrame(
            columns=[
                "station_code",
                "issue_time_utc",
                "valid_time_utc",
                "forecast_hour",
                "cams_pm25_ug_m3",
            ]
        )
        for attempt in range(1, 4):
            try:
                cams, cams_manifest = acquire_direct_cams(paths, issue_time)
                cams_error = None
                break
            except Exception as error:  # preserve a degraded forecast on source outage
                cams_error = f"{type(error).__name__}: {error}"
                if attempt < 3:
                    time.sleep(30 * attempt)
        if cams_error:
            warnings.append(f"CAMS unavailable after three attempts: {cams_error}")
        forecast, metadata = run_operational_forecast(
            issue_time.isoformat(),
            output_path=forecast_path,
            paths=paths,
            issue_features_frame=features,
            cams_frame=cams,
        )
        forecast["generated_utc"] = generated
        forecast["availability_lag_hours"] = (
            generated - forecast.issue_time_utc
        ) / pd.Timedelta(hours=1)
        forecast["observation_age_at_generation_hours"] = (
            forecast.latest_pm25_age_hours + forecast.availability_lag_hours
        )
        forecast["generation_status"] = np.where(
            forecast.target_time_utc.gt(generated),
            "prospective",
                "target_time_reached_before_generation",
        )
        forecast["shadow_input_status"] = np.where(
            forecast.observation_age_at_generation_hours.gt(6),
            "observation_stale_at_generation",
            "observation_fresh_at_generation",
        )
        _atomic_csv(forecast, forecast_path)
        metadata.update(
            {
                "generated_utc": generated.isoformat(),
                "output_sha256": file_sha256(forecast_path),
                "prospective_rows": int(forecast.generation_status.eq("prospective").sum()),
                "late_rows": int(forecast.generation_status.ne("prospective").sum()),
                "stale_at_generation_rows": int(
                    forecast.shadow_input_status.eq(
                        "observation_stale_at_generation"
                    ).sum()
                ),
                "observation_manifest": observation_manifest,
                "feature_manifest": feature_manifest,
                "cams_manifest": cams_manifest,
                "warnings": warnings,
            }
        )
        write_json(forecast_path.with_suffix(".json"), metadata)
        status = "forecast_generated"
    matched, scorecard = verify_shadow_forecasts(paths, observations)
    run = {
        "run_id": f"shadow-{issue_time:%Y%m%dT%H%MZ}",
        "status": status,
        "generated_utc": generated.isoformat(),
        "issue_time_utc": issue_time.isoformat(),
        "forecast_rows": len(forecast),
        "forecast_sha256": file_sha256(forecast_path),
        "matched_verification_rows": len(matched),
        "scorecard_leads": len(scorecard),
        "warnings": warnings,
        "elapsed_seconds": time.perf_counter() - started,
        "automatic_retraining": False,
        "retraining_policy": (
            "Freeze the deployed model during prospective evaluation; review retraining "
            "after 60-90 days using a separately versioned candidate and untouched holdout."
        ),
    }
    write_json(shadow / "state" / "latest_run.json", run)
    first_success_path = shadow / "state" / "first_successful_run.json"
    if status == "forecast_generated" and not first_success_path.exists():
        write_json(first_success_path, run)
    log_path = shadow / "logs" / "runs.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(run, sort_keys=True) + "\n")
    return run
