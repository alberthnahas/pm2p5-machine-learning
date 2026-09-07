"""Prospective, non-public shadow forecasting and delayed verification."""

from __future__ import annotations

import json
import os
import signal
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
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


DASHBOARD_URL = "https://cews.bmkg.go.id/tempatirk/TEMPORARY/dashboard_pm2p5.html"
CAMS_DATASET = "cams-global-atmospheric-composition-forecasts"
CAMS_SOURCE_URL = (
    "https://ads.atmosphere.copernicus.eu/datasets/"
    "cams-global-atmospheric-composition-forecasts?tab=overview"
)
CAMS_REQUEST_TIMEOUT_SECONDS = 60
CAMS_REQUEST_RETRY_MAX = 2
CAMS_REQUEST_TOTAL_SECONDS = 300


def load_operational_config(paths: ExperimentPaths) -> dict[str, Any]:
    path = paths.root / "operational_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported operational configuration schema")
    return config


class CamsRetrievalTimeout(TimeoutError):
    """Raised when the entire ADS retrieval exceeds the local service budget."""


def _retrieve_with_total_timeout(
    client: cdsapi.Client,
    dataset: str,
    request: dict[str, Any],
    target: Path,
    timeout_seconds: int,
) -> None:
    if not hasattr(signal, "SIGALRM"):
        client.retrieve(dataset, request, str(target))
        return

    def _raise_timeout(signum: int, frame: Any) -> None:
        del signum, frame
        raise CamsRetrievalTimeout(
            f"ADS retrieval exceeded {timeout_seconds} seconds"
        )

    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(timeout_seconds)
    try:
        client.retrieve(dataset, request, str(target))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _cams_error_class(error: Exception) -> str:
    """Classify whether a same-run retry can plausibly succeed."""

    if isinstance(error, urllib.error.HTTPError):
        if error.code == 400:
            return "permanent_request_or_cycle_unavailable"
        if error.code in {408, 425, 429} or error.code >= 500:
            return "transient"
        return "permanent_http"
    message = str(error).lower()
    if "400 client error" in message or "invalid request" in message:
        return "permanent_request_or_cycle_unavailable"
    if isinstance(error, (TimeoutError, ConnectionError)):
        return "transient"
    if any(token in message for token in ("timed out", "temporarily unavailable", "connection")):
        return "transient"
    return "permanent_unknown"


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
    availability_cutoff_utc: pd.Timestamp | None = None,
    feature_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = feature_config or load_config(paths.config)
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
    availability_cutoff_utc = availability_cutoff_utc or pd.Timestamp.now(tz="UTC")
    if availability_cutoff_utc.tzinfo is None:
        raise ValueError("availability_cutoff_utc must be timezone-aware")
    available_dashboard = dashboard_observations.loc[
        pd.to_datetime(dashboard_observations.source_retrieved_utc, utc=True).le(
            availability_cutoff_utc
        )
    ]
    dashboard = available_dashboard.loc[
        available_dashboard.timestamp_utc.between(
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
        "observation_snapshot_cutoff_utc": availability_cutoff_utc.isoformat(),
        "dashboard_rows_excluded_after_cutoff": int(
            len(dashboard_observations) - len(available_dashboard)
        ),
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
    retrieval_started = pd.Timestamp.now(tz="UTC")
    cache_reused = archive.exists()
    if not cache_reused:
        temporary = archive.with_suffix(".zip.part")
        client = cdsapi.Client(
            url=os.environ.get(
                "AQ_ADS_URL", "https://ads.atmosphere.copernicus.eu/api"
            ),
            key=key,
            quiet=True,
            timeout=CAMS_REQUEST_TIMEOUT_SECONDS,
            retry_max=CAMS_REQUEST_RETRY_MAX,
            sleep_max=5,
        )
        try:
            _retrieve_with_total_timeout(
                client,
                CAMS_DATASET,
                request,
                temporary,
                CAMS_REQUEST_TOTAL_SECONDS,
            )
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        if not zipfile.is_zipfile(temporary):
            raise ValueError("CAMS response is not a valid ZIP archive")
        os.replace(temporary, archive)
    retrieval_completed = pd.Timestamp.now(tz="UTC")
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
        "retrieval_started_utc": retrieval_started.isoformat(),
        "retrieval_completed_utc": retrieval_completed.isoformat(),
        "cache_reused": cache_reused,
        "request_limits": {
            "network_timeout_seconds": CAMS_REQUEST_TIMEOUT_SECONDS,
            "network_retry_max": CAMS_REQUEST_RETRY_MAX,
            "overall_timeout_seconds": CAMS_REQUEST_TOTAL_SECONDS,
        },
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


def _read_shadow_runs(paths: ExperimentPaths) -> list[dict[str, Any]]:
    log_path = paths.root / "shadow" / "logs" / "runs.jsonl"
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def record_shadow_failure(
    issue_date: str | None,
    error: Exception,
    paths: ExperimentPaths | None = None,
) -> dict[str, Any]:
    """Persist a structured failed attempt so reliability rates include no-output runs."""

    paths = paths or ExperimentPaths()
    now = pd.Timestamp.now(tz="UTC")
    issue_time = pd.Timestamp(issue_date, tz="UTC") if issue_date else now.floor("D")
    record = {
        "run_id": f"shadow-{issue_time:%Y%m%dT%H%MZ}",
        "status": "run_failed",
        "recorded_utc": now.isoformat(),
        "issue_time_utc": issue_time.isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error)[:2000],
    }
    log_path = paths.root / "shadow" / "logs" / "runs.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    write_json(paths.root / "shadow" / "state" / "latest_run.json", record)
    return record


def _completion_evidence(
    forecast: pd.DataFrame,
    metadata: dict[str, Any],
    source_sha256: str,
    run_rows: list[dict[str, Any]],
) -> tuple[pd.Timestamp, str]:
    if "generation_completed_utc" in forecast.columns:
        values = pd.to_datetime(forecast.generation_completed_utc, utc=True).dropna()
        if values.nunique() == 1:
            return values.iloc[0], "exact_completion_timestamp"
    if metadata.get("generation_timestamp_semantics") == "forecast_write_completion":
        return (
            pd.Timestamp(metadata["generation_completed_utc"]),
            "exact_completion_timestamp",
        )
    candidates = [
        row
        for row in run_rows
        if row.get("status") == "forecast_generated"
        and row.get("forecast_sha256") == source_sha256
        and row.get("elapsed_seconds") is not None
    ]
    if candidates:
        row = candidates[0]
        upper_bound = pd.Timestamp(row["generated_utc"]) + pd.Timedelta(
            seconds=float(row["elapsed_seconds"])
        )
        return upper_bound, "conservative_run_end_upper_bound"
    internal_times: list[pd.Timestamp] = []
    for container, key in (
        (metadata, "generated_utc"),
        (metadata.get("observation_manifest", {}), "retrieved_utc"),
        (metadata.get("feature_manifest", {}), "observation_snapshot_cutoff_utc"),
        (metadata.get("cams_manifest", {}), "retrieved_utc"),
    ):
        if container.get(key):
            internal_times.append(pd.Timestamp(container[key]))
    if not internal_times:
        raise ValueError("Historical shadow forecast lacks completion-time evidence")
    return max(internal_times), "conservative_latest_internal_timestamp_lower_bound"


def _station_balanced_metrics(group: pd.DataFrame) -> dict[str, Any]:
    common = (
        group.observed_pm25_ug_m3.notna()
        & group.forecast_pm25_ug_m3.notna()
        & group.pm25_lag_0h.notna()
        & np.isfinite(group.observed_pm25_ug_m3)
        & np.isfinite(group.forecast_pm25_ug_m3)
        & np.isfinite(group.pm25_lag_0h)
    )
    paired = group.loc[common].copy()
    if paired.empty:
        return {
            "n": 0,
            "excluded_noncommon_rows": len(group),
            "stations": 0,
            "minimum_rows_per_station": 0,
            "station_balanced_forecast_mae_ug_m3": np.nan,
            "station_balanced_persistence_mae_ug_m3": np.nan,
            "skill_from_station_balanced_mae_pct": np.nan,
            "mean_per_station_skill_pct": np.nan,
            "stations_harmed": 0,
            "interval_coverage_pct": np.nan,
        }
    stations = (
        paired.groupby("station_code", observed=True)
        .agg(
            n=("observed_pm25_ug_m3", "size"),
            forecast_mae_ug_m3=("forecast_absolute_error_ug_m3", "mean"),
            persistence_mae_ug_m3=("persistence_absolute_error_ug_m3", "mean"),
        )
        .reset_index()
    )
    forecast_mae = float(stations.forecast_mae_ug_m3.mean())
    persistence_mae = float(stations.persistence_mae_ug_m3.mean())
    stations["station_skill_pct"] = 100.0 * (
        1.0 - stations.forecast_mae_ug_m3 / stations.persistence_mae_ug_m3
    )
    return {
        "n": len(paired),
        "excluded_noncommon_rows": int(len(group) - len(paired)),
        "stations": int(stations.station_code.nunique()),
        "minimum_rows_per_station": int(stations.n.min()),
        "station_balanced_forecast_mae_ug_m3": forecast_mae,
        "station_balanced_persistence_mae_ug_m3": persistence_mae,
        "skill_from_station_balanced_mae_pct": (
            100.0 * (1.0 - forecast_mae / persistence_mae)
            if persistence_mae > 0
            else np.nan
        ),
        "mean_per_station_skill_pct": float(stations.station_skill_pct.mean()),
        "stations_harmed": int(stations.station_skill_pct.lt(0).sum()),
        "interval_coverage_pct": 100.0 * float(paired.interval_covered.mean()),
    }


def _week_block_improvement_interval(
    group: pd.DataFrame, replicates: int = 1000, seed: int = 20260907
) -> tuple[float, float]:
    if group.empty:
        return np.nan, np.nan
    working = group.copy()
    working["calendar_week"] = working.target_time_utc.dt.strftime("%G-W%V")
    weeks = working.calendar_week.unique()
    if len(weeks) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    improvements: list[float] = []
    by_week = {week: working.loc[working.calendar_week.eq(week)] for week in weeks}
    for _ in range(replicates):
        sampled = rng.choice(weeks, size=len(weeks), replace=True)
        draw = pd.concat([by_week[week] for week in sampled], ignore_index=True)
        metrics = _station_balanced_metrics(draw)
        improvements.append(
            metrics["station_balanced_persistence_mae_ug_m3"]
            - metrics["station_balanced_forecast_mae_ug_m3"]
        )
    return tuple(np.quantile(improvements, [0.025, 0.975]).tolist())


def _add_shadow_evidence_fields(
    forecasts: pd.DataFrame,
    completion_utc: pd.Timestamp,
    timing_quality: str,
    observation_fresh_max_hours: float = 6.0,
) -> pd.DataFrame:
    result = forecasts.copy()
    result["generation_completed_utc_for_evaluation"] = completion_utc
    result["timing_evidence_quality"] = timing_quality
    result["lead_remaining_at_generation_hours"] = (
        result.target_time_utc - completion_utc
    ) / pd.Timedelta(hours=1)
    result["availability_lag_at_completion_hours"] = (
        completion_utc - result.issue_time_utc
    ) / pd.Timedelta(hours=1)
    observation_reference = (
        pd.to_datetime(result.asof_time_utc, utc=True)
        if "asof_time_utc" in result
        else result.issue_time_utc
    )
    result["observation_age_at_completion_hours"] = (
        pd.to_numeric(result.latest_pm25_age_hours, errors="coerce")
        + (completion_utc - observation_reference) / pd.Timedelta(hours=1)
    )
    result["generation_status_evaluated"] = np.where(
        result.lead_remaining_at_generation_hours.gt(0),
        "prospective_target",
        "target_reached_before_completion",
    )
    observation_fresh = (
        np.isfinite(result.observation_age_at_completion_hours)
        & result.observation_age_at_completion_hours.ge(0)
        & result.observation_age_at_completion_hours.le(
            observation_fresh_max_hours
        )
    )
    result["prospective_evaluation_eligible"] = (
        result.lead_remaining_at_generation_hours.gt(0)
        & observation_fresh
        & np.isfinite(result.forecast_pm25_ug_m3)
    )
    reasons = np.full(len(result), "eligible_for_shadow_evaluation", dtype=object)
    reasons[(~np.isfinite(result.forecast_pm25_ug_m3)).to_numpy()] = (
        "forecast_nonfinite"
    )
    reasons[(~observation_fresh).to_numpy()] = "observation_stale_or_unavailable"
    reasons[result.lead_remaining_at_generation_hours.le(0).to_numpy()] = (
        "target_reached_before_completion"
    )
    uncertain_completion = timing_quality.endswith("lower_bound")
    if uncertain_completion:
        result["prospective_evaluation_eligible"] = False
        reasons[:] = "completion_time_not_bounded_above"
    result["evaluation_eligibility_reason"] = reasons
    result["duty_service_eligible"] = False
    result["public_service_eligible"] = False
    result["service_eligibility_reason"] = np.where(
        result.prospective_evaluation_eligible,
        "prospective_promotion_gate_not_passed",
        result.evaluation_eligibility_reason,
    )
    return result


def evaluate_promotion_gate(
    matched: pd.DataFrame,
    scorecard: pd.DataFrame,
    run_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    operational = config.get("operational", {})
    thresholds = operational.get("promotion_gate", {})
    required = {
        "minimum_days": int(thresholds.get("minimum_days", 60)),
        "minimum_stations_per_lead": int(
            thresholds.get("minimum_stations_per_lead", 20)
        ),
        "minimum_rows_per_station_lead": int(
            thresholds.get("minimum_rows_per_station_lead", 30)
        ),
        "minimum_skill_from_station_balanced_mae_pct": float(
            thresholds.get("minimum_skill_from_station_balanced_mae_pct", 5.0)
        ),
        "maximum_harmed_station_fraction": float(
            thresholds.get("maximum_harmed_station_fraction", 0.25)
        ),
        "interval_coverage_minimum_pct": float(
            thresholds.get("interval_coverage_minimum_pct", 70.0)
        ),
        "interval_coverage_maximum_pct": float(
            thresholds.get("interval_coverage_maximum_pct", 90.0)
        ),
        "maximum_failed_cycle_fraction": float(
            thresholds.get("maximum_failed_cycle_fraction", 0.10)
        ),
    }
    supported_horizons = [
        int(value)
        for value in operational.get(
            "frozen_service_initialization_horizons_hours", [12, 24, 48, 72]
        )
    ]
    cycle_rows: dict[str, list[dict[str, Any]]] = {}
    for row in run_rows:
        if row.get("status") in {"forecast_generated", "run_failed"}:
            cycle_rows.setdefault(str(row.get("run_id", "unknown")), []).append(row)
    failed_cycles = sum(
        not any(
            item.get("status") == "forecast_generated" and not item.get("warnings")
            for item in items
        )
        for items in cycle_rows.values()
    )
    failed_fraction = failed_cycles / len(cycle_rows) if cycle_rows else 1.0

    candidate_cycles: dict[str, list[dict[str, Any]]] = {}
    for row in run_rows:
        candidate_status = row.get("candidate_shadow", {})
        if candidate_status.get("status") in {
            "candidate_forecast_generated",
            "candidate_forecast_already_exists",
            "candidate_forecast_failed",
        }:
            candidate_cycles.setdefault(
                str(row.get("run_id", row.get("issue_time_utc", "unknown"))), []
            ).append(candidate_status)
    candidate_failed_cycles = sum(
        not any(
            item.get("status")
            in {"candidate_forecast_generated", "candidate_forecast_already_exists"}
            for item in items
        )
        for items in candidate_cycles.values()
    )
    candidate_failed_fraction = (
        candidate_failed_cycles / len(candidate_cycles) if candidate_cycles else 1.0
    )
    bundle_results: list[dict[str, Any]] = []
    bundle_ids = sorted(matched.model_bundle_id.dropna().astype(str).unique())
    for bundle in bundle_ids:
        point_roles = sorted(
            set(
                matched.loc[
                    matched.model_bundle_id.astype(str).eq(bundle), "point_model_role"
                ].dropna()
            ).intersection({"primary_point", "candidate_point"})
        )
        if len(point_roles) != 1:
            bundle_results.append(
                {
                    "model_bundle_id": bundle,
                    "status": "fail",
                    "eligible_days": 0,
                    "failure_reasons": ["bundle_has_ambiguous_or_missing_point_model_role"],
                }
            )
            continue
        point_role = point_roles[0]
        bundle_failed_fraction = (
            candidate_failed_fraction
            if point_role == "candidate_point"
            else failed_fraction
        )
        eligible = matched.loc[
            matched.model_bundle_id.astype(str).eq(bundle)
            & matched.prospective_evaluation_eligible
            & matched.point_model_role.eq(point_role)
        ].copy()
        eligible_days = int(eligible.issue_time_utc.dt.floor("D").nunique())
        reasons: list[str] = []
        if eligible_days < required["minimum_days"]:
            reasons.append(
                f"eligible_days={eligible_days} below {required['minimum_days']}"
            )
        if bundle_failed_fraction > required["maximum_failed_cycle_fraction"]:
            reasons.append(
                f"failed_cycle_fraction={bundle_failed_fraction:.3f} exceeds "
                f"{required['maximum_failed_cycle_fraction']:.3f}"
            )
        prospective_rows = scorecard.loc[
            scorecard.model_bundle_id.astype(str).eq(bundle)
            & scorecard.eligibility_scope.eq("prospective_eligible")
            & scorecard.point_model_role.eq(point_role)
        ]
        for horizon in supported_horizons:
            rows = prospective_rows.loc[prospective_rows.forecast_hour.eq(horizon)]
            if rows.empty:
                reasons.append(
                    f"lead_{horizon:03d}h_has_no_eligible_{point_role}_scorecard"
                )
                continue
            row = rows.iloc[0]
            if (
                int(row.stations) < required["minimum_stations_per_lead"]
                or int(row.minimum_rows_per_station)
                < required["minimum_rows_per_station_lead"]
            ):
                reasons.append(
                    f"lead_{horizon:03d}h_has_insufficient_station_support"
                )
            if row.skill_from_station_balanced_mae_pct < required[
                "minimum_skill_from_station_balanced_mae_pct"
            ] or not (row.improvement_ci95_lower_ug_m3 > 0):
                reasons.append(f"lead_{horizon:03d}h_skill_or_uncertainty_failed")
            harmed_fraction = (
                float(row.stations_harmed) / float(row.stations)
                if int(row.stations) > 0
                else np.inf
            )
            if harmed_fraction > required["maximum_harmed_station_fraction"]:
                reasons.append(f"lead_{horizon:03d}h_station_harm_failed")
            if not (
                required["interval_coverage_minimum_pct"]
                <= row.interval_coverage_pct
                <= required["interval_coverage_maximum_pct"]
            ):
                reasons.append(f"lead_{horizon:03d}h_interval_coverage_failed")
        bundle_results.append(
            {
                "model_bundle_id": bundle,
                "point_model_role": point_role,
                "status": "pass" if not reasons else "fail",
                "eligible_days": eligible_days,
                "attempted_cycles": (
                    len(candidate_cycles)
                    if point_role == "candidate_point"
                    else len(cycle_rows)
                ),
                "failed_cycles": (
                    candidate_failed_cycles
                    if point_role == "candidate_point"
                    else failed_cycles
                ),
                "failed_cycle_fraction": bundle_failed_fraction,
                "failure_reasons": reasons,
            }
        )
    if not bundle_results:
        bundle_results.append(
            {
                "model_bundle_id": None,
                "status": "fail",
                "eligible_days": 0,
                "failure_reasons": ["no_model_bundle_with_matched_forecasts"],
            }
        )
    return {
        "status": (
            "pass" if bundle_results and all(row["status"] == "pass" for row in bundle_results) else "fail"
        ),
        "promotion_authorized": False,
        "supported_initialization_horizons_hours": supported_horizons,
        "attempted_cycles_from_structured_log": len(cycle_rows),
        "failed_cycles": failed_cycles,
        "failed_cycle_fraction": failed_fraction,
        "thresholds": required,
        "bundle_results": bundle_results,
        "note": (
            "A numerical pass would only make the system eligible for human review; "
            "promotion is never automatic. Development-period results do not satisfy "
            "this prospective gate."
        ),
    }


def verify_shadow_forecasts(
    paths: ExperimentPaths,
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_paths = sorted((paths.root / "shadow" / "forecasts").glob("*.csv"))
    candidate_paths = sorted(
        (paths.root / "shadow" / "candidate_forecasts").glob("*.csv")
    )
    forecast_paths = [
        *((path, "frozen") for path in primary_paths),
        *((path, "candidate") for path in candidate_paths),
    ]
    if not forecast_paths:
        return pd.DataFrame(), pd.DataFrame()
    run_rows = _read_shadow_runs(paths)
    operational_config = load_operational_config(paths)
    forecast_frames: list[pd.DataFrame] = []
    for path, forecast_family in forecast_paths:
        frame = pd.read_csv(
            path,
            parse_dates=["issue_time_utc", "target_time_utc", "generated_utc"],
            low_memory=False,
        )
        metadata_path = path.with_suffix(".json")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        source_hash = file_sha256(path)
        if metadata.get("output_sha256") not in {None, source_hash}:
            raise ValueError(f"Shadow forecast checksum mismatch: {path.name}")
        source_eligible = frame.get("prospective_evaluation_eligible")
        source_reason = frame.get("evaluation_eligibility_reason")
        if forecast_family == "candidate":
            required_candidate_columns = {
                "candidate_version",
                "interval_version",
                "asof_time_utc",
                "prediction_lower_ug_m3",
                "prediction_upper_ug_m3",
            }
            missing = required_candidate_columns.difference(frame.columns)
            if missing:
                raise ValueError(
                    f"Candidate shadow forecast lacks columns: {sorted(missing)}"
                )
            frame["prediction_q10_ug_m3"] = frame.prediction_lower_ug_m3
            frame["prediction_q50_ug_m3"] = frame.forecast_pm25_ug_m3
            frame["prediction_q90_ug_m3"] = frame.prediction_upper_ug_m3
        completion, timing_quality = _completion_evidence(
            frame, metadata, source_hash, run_rows
        )
        frame = _add_shadow_evidence_fields(
            frame,
            completion,
            timing_quality,
            float(operational_config["observation_fresh_max_hours"]),
        )
        frame["forecast_artifact_sha256"] = source_hash
        frame["forecast_family"] = forecast_family
        if forecast_family == "candidate":
            candidate_versions = frame.candidate_version.dropna().astype(str).unique()
            interval_versions = frame.interval_version.dropna().astype(str).unique()
            if len(candidate_versions) != 1 or len(interval_versions) != 1:
                raise ValueError("Candidate forecast mixes model or interval versions")
            frame["model_bundle_id"] = (
                f"candidate::{candidate_versions[0]}::interval::{interval_versions[0]}"
            )
            frame["point_model_role"] = "candidate_point"
            if source_eligible is not None:
                source_eligible = source_eligible.astype(bool)
                computed_eligible = frame.prospective_evaluation_eligible.copy()
                frame["prospective_evaluation_eligible"] = (
                    computed_eligible & source_eligible
                )
                if source_reason is not None:
                    frame["evaluation_eligibility_reason"] = np.where(
                        computed_eligible & ~source_eligible,
                        source_reason,
                        frame.evaluation_eligibility_reason,
                    )
        elif "model_bundle_id" not in frame:
            frame["model_bundle_id"] = metadata.get(
                "model_bundle_id",
                metadata.get("deployment_manifest_sha256", "legacy_unknown_bundle"),
            )
        if "point_model_role" not in frame:
            frame["point_model_role"] = np.where(
                frame.forecast_status.str.startswith("primary"),
                "primary_point",
                "observation_only_fallback",
            )
        forecast_frames.append(frame)
    forecasts = pd.concat(forecast_frames, ignore_index=True)
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
    scopes = {
        "all_research_records": pd.Series(True, index=matched.index),
        "prospective_eligible": matched.prospective_evaluation_eligible,
    }
    for scope_name, scope_mask in scopes.items():
        scoped = matched.loc[scope_mask].copy()
        for (bundle, role, horizon), group in scoped.groupby(
            ["model_bundle_id", "point_model_role", "forecast_hour"], observed=True
        ):
            values = _station_balanced_metrics(group)
            lower, upper = _week_block_improvement_interval(group)
            values.update(
                {
                    "model_bundle_id": bundle,
                    "point_model_role": role,
                    "forecast_hour": int(horizon),
                    "eligibility_scope": scope_name,
                    "issue_days": int(group.issue_time_utc.dt.floor("D").nunique()),
                    "completion_timing_quality": ";".join(
                        sorted(group.timing_evidence_quality.unique())
                    ),
                    "improvement_ci95_lower_ug_m3": lower,
                    "improvement_ci95_upper_ug_m3": upper,
                    "forecast_bias_ug_m3": float(group.forecast_error_ug_m3.mean()),
                }
            )
            rows.append(values)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["model_bundle_id", "point_model_role", "eligibility_scope", "forecast_hour"]
        )
    _atomic_csv(summary, verification_dir / "scorecard_by_lead.csv")
    gate_config = load_config(paths.config)
    gate_config["operational"] = operational_config
    gate = evaluate_promotion_gate(matched, summary, run_rows, gate_config)
    write_json(verification_dir / "promotion_gate.json", gate)
    return matched, summary


def _candidate_cams_is_complete(
    cams: pd.DataFrame | None,
    issue_time: pd.Timestamp,
    station_codes: set[str],
    forecast_hours: set[int],
) -> bool:
    if cams is None or cams.empty:
        return False
    required = {
        "station_code",
        "issue_time_utc",
        "valid_time_utc",
        "forecast_hour",
        "cams_pm25_ug_m3",
    }
    if not required.issubset(cams.columns):
        return False
    issue_values = pd.to_datetime(cams.issue_time_utc, utc=True)
    subset = cams.loc[
        issue_values.eq(issue_time)
        & pd.to_numeric(cams.forecast_hour, errors="coerce").isin(forecast_hours)
    ].copy()
    if len(subset) != len(station_codes) * len(forecast_hours):
        return False
    if subset.duplicated(["station_code", "issue_time_utc", "forecast_hour"]).any():
        return False
    if set(subset.station_code.astype(str)) != station_codes:
        return False
    values = pd.to_numeric(subset.cams_pm25_ug_m3, errors="coerce")
    return bool(np.isfinite(values).all() and values.ge(0).all())


def _run_candidate_shadow(
    paths: ExperimentPaths,
    issue_time: pd.Timestamp,
    observations: pd.DataFrame,
    observation_manifest: dict[str, Any],
    cams: pd.DataFrame | None = None,
    allow_cams_acquisition: bool = True,
) -> dict[str, Any]:
    """Run or resume one independently versioned candidate shadow forecast."""

    now = pd.Timestamp.now(tz="UTC")
    candidate_issue = issue_time + pd.Timedelta(hours=10)
    if now < candidate_issue:
        return {
            "status": "candidate_not_yet_due",
            "candidate_asof_time_utc": candidate_issue.isoformat(),
            "retryable": True,
        }
    output = (
        paths.root
        / "shadow"
        / "candidate_forecasts"
        / f"pm25_candidate_shadow_{issue_time:%Y%m%dT%H%MZ}.csv"
    )
    if output.exists():
        metadata_path = output.with_suffix(".json")
        if not metadata_path.exists():
            raise ValueError("Existing candidate output lacks its integrity metadata")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual_hash = file_sha256(output)
        if metadata.get("output_sha256") != actual_hash:
            raise ValueError("Existing candidate output checksum mismatch")
        return {
            "status": "candidate_forecast_already_exists",
            "retryable": False,
            **metadata,
        }

    from .candidate import run_candidate_forecast

    candidate_config = json.loads(
        (paths.root / "candidate_config.json").read_text(encoding="utf-8")
    )
    horizons = {int(value) for value in candidate_config["lead_mapping_hours"]}
    registry = pd.read_csv(paths.station_metadata)
    station_codes = set(registry.station_code.astype(str))
    acquisition_attempts = 0
    cams_error_class: str | None = None
    if not _candidate_cams_is_complete(
        cams, issue_time, station_codes, horizons
    ):
        if not allow_cams_acquisition:
            raise ValueError(
                "Candidate CAMS acquisition suppressed after a permanent same-run "
                "primary acquisition failure"
            )
        for attempt in range(1, 3):
            acquisition_attempts = attempt
            try:
                cams, _ = acquire_direct_cams(paths, issue_time)
                if not _candidate_cams_is_complete(
                    cams, issue_time, station_codes, horizons
                ):
                    raise ValueError("Candidate requires complete all-station CAMS input")
                break
            except Exception as error:
                cams_error_class = _cams_error_class(error)
                if cams_error_class != "transient" or attempt >= 2:
                    raise
                time.sleep(5)
    candidate_feature_config = load_config(paths.config)
    candidate_feature_config["forecast_cycle_hours_utc"] = [10]
    candidate_feature_config["forecast_hours"] = sorted(
        int(value) for value in candidate_config["lead_mapping_hours"].values()
    )
    candidate_features, feature_manifest = build_shadow_issue_features(
        paths,
        observations,
        candidate_issue,
        availability_cutoff_utc=pd.Timestamp(observation_manifest["retrieved_utc"]),
        feature_config=candidate_feature_config,
    )
    _, metadata = run_candidate_forecast(
        candidate_issue,
        candidate_features,
        cams if cams is not None else pd.DataFrame(),
        output,
        paths,
    )
    return {
        "status": "candidate_forecast_generated",
        "retryable": False,
        "cams_acquisition_attempts": acquisition_attempts,
        "cams_error_class": cams_error_class,
        "feature_manifest": feature_manifest,
        **metadata,
    }


def _attempt_candidate_shadow(
    paths: ExperimentPaths,
    issue_time: pd.Timestamp,
    observations: pd.DataFrame,
    observation_manifest: dict[str, Any],
    cams: pd.DataFrame | None = None,
    allow_cams_acquisition: bool = True,
) -> dict[str, Any]:
    try:
        return _run_candidate_shadow(
            paths,
            issue_time,
            observations,
            observation_manifest,
            cams,
            allow_cams_acquisition,
        )
    except Exception as error:
        recorded = pd.Timestamp.now(tz="UTC")
        status = {
            "status": "candidate_forecast_failed",
            "recorded_utc": recorded.isoformat(),
            "issue_time_utc": issue_time.isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error)[:2000],
            "retryable": True,
        }
        write_json(
            paths.root
            / "shadow"
            / "candidate_state"
            / f"failed_{issue_time:%Y%m%dT%H%MZ}_{recorded:%Y%m%dT%H%M%S%fZ}.json",
            status,
        )
        return status


def run_daily_shadow(
    issue_date: str | None = None,
    paths: ExperimentPaths | None = None,
) -> dict[str, Any]:
    paths = paths or ExperimentPaths()
    started = time.perf_counter()
    run_started_utc = pd.Timestamp.now(tz="UTC")
    issue_time = (
        pd.Timestamp(issue_date, tz="UTC")
        if issue_date is not None
        else run_started_utc.floor("D")
    )
    if issue_time.hour != 0:
        raise ValueError("Shadow workflow currently supports the validated 00 UTC cycle only")
    shadow = paths.root / "shadow"
    (shadow / "logs").mkdir(parents=True, exist_ok=True)
    observations, observation_manifest = acquire_dashboard_observations(paths)
    forecast_path = shadow / "forecasts" / f"pm25_shadow_{issue_time:%Y%m%dT%H%MZ}.csv"
    warnings: list[str] = []
    cams_for_candidate: pd.DataFrame | None = None
    allow_candidate_cams_acquisition = True
    if forecast_path.exists():
        forecast = pd.read_csv(
            forecast_path,
            parse_dates=["issue_time_utc", "target_time_utc", "generated_utc"],
            low_memory=False,
        )
        forecast_metadata = json.loads(
            forecast_path.with_suffix(".json").read_text(encoding="utf-8")
        )
        warnings = list(forecast_metadata.get("warnings", []))
        status = "forecast_already_exists"
    else:
        features, feature_manifest = build_shadow_issue_features(
            paths,
            observations,
            issue_time,
            availability_cutoff_utc=pd.Timestamp(
                observation_manifest["retrieved_utc"]
            ),
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
        cams_error_class: str | None = None
        acquisition_attempts = 0
        for attempt in range(1, 3):
            acquisition_attempts = attempt
            try:
                cams, cams_manifest = acquire_direct_cams(paths, issue_time)
                cams_error = None
                break
            except Exception as error:  # preserve a degraded forecast on source outage
                cams_error = f"{type(error).__name__}: {error}"
                cams_error_class = _cams_error_class(error)
                if cams_error_class != "transient" or attempt >= 2:
                    break
                time.sleep(5)
        if cams_error:
            warnings.append(
                "CAMS unavailable; "
                f"class={cams_error_class}; attempts={acquisition_attempts}: {cams_error}"
            )
            if cams_error_class == "permanent_request_or_cycle_unavailable":
                allow_candidate_cams_acquisition = False
        forecast, metadata = run_operational_forecast(
            issue_time.isoformat(),
            output_path=forecast_path,
            paths=paths,
            issue_features_frame=features,
            cams_frame=cams,
        )
        generation_completed_utc = pd.Timestamp.now(tz="UTC")
        forecast["run_started_utc"] = run_started_utc
        forecast["generated_utc"] = generation_completed_utc
        forecast["generation_completed_utc"] = generation_completed_utc
        forecast["availability_lag_hours"] = (
            generation_completed_utc - forecast.issue_time_utc
        ) / pd.Timedelta(hours=1)
        forecast["observation_age_at_generation_hours"] = (
            forecast.latest_pm25_age_hours + forecast.availability_lag_hours
        )
        forecast = _add_shadow_evidence_fields(
            forecast,
            generation_completed_utc,
            "exact_completion_timestamp",
        )
        forecast["shadow_input_status"] = np.where(
            ~np.isfinite(forecast.observation_age_at_generation_hours)
            | forecast.observation_age_at_generation_hours.gt(6),
            "observation_stale_at_generation",
            "observation_fresh_at_generation",
        )
        _atomic_csv(forecast, forecast_path)
        cams_for_candidate = cams
        metadata.update(
            {
                "run_started_utc": run_started_utc.isoformat(),
                "generated_utc": generation_completed_utc.isoformat(),
                "generation_completed_utc": generation_completed_utc.isoformat(),
                "generation_timestamp_semantics": "forecast_write_completion",
                "output_sha256": file_sha256(forecast_path),
                "temporally_prospective_rows": int(
                    forecast.generation_status_evaluated.eq("prospective_target").sum()
                ),
                "prospective_evaluation_eligible_rows": int(
                    forecast.prospective_evaluation_eligible.sum()
                ),
                "late_rows": int(
                    forecast.generation_status_evaluated.ne("prospective_target").sum()
                ),
                "stale_at_generation_rows": int(
                    forecast.shadow_input_status.eq(
                        "observation_stale_at_generation"
                    ).sum()
                ),
                "observation_manifest": observation_manifest,
                "feature_manifest": feature_manifest,
                "cams_manifest": cams_manifest,
                "cams_error_class": cams_error_class,
                "cams_acquisition_attempts": acquisition_attempts,
                "warnings": warnings,
            }
        )
        write_json(forecast_path.with_suffix(".json"), metadata)
        status = "forecast_generated"
    candidate_status = _attempt_candidate_shadow(
        paths,
        issue_time,
        observations,
        observation_manifest,
        cams_for_candidate,
        allow_candidate_cams_acquisition,
    )
    matched, scorecard = verify_shadow_forecasts(paths, observations)
    run = {
        "run_id": f"shadow-{issue_time:%Y%m%dT%H%MZ}",
        "status": status,
        "run_started_utc": run_started_utc.isoformat(),
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "issue_time_utc": issue_time.isoformat(),
        "forecast_rows": len(forecast),
        "forecast_sha256": file_sha256(forecast_path),
        "matched_verification_rows": len(matched),
        "scorecard_leads": len(scorecard),
        "warnings": warnings,
        "candidate_shadow": candidate_status,
        "candidate_warnings": (
            [
                "Candidate shadow failed without affecting the frozen output: "
                f"{candidate_status.get('error_type')}: "
                f"{candidate_status.get('error_message')}"
            ]
            if candidate_status.get("status") == "candidate_forecast_failed"
            else []
        ),
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
