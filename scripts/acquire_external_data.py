#!/usr/bin/env python3
"""Acquire bounded external inputs for the PM2.5 station forecast experiment.

The script is restartable and records every request, checksum, and source URL.
Credentials are read from the AQ root .env file but are never printed or copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cdsapi
import ee
import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
AQ_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = EXPERIMENT_ROOT / "config.json"
STATION_METADATA_PATH = EXPERIMENT_ROOT / "station_metadata.csv"
EXTERNAL_ROOT = EXPERIMENT_ROOT / "data" / "external"
BMKG_MARKER_URL = "https://awscenter.bmkg.go.id/base/marker_login_map"
CAMS_DATASET_URL = (
    "https://ads.atmosphere.copernicus.eu/datasets/"
    "cams-global-atmospheric-composition-forecasts?tab=overview"
)
CAMS_EARTH_ENGINE_URL = (
    "https://developers.google.com/earth-engine/datasets/catalog/ECMWF_CAMS_NRT"
)
CAMS_EARTH_ENGINE_COLLECTION = "ECMWF/CAMS/NRT"
CAMS_EARTH_ENGINE_BAND = "particulate_matter_d_less_than_25_um_surface"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_local_environment() -> None:
    env_path = AQ_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def acquire_bmkg_station_markers(refresh: bool) -> Path:
    output_dir = EXTERNAL_ROOT / "bmkg_station_markers"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshots = sorted(output_dir.glob("bmkg_station_markers_*.json"))
    if snapshots and not refresh:
        snapshot = snapshots[-1]
    else:
        snapshot = output_dir / f"bmkg_station_markers_{stamp}.json"
        temporary = snapshot.with_suffix(".json.part")
        request = urllib.request.Request(
            BMKG_MARKER_URL,
            headers={"User-Agent": "BMKG-PM25-ML-research/1.0"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, snapshot)

    marker_rows = json.loads(snapshot.read_text(encoding="utf-8"))
    if not isinstance(marker_rows, list) or not marker_rows:
        raise ValueError("BMKG marker snapshot is not a non-empty JSON list")
    by_id = {str(row.get("id_station")): row for row in marker_rows}
    metadata = pd.read_csv(STATION_METADATA_PATH, dtype=str)
    checked = []
    for row in metadata.loc[metadata.coordinate_source.eq("bmkg_station_marker")].itertuples():
        source = by_id.get(str(row.coordinate_source_id))
        if source is None:
            raise ValueError(
                f"BMKG marker {row.coordinate_source_id} for {row.station_code} is absent"
            )
        latitude_difference = abs(float(source["lat"]) - float(row.latitude))
        longitude_difference = abs(float(source["lng"]) - float(row.longitude))
        if max(latitude_difference, longitude_difference) > 0.01:
            raise ValueError(
                f"Coordinate mismatch for {row.station_code}: metadata and BMKG marker "
                f"differ by more than 0.01 degree"
            )
        checked.append(
            {
                "station_code": row.station_code,
                "source_station_id": row.coordinate_source_id,
                "source_station_name": source.get("name_station"),
                "latitude": float(source["lat"]),
                "longitude": float(source["lng"]),
            }
        )

    manifest = {
        "source_name": "BMKG AWS Center station marker service",
        "source_url": BMKG_MARKER_URL,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot": snapshot.name,
        "bytes": snapshot.stat().st_size,
        "sha256": sha256(snapshot),
        "records": len(marker_rows),
        "experiment_coordinates_checked": checked,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "source": "BMKG station markers",
                "snapshot": str(snapshot.relative_to(EXPERIMENT_ROOT)),
                "bytes": snapshot.stat().st_size,
                "checked_coordinates": len(checked),
            }
        )
    )
    return snapshot


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def annual_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    year = start.year
    while year <= end.year:
        left = max(start, date(year, 1, 1))
        right = min(end, date(year, 12, 31))
        chunks.append((left, right))
        year += 1
    return chunks


def quarterly_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    left = start
    while left <= end:
        right = min(left + timedelta(days=91), end)
        chunks.append((left, right))
        left = right + timedelta(days=1)
    return chunks


def _earth_engine_station_features(metadata: pd.DataFrame) -> ee.FeatureCollection:
    return ee.FeatureCollection(
        [
            ee.Feature(
                ee.Geometry.Point([float(row.longitude), float(row.latitude)]),
                {
                    "station_code": str(row.station_code),
                    "sample_latitude": float(row.latitude),
                    "sample_longitude": float(row.longitude),
                },
            )
            for row in metadata.itertuples()
        ]
    )


def _extract_earth_engine_chunk(
    stations: ee.FeatureCollection,
    start: date,
    end: date,
    forecast_hours: list[int],
) -> pd.DataFrame:
    valid_end = end + timedelta(days=4)
    collection = (
        ee.ImageCollection(CAMS_EARTH_ENGINE_COLLECTION)
        .filterDate(start.isoformat(), valid_end.isoformat())
        .filter(ee.Filter.eq("model_initialization_hour", 0))
        .filter(ee.Filter.inList("model_forecast_hour", forecast_hours))
    )

    def sample(image: ee.Image) -> ee.FeatureCollection:
        image = ee.Image(image)
        properties = {
            "issue_time_text": image.get("model_initialization_datetime"),
            "valid_time_ms": image.get("system:time_start"),
            "forecast_hour": image.get("model_forecast_hour"),
            "source_image_id": image.id(),
        }
        sampled = (
            image.select(CAMS_EARTH_ENGINE_BAND)
            .resample("bilinear")
            .sampleRegions(
                collection=stations,
                scale=1000,
                geometries=False,
                tileScale=4,
            )
        )
        return sampled.map(lambda feature: ee.Feature(feature).set(properties))

    features = ee.FeatureCollection(collection.map(sample).flatten())
    frame = ee.data.computeFeatures(
        {"expression": features, "fileFormat": "PANDAS_DATAFRAME"}
    )
    frame = frame.rename(
        columns={CAMS_EARTH_ENGINE_BAND: "cams_pm25_kg_m3"}
    )
    required = {
        "station_code",
        "issue_time_text",
        "valid_time_ms",
        "forecast_hour",
        "cams_pm25_kg_m3",
        "source_image_id",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Earth Engine CAMS output lacks columns: {sorted(missing)}")
    frame["issue_time_utc"] = pd.to_datetime(
        frame.issue_time_text, utc=True, errors="coerce"
    )
    frame["valid_time_utc"] = pd.to_datetime(
        frame.valid_time_ms, unit="ms", utc=True, errors="coerce"
    )
    frame["forecast_hour"] = pd.to_numeric(
        frame.forecast_hour, errors="raise"
    ).astype(int)
    frame["cams_pm25_kg_m3"] = pd.to_numeric(
        frame.cams_pm25_kg_m3, errors="coerce"
    )
    start_time = pd.Timestamp(start, tz="UTC")
    end_time = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    frame = frame.loc[frame.issue_time_utc.between(start_time, end_time)].copy()
    frame["cams_pm25_ug_m3"] = frame.cams_pm25_kg_m3 * 1.0e9
    expected_valid = frame.issue_time_utc + pd.to_timedelta(
        frame.forecast_hour, unit="h"
    )
    if not expected_valid.eq(frame.valid_time_utc).all():
        raise ValueError("Earth Engine CAMS issue, lead, and valid time are inconsistent")
    if frame.cams_pm25_ug_m3.isna().any() or frame.cams_pm25_ug_m3.lt(0).any():
        raise ValueError("Earth Engine CAMS contains missing or negative PM2.5 samples")
    keep = [
        "station_code",
        "sample_latitude",
        "sample_longitude",
        "issue_time_utc",
        "valid_time_utc",
        "forecast_hour",
        "cams_pm25_kg_m3",
        "cams_pm25_ug_m3",
        "source_image_id",
    ]
    return frame[keep].sort_values(
        ["issue_time_utc", "station_code", "forecast_hour"]
    )


def acquire_cams_earth_engine(
    start: date,
    end: date,
    refresh: bool,
    project: str,
) -> Path:
    """Extract forecast-valid station samples from the official CAMS GEE mirror."""

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    forecast_hours = [int(value) for value in config["forecast_hours"]]
    if any(hour % 3 for hour in forecast_hours):
        raise ValueError(
            "ECMWF/CAMS/NRT in Earth Engine is three-hourly; every lead must be divisible by 3"
        )
    metadata = pd.read_csv(STATION_METADATA_PATH)
    ee.Initialize(project=project)
    stations = _earth_engine_station_features(metadata)
    output_dir = EXTERNAL_ROOT / "cams_earth_engine"
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_manifests = []
    frames = []
    for chunk_start, chunk_end in quarterly_chunks(start, end):
        stem = f"cams_station_samples_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}"
        output = chunk_dir / f"{stem}.csv.gz"
        started = time.perf_counter()
        if refresh or not output.exists():
            frame = _extract_earth_engine_chunk(
                stations, chunk_start, chunk_end, forecast_hours
            )
            temporary = output.with_suffix(".csv.gz.part")
            frame.to_csv(temporary, index=False, compression="gzip")
            os.replace(temporary, output)
            downloaded = True
        else:
            frame = pd.read_csv(
                output,
                parse_dates=["issue_time_utc", "valid_time_utc"],
                low_memory=False,
            )
            downloaded = False
        expected_maximum = (
            (chunk_end - chunk_start).days + 1
        ) * len(metadata) * len(forecast_hours)
        manifest = {
            "period_start": chunk_start.isoformat(),
            "period_end": chunk_end.isoformat(),
            "rows": len(frame),
            "maximum_expected_rows": expected_maximum,
            "coverage_pct": 100.0 * len(frame) / expected_maximum,
            "downloaded_this_invocation": downloaded,
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "path": str(output.relative_to(output_dir)),
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
        }
        write_json(chunk_dir / f"manifest_{stem}.json", manifest)
        chunk_manifests.append(manifest)
        frames.append(frame)
        print(json.dumps({"source": "CAMS Earth Engine", **manifest}), flush=True)

    combined = pd.concat(frames, ignore_index=True)
    key = ["station_code", "issue_time_utc", "forecast_hour"]
    duplicates = combined.duplicated(key, keep=False)
    if duplicates.any():
        conflicts = (
            combined.loc[duplicates]
            .groupby(key)["cams_pm25_ug_m3"]
            .nunique(dropna=False)
            .gt(1)
        )
        if conflicts.any():
            raise ValueError(
                f"Earth Engine CAMS contains {int(conflicts.sum())} conflicting keys"
            )
        combined = combined.drop_duplicates(key, keep="last")
    combined = combined.sort_values(key).reset_index(drop=True)
    expected = ((end - start).days + 1) * len(metadata) * len(forecast_hours)
    coverage_pct = 100.0 * len(combined) / expected
    if combined.station_code.nunique() != len(metadata):
        raise ValueError("Earth Engine CAMS extraction does not cover every station")
    if set(combined.forecast_hour.unique()) != set(forecast_hours):
        raise ValueError("Earth Engine CAMS extraction does not cover every forecast lead")
    if coverage_pct < 98.0:
        raise ValueError(
            f"Earth Engine CAMS extraction coverage is only {coverage_pct:.2f}%"
        )
    expected_index = pd.MultiIndex.from_product(
        [
            metadata.station_code.astype(str).sort_values(),
            pd.date_range(start=start, end=end, freq="D", tz="UTC"),
            sorted(forecast_hours),
        ],
        names=key,
    )
    observed_index = pd.MultiIndex.from_frame(combined[key])
    missing = expected_index.difference(observed_index).to_frame(index=False)
    missing_path = output_dir / "missing_station_issue_leads.csv"
    temporary_missing = missing_path.with_suffix(".csv.part")
    missing.to_csv(temporary_missing, index=False)
    os.replace(temporary_missing, missing_path)
    missing_issue_summary = [
        {
            "issue_time_utc": pd.Timestamp(issue_time).isoformat(),
            "missing_station_leads": int(count),
        }
        for issue_time, count in missing.groupby("issue_time_utc").size().items()
    ]
    output = output_dir / f"cams_station_forecasts_{start:%Y%m%d}_{end:%Y%m%d}.csv.gz"
    temporary = output.with_suffix(".csv.gz.part")
    combined.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, output)
    manifest = {
        "source_name": "CAMS Global Near-Real-Time official Earth Engine mirror",
        "source_url": CAMS_EARTH_ENGINE_URL,
        "earth_engine_collection": CAMS_EARTH_ENGINE_COLLECTION,
        "source_band": CAMS_EARTH_ENGINE_BAND,
        "source_units": "kg m-3",
        "retrieved_or_verified_utc": datetime.now(timezone.utc).isoformat(),
        "issue_cycle_utc": 0,
        "forecast_hours": forecast_hours,
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "sampling": "bilinear resampling at station coordinates",
        "conversion": "kg m-3 multiplied by 1e9 to micrograms m-3",
        "rows": len(combined),
        "maximum_expected_rows": expected,
        "coverage_pct": coverage_pct,
        "missing_station_issue_leads": len(missing),
        "missing_issue_times": missing_issue_summary,
        "missing_inventory": {
            "path": missing_path.name,
            "bytes": missing_path.stat().st_size,
            "sha256": sha256(missing_path),
        },
        "stations": int(combined.station_code.nunique()),
        "output": output.name,
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256(output),
        "chunks": chunk_manifests,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"source": "CAMS Earth Engine combined", **manifest}), flush=True)
    return output


def validate_cams_archive(archive: Path) -> list[str]:
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"CAMS output is not a ZIP archive: {archive}")
    with zipfile.ZipFile(archive) as bundle:
        bad_member = bundle.testzip()
        if bad_member is not None:
            raise ValueError(f"CAMS archive has a corrupt member: {bad_member}")
        members = [name for name in bundle.namelist() if name.lower().endswith(".nc")]
    if not members:
        raise ValueError(f"CAMS archive contains no NetCDF member: {archive}")
    return members


def extract_cams_archive(archive: Path, members: list[str]) -> list[Path]:
    netcdf_dir = archive.parent / "netcdf"
    netcdf_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    with zipfile.ZipFile(archive) as bundle:
        for index, member in enumerate(members, start=1):
            suffix = "" if len(members) == 1 else f"_{index:02d}"
            output = netcdf_dir / f"{archive.stem}{suffix}.nc"
            if not output.exists():
                temporary = output.with_suffix(".nc.part")
                with bundle.open(member) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
                os.replace(temporary, output)
            outputs.append(output)
    return outputs


def _cams_request(
    config: dict[str, Any], cams: dict[str, Any], start: date, end: date, area: list[float]
) -> dict[str, Any]:
    return {
        "variable": cams["variables"],
        "date": f"{start:%Y-%m-%d}/{end:%Y-%m-%d}",
        "time": [f"{hour:02d}:00" for hour in config["forecast_cycle_hours_utc"]],
        "leadtime_hour": [str(hour) for hour in config["forecast_hours"]],
        "type": "forecast",
        "area": area,
        "data_format": cams["data_format"],
    }


def _acquire_cams_request(
    client: cdsapi.Client,
    cams: dict[str, Any],
    request: dict[str, Any],
    archive: Path,
    refresh: bool,
    request_label: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    downloaded = False
    if refresh or not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        if temporary.exists():
            temporary.unlink()
        client.retrieve(cams["dataset"], request, str(temporary))
        validate_cams_archive(temporary)
        os.replace(temporary, archive)
        downloaded = True
    members = validate_cams_archive(archive)
    netcdf_paths = extract_cams_archive(archive, members)
    manifest = {
        "source_name": "CAMS global atmospheric composition forecasts",
        "source_url": CAMS_DATASET_URL,
        "dataset": cams["dataset"],
        "request_label": request_label,
        "request": request,
        "downloaded_this_invocation": downloaded,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "retrieved_or_verified_utc": datetime.now(timezone.utc).isoformat(),
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "netcdf_members": [
            {
                "archive_member": member,
                "path": str(path.relative_to(archive.parent)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for member, path in zip(members, netcdf_paths, strict=True)
        ],
    }
    write_json(archive.parent / f"manifest_{archive.stem}.json", manifest)
    completed = {
        "request_label": request_label,
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "netcdf_files": len(netcdf_paths),
    }
    print(json.dumps(completed))
    return completed


def acquire_cams(
    start: date,
    end: date,
    refresh: bool,
    station_code: str | None = None,
) -> None:
    load_local_environment()
    key = os.environ.get("AQ_ADS_KEY")
    if not key:
        raise RuntimeError("AQ_ADS_KEY is required but is not configured")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cams = config["cams"]
    output_dir = EXTERNAL_ROOT / "cams_global_forecasts"
    output_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(
        url=os.environ.get("AQ_ADS_URL", "https://ads.atmosphere.copernicus.eu/api"),
        key=key,
        quiet=False,
    )
    completed = []
    layout = cams.get("request_layout", "annual_domain")
    if layout in {"station_boxes", "earth_engine_station_samples"}:
        metadata = pd.read_csv(STATION_METADATA_PATH)
        if station_code is not None:
            metadata = metadata.loc[metadata.station_code.eq(station_code)].copy()
            if metadata.empty:
                raise ValueError(f"Unknown station code requested: {station_code}")
        buffer_degrees = float(cams["station_box_buffer_degrees"])
        for row in metadata.itertuples():
            area = [
                float(row.latitude) + buffer_degrees,
                float(row.longitude) - buffer_degrees,
                float(row.latitude) - buffer_degrees,
                float(row.longitude) + buffer_degrees,
            ]
            tag = f"{row.station_code}_{start:%Y%m%d}_{end:%Y%m%d}"
            archive = output_dir / f"cams_{tag}.zip"
            request = _cams_request(config, cams, start, end, area)
            completed.append(
                _acquire_cams_request(
                    client, cams, request, archive, refresh, str(row.station_code)
                )
            )
    elif layout == "annual_domain":
        if station_code is not None:
            raise ValueError("--station-code requires CAMS request_layout=station_boxes")
        for chunk_start, chunk_end in annual_chunks(start, end):
            tag = f"{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}"
            archive = output_dir / f"cams_station_domain_{tag}.zip"
            request = _cams_request(
                config,
                cams,
                chunk_start,
                chunk_end,
                cams["area_north_west_south_east"],
            )
            completed.append(
                _acquire_cams_request(
                    client, cams, request, archive, refresh, f"station_domain_{tag}"
                )
            )
    else:
        raise ValueError(f"Unsupported CAMS request layout: {layout}")
    all_manifests = []
    for manifest_path in sorted(output_dir.glob("manifest_cams_*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        all_manifests.append(
            {
                "request_label": manifest["request_label"],
                "archive": manifest["archive"],
                "archive_bytes": manifest["archive_bytes"],
                "archive_sha256": manifest["archive_sha256"],
                "manifest": manifest_path.name,
            }
        )
    write_json(
        output_dir / "manifest_index.json",
        {
            "source_url": CAMS_DATASET_URL,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "request_layout": layout,
            "completed_this_invocation": completed,
            "available_requests": all_manifests,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=("all", "bmkg", "cams", "cams-earth-engine"),
        default="all",
        help="External source to acquire",
    )
    parser.add_argument("--start", help="Inclusive CAMS start date")
    parser.add_argument("--end", help="Inclusive CAMS end date")
    parser.add_argument(
        "--station-code",
        help="Acquire one direct-archive station box; applies only to --source cams",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--earth-engine-project",
        default="ee-alberthnahas",
        help="Earth Engine project used only for CAMS mirror extraction",
    )
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.source in {"all", "bmkg"}:
        acquire_bmkg_station_markers(refresh=args.refresh)
    if args.source in {"all", "cams-earth-engine"}:
        cams = config["cams"]
        start = parse_date(args.start or cams["start_date"])
        end = parse_date(args.end or cams["end_date"])
        if end < start:
            raise ValueError("CAMS end date precedes start date")
        acquire_cams_earth_engine(
            start=start,
            end=end,
            refresh=args.refresh,
            project=args.earth_engine_project,
        )
    if args.source == "cams":
        cams = config["cams"]
        start = parse_date(args.start or cams["start_date"])
        end = parse_date(args.end or cams["end_date"])
        if end < start:
            raise ValueError("CAMS end date precedes start date")
        acquire_cams(
            start=start,
            end=end,
            refresh=args.refresh,
            station_code=args.station_code,
        )


if __name__ == "__main__":
    main()
