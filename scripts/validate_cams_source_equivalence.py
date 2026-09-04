#!/usr/bin/env python3
"""Compare the Earth Engine CAMS mirror with a direct ADS NetCDF sample.

This is a source-equivalence check, not an accuracy validation against surface
observations. Both sources represent the same CAMS forecast field.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import ee


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    ROOT
    / "data"
    / "external"
    / "cams_source_validation"
    / "cams_pm25_direct_ads_20260101.zip"
)
DEFAULT_EARTH_ENGINE = (
    ROOT
    / "data"
    / "external"
    / "cams_earth_engine"
    / "cams_station_forecasts_20230101_20260831.csv.gz"
)
SOURCE_URL = (
    "https://ads.atmosphere.copernicus.eu/datasets/"
    "cams-global-atmospheric-composition-forecasts?tab=overview"
)
EARTH_ENGINE_COLLECTION = "ECMWF/CAMS/NRT"
EARTH_ENGINE_BAND = "particulate_matter_d_less_than_25_um_surface"


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _native_grid_comparison(
    requests: list[dict], project: str
) -> pd.DataFrame:
    """Verify identity at CAMS native pixel centres before station resampling."""

    ee.Initialize(project=project)
    rows: list[dict] = []
    request_frame = pd.DataFrame(requests)
    for (issue_time_text, forecast_hour), group in request_frame.groupby(
        ["issue_time_utc", "forecast_hour"]
    ):
        issue_time = pd.Timestamp(issue_time_text)
        image_id = (
            f"{EARTH_ENGINE_COLLECTION}/"
            f"{issue_time:%Y%m%dT%H}F{int(forecast_hour):03d}"
        )
        image = ee.Image(image_id).select(EARTH_ENGINE_BAND)
        features = ee.FeatureCollection(
            [
                ee.Feature(
                    ee.Geometry.Point(
                        [float(row.grid_longitude), float(row.grid_latitude)]
                    ),
                    {"station_code": str(row.station_code)},
                )
                for row in group.itertuples()
            ]
        )
        sampled = image.sampleRegions(
            collection=features,
            projection=image.projection(),
            geometries=False,
        )
        earth_engine = ee.data.computeFeatures(
            {"expression": sampled, "fileFormat": "PANDAS_DATAFRAME"}
        ).set_index("station_code")
        for row in group.itertuples():
            mirror_ug_m3 = float(
                earth_engine.loc[row.station_code, EARTH_ENGINE_BAND]
            ) * 1.0e9
            absolute_difference = abs(mirror_ug_m3 - row.direct_ads_pm25_ug_m3)
            rows.append(
                {
                    "comparison_type": "native_grid",
                    "station_code": row.station_code,
                    "latitude": row.grid_latitude,
                    "longitude": row.grid_longitude,
                    "issue_time_utc": issue_time.isoformat(),
                    "forecast_hour": int(forecast_hour),
                    "direct_ads_pm25_ug_m3": row.direct_ads_pm25_ug_m3,
                    "earth_engine_pm25_ug_m3": mirror_ug_m3,
                    "absolute_difference_ug_m3": absolute_difference,
                    "relative_absolute_difference_pct": 100.0
                    * absolute_difference
                    / max(abs(row.direct_ads_pm25_ug_m3), 1.0e-12),
                    "earth_engine_source_image_id": image_id.rsplit("/", 1)[-1],
                }
            )
    return pd.DataFrame(rows)


def compare(
    archive: Path, earth_engine_path: Path, project: str
) -> tuple[pd.DataFrame, dict]:
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Direct CAMS artifact is not a valid ZIP archive: {archive}")
    metadata = pd.read_csv(ROOT / "station_metadata.csv")
    mirror = pd.read_csv(
        earth_engine_path,
        parse_dates=["issue_time_utc", "valid_time_utc"],
        low_memory=False,
    )
    mirror["issue_time_utc"] = pd.to_datetime(mirror.issue_time_utc, utc=True)

    rows: list[dict] = []
    native_requests: list[dict] = []
    source_details: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="cams-source-check-") as temporary:
        with zipfile.ZipFile(archive) as bundle:
            bad_member = bundle.testzip()
            if bad_member is not None:
                raise ValueError(f"Corrupt member in direct CAMS archive: {bad_member}")
            bundle.extractall(temporary)
        netcdf_paths = sorted(Path(temporary).glob("*.nc"))
        if not netcdf_paths:
            raise ValueError("Direct CAMS archive contains no NetCDF file")
        for netcdf_path in netcdf_paths:
            with xr.open_dataset(netcdf_path) as dataset:
                if "pm2p5" not in dataset:
                    raise KeyError("Direct CAMS NetCDF lacks pm2p5")
                variable = dataset["pm2p5"]
                units = str(variable.attrs.get("units", ""))
                if units.replace(" ", "") not in {"kgm**-3", "kgm-3"}:
                    raise ValueError(f"Unexpected direct CAMS PM2.5 units: {units}")
                latitude_min = float(dataset.latitude.min())
                latitude_max = float(dataset.latitude.max())
                longitude_min = float(dataset.longitude.min())
                longitude_max = float(dataset.longitude.max())
                source_details.append(
                    {
                        "member": netcdf_path.name,
                        "forecast_reference_times": [
                            pd.Timestamp(value).isoformat()
                            for value in dataset.forecast_reference_time.values
                        ],
                        "forecast_hours": [
                            int(pd.Timedelta(value) / pd.Timedelta(hours=1))
                            for value in dataset.forecast_period.values
                        ],
                        "latitude_bounds": [latitude_min, latitude_max],
                        "longitude_bounds": [longitude_min, longitude_max],
                        "units": units,
                    }
                )
                in_domain = metadata.loc[
                    metadata.latitude.between(latitude_min, latitude_max)
                    & metadata.longitude.between(longitude_min, longitude_max)
                ]
                for issue_value in dataset.forecast_reference_time.values:
                    issue_time = pd.Timestamp(issue_value, tz="UTC")
                    mirror_issue = mirror.loc[mirror.issue_time_utc.eq(issue_time)]
                    for period_value in dataset.forecast_period.values:
                        forecast_hour = int(
                            pd.Timedelta(period_value) / pd.Timedelta(hours=1)
                        )
                        mirror_lead = mirror_issue.loc[
                            mirror_issue.forecast_hour.eq(forecast_hour)
                        ].set_index("station_code")
                        if mirror_lead.empty:
                            continue
                        field = variable.sel(
                            forecast_reference_time=issue_value,
                            forecast_period=period_value,
                        )
                        for station in in_domain.itertuples():
                            if station.station_code not in mirror_lead.index:
                                continue
                            direct_kg_m3 = float(
                                field.interp(
                                    latitude=float(station.latitude),
                                    longitude=float(station.longitude),
                                    method="linear",
                                ).item()
                            )
                            direct_ug_m3 = direct_kg_m3 * 1.0e9
                            earth_engine_ug_m3 = float(
                                mirror_lead.loc[
                                    station.station_code, "cams_pm25_ug_m3"
                                ]
                            )
                            absolute_difference = abs(
                                earth_engine_ug_m3 - direct_ug_m3
                            )
                            rows.append(
                                {
                                    "comparison_type": "station_bilinear",
                                    "station_code": station.station_code,
                                    "latitude": float(station.latitude),
                                    "longitude": float(station.longitude),
                                    "issue_time_utc": issue_time.isoformat(),
                                    "forecast_hour": forecast_hour,
                                    "direct_ads_pm25_ug_m3": direct_ug_m3,
                                    "earth_engine_pm25_ug_m3": earth_engine_ug_m3,
                                    "absolute_difference_ug_m3": absolute_difference,
                                    "relative_absolute_difference_pct": 100.0
                                    * absolute_difference
                                    / max(abs(direct_ug_m3), 1.0e-12),
                                    "earth_engine_source_image_id": str(
                                        mirror_lead.loc[
                                            station.station_code, "source_image_id"
                                        ]
                                    ),
                                }
                            )
                            grid_latitude = float(
                                dataset.latitude.sel(
                                    latitude=float(station.latitude), method="nearest"
                                ).item()
                            )
                            grid_longitude = float(
                                dataset.longitude.sel(
                                    longitude=float(station.longitude), method="nearest"
                                ).item()
                            )
                            native_requests.append(
                                {
                                    "station_code": station.station_code,
                                    "issue_time_utc": issue_time.isoformat(),
                                    "forecast_hour": forecast_hour,
                                    "grid_latitude": grid_latitude,
                                    "grid_longitude": grid_longitude,
                                    "direct_ads_pm25_ug_m3": float(
                                        field.sel(
                                            latitude=grid_latitude,
                                            longitude=grid_longitude,
                                            method="nearest",
                                        ).item()
                                    )
                                    * 1.0e9,
                                }
                            )

    station_comparison = pd.DataFrame(rows)
    native_comparison = _native_grid_comparison(native_requests, project)
    comparison = pd.concat(
        [native_comparison, station_comparison], ignore_index=True
    ).sort_values(
        ["comparison_type", "issue_time_utc", "station_code", "forecast_hour"]
    )
    if comparison.empty:
        raise ValueError("No overlapping direct ADS and Earth Engine CAMS samples")
    if comparison.duplicated(
        ["comparison_type", "station_code", "issue_time_utc", "forecast_hour"]
    ).any():
        raise ValueError("Source-equivalence comparison contains duplicate keys")
    native = comparison.loc[comparison.comparison_type.eq("native_grid")]
    station_sample = comparison.loc[
        comparison.comparison_type.eq("station_bilinear")
    ]
    native_maximum = float(native.absolute_difference_ug_m3.max())
    station_maximum = float(station_sample.absolute_difference_ug_m3.max())
    station_mean = float(station_sample.absolute_difference_ug_m3.mean())
    manifest = {
        "purpose": "source equivalence, not surface-observation validation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "direct_source_url": SOURCE_URL,
        "direct_archive": {
            "path": str(archive.relative_to(ROOT)),
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
            "netcdf": source_details,
        },
        "earth_engine_extract": {
            "path": str(earth_engine_path.relative_to(ROOT)),
            "bytes": earth_engine_path.stat().st_size,
            "sha256": sha256(earth_engine_path),
        },
        "comparison_rows": len(comparison),
        "stations": int(comparison.station_code.nunique()),
        "forecast_hours": sorted(comparison.forecast_hour.unique().tolist()),
        "native_grid_comparison": {
            "rows": len(native),
            "maximum_absolute_difference_ug_m3": native_maximum,
            "maximum_relative_absolute_difference_pct": float(
                native.relative_absolute_difference_pct.max()
            ),
        },
        "station_bilinear_comparison": {
            "rows": len(station_sample),
            "maximum_absolute_difference_ug_m3": station_maximum,
            "mean_absolute_difference_ug_m3": station_mean,
            "maximum_relative_absolute_difference_pct": float(
                station_sample.relative_absolute_difference_pct.max()
            ),
            "p99_relative_absolute_difference_pct": float(
                station_sample.relative_absolute_difference_pct.quantile(0.99)
            ),
        },
        "pass_criterion": (
            "native-grid maximum absolute difference <= 5e-5 micrograms m-3; "
            "station-sampling mean <= 0.2 and maximum <= 2.0 micrograms m-3"
        ),
        "passed": bool(
            native_maximum <= 5.0e-5
            and station_mean <= 0.2
            and station_maximum <= 2.0
        ),
    }
    return comparison, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--earth-engine", type=Path, default=DEFAULT_EARTH_ENGINE)
    parser.add_argument("--earth-engine-project", default="ee-alberthnahas")
    args = parser.parse_args()
    comparison, manifest = compare(
        args.direct_archive, args.earth_engine, args.earth_engine_project
    )
    output = ROOT / "tables" / "cams_source_equivalence.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.part")
    comparison.to_csv(temporary, index=False)
    os.replace(temporary, output)
    manifest["comparison_table"] = {
        "path": str(output.relative_to(ROOT)),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
    }
    manifest_path = (
        ROOT / "data" / "external" / "cams_source_validation" / "manifest.json"
    )
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
