from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pm25ml.data import (  # noqa: E402
    ExperimentPaths,
    _add_network_features,
    assign_split,
    build_issue_features,
    load_and_quality_control_observations,
    sample_cams_at_stations,
    validate_modeling_table,
)


def _config() -> dict:
    return {
        "observation_glob": "../obs/*.csv",
        "forecast_hours": [1],
        "forecast_cycle_hours_utc": [0],
        "quality_control": {
            "pm25_min_inclusive_ug_m3": 0.0,
            "pm25_max_exclusive_ug_m3": 985.0,
            "relative_humidity_min_exclusive_pct": 0.0,
            "relative_humidity_max_inclusive_pct": 100.0,
            "temperature_min_inclusive_c": -10.0,
            "temperature_max_inclusive_c": 50.0,
        },
        "features": {
            "pm25_lags_hours": [0, 1],
            "meteorology_lags_hours": [0, 1],
            "rolling_windows_hours": [2],
            "network_radius_km": 400.0,
        },
        "splits": {
            "training_target_start_utc": "2023-01-01T00:00:00Z",
            "training_target_end_utc": "2023-12-31T23:00:00Z",
            "validation_target_start_utc": "2024-01-01T00:00:00Z",
            "validation_target_end_utc": "2024-12-31T23:00:00Z",
            "test_target_start_utc": "2025-01-01T00:00:00Z",
            "test_target_end_utc": "2025-12-31T23:00:00Z",
        },
    }


def _metadata(source_file: str = "station.csv") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_file": [source_file],
            "station_code": ["STA"],
            "station_name": ["Test Station"],
            "province": ["Test"],
            "region": ["Test"],
            "latitude": [-0.5],
            "longitude": [100.5],
            "timezone": ["WIB"],
            "utc_offset_hours": [7],
        }
    )


def test_qc_preserves_plausible_extreme_and_removes_only_exact_duplicate(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    obs = tmp_path / "obs"
    root.mkdir()
    obs.mkdir()
    _metadata().to_csv(root / "station_metadata.csv", index=False)
    source = pd.DataFrame(
        {
            "STASIUN": ["STA"] * 5,
            "TANGGAL": ["2023-01-01"] * 5,
            "JAM_UTC": [0, 0, 1, 2, 3],
            "PM25": [10, 10, -1, 985, 984.9],
            "RH": [80] * 5,
            "TEMP": [25, 25, 25, 25, 688.0],
        }
    )
    source.to_csv(obs / "station.csv", index=False)
    clean, quality, _, overall = load_and_quality_control_observations(
        _config(), ExperimentPaths(root)
    )
    assert len(clean) == 4
    assert overall["duplicate_rows_removed"] == 1
    assert overall["negative_pm25"] == 1
    assert overall["pm25_at_or_above_985"] == 1
    assert clean.loc[clean.PM25.eq(984.9), "pm25_ug_m3"].iloc[0] == 984.9
    assert np.isnan(clean.loc[clean.TEMP.eq(688.0), "temperature_c"].iloc[0])
    assert quality.invalid_temperature.sum() == 1


def test_conflicting_duplicate_fails(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    obs = tmp_path / "obs"
    root.mkdir()
    obs.mkdir()
    _metadata().to_csv(root / "station_metadata.csv", index=False)
    pd.DataFrame(
        {
            "STASIUN": ["STA", "STA"],
            "TANGGAL": ["2023-01-01", "2023-01-01"],
            "JAM_UTC": [0, 0],
            "PM25": [10, 11],
            "RH": [80, 80],
            "TEMP": [25, 25],
        }
    ).to_csv(obs / "station.csv", index=False)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        load_and_quality_control_observations(_config(), ExperimentPaths(root))


def test_absolute_observation_glob_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "standalone"
    obs = tmp_path / "authorized-input"
    root.mkdir()
    obs.mkdir()
    _metadata().to_csv(root / "station_metadata.csv", index=False)
    pd.DataFrame(
        {
            "STASIUN": ["STA"],
            "TANGGAL": ["2023-01-01"],
            "JAM_UTC": [0],
            "PM25": [10.0],
            "RH": [80.0],
            "TEMP": [25.0],
        }
    ).to_csv(obs / "station.csv", index=False)
    monkeypatch.setenv("PM25_OBSERVATION_GLOB", str(obs / "*.csv"))
    clean, _, _, overall = load_and_quality_control_observations(
        _config(), ExperimentPaths(root)
    )
    assert len(clean) == 1
    assert overall["source_files"] == 1


def test_network_mean_is_valid_when_receptor_value_is_missing() -> None:
    timestamp = pd.Timestamp("2023-01-01T00:00:00Z")
    issues = pd.DataFrame(
        {
            "timestamp_utc": [timestamp] * 3,
            "station_code": ["A", "B", "C"],
            "pm25_lag_0h": [np.nan, 10.0, 20.0],
        }
    )
    metadata = pd.DataFrame(
        {
            "station_code": ["A", "B", "C"],
            "latitude": [0.0, 0.1, 0.2],
            "longitude": [100.0, 100.1, 100.2],
        }
    )
    output = _add_network_features(issues, metadata, 400.0).set_index("station_code")
    assert output.loc["A", "network_pm25_mean_ug_m3"] == pytest.approx(15.0)
    assert output.loc["B", "network_pm25_mean_ug_m3"] == pytest.approx(20.0)


def test_issue_features_do_not_change_when_future_target_changes() -> None:
    metadata = _metadata().drop(columns="source_file")
    time = pd.date_range("2023-01-01", periods=27, freq="h", tz="UTC")
    observations = pd.DataFrame(
        {
            "station_code": "STA",
            "timestamp_utc": time,
            "pm25_ug_m3": np.arange(27, dtype=float),
            "relative_humidity_pct": 80.0,
            "temperature_c": 25.0,
        }
    )
    changed = observations.copy()
    changed.loc[changed.timestamp_utc.eq(pd.Timestamp("2023-01-02T01:00:00Z")), "pm25_ug_m3"] = 900.0
    original_features = build_issue_features(observations, metadata, _config())
    changed_features = build_issue_features(changed, metadata, _config())
    original_issue = original_features.loc[
        original_features.timestamp_utc.eq(pd.Timestamp("2023-01-02T00:00:00Z"))
    ].iloc[0]
    changed_issue = changed_features.loc[
        changed_features.timestamp_utc.eq(pd.Timestamp("2023-01-02T00:00:00Z"))
    ].iloc[0]
    feature_columns = [column for column in original_features if "target_pm25" not in column]
    pd.testing.assert_series_equal(
        original_issue[feature_columns], changed_issue[feature_columns], check_names=False
    )
    assert original_issue.target_pm25_1h != changed_issue.target_pm25_1h


def test_cams_sampling_bilinear_and_unit_conversion(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    netcdf = root / "data" / "external" / "cams_global_forecasts" / "netcdf"
    netcdf.mkdir(parents=True)
    reference = np.array(["2023-01-01T00:00:00"], dtype="datetime64[ns]")
    period = np.array([1, 6], dtype="timedelta64[h]").astype("timedelta64[ns]")
    values = np.array(
        [
            [[1.0e-8, 2.0e-8], [3.0e-8, 4.0e-8]],
            [[2.0e-8, 3.0e-8], [4.0e-8, 5.0e-8]],
        ]
    )[:, None, :, :]
    dataset = xr.Dataset(
        {"pm2p5": (("forecast_period", "forecast_reference_time", "latitude", "longitude"), values)},
        coords={
            "forecast_period": period,
            "forecast_reference_time": reference,
            "latitude": [-1.0, 0.0],
            "longitude": [100.0, 101.0],
        },
    )
    dataset.pm2p5.attrs["units"] = "kg m**-3"
    dataset.to_netcdf(netcdf / "cams_STA_20230101_20230101.nc")
    sampled, manifest = sample_cams_at_stations(_metadata(), ExperimentPaths(root))
    assert sampled.cams_pm25_ug_m3.tolist() == pytest.approx([25.0, 35.0])
    assert sampled.forecast_hour.tolist() == [1, 6]
    assert sampled.cams_temperature_c.isna().all()
    assert manifest["stations"] == 1


def test_split_and_audit_use_target_time_and_required_cams_only() -> None:
    config = _config()
    target = pd.Series(
        pd.to_datetime(
            ["2023-06-01T01:00:00Z", "2024-06-01T01:00:00Z", "2025-06-01T01:00:00Z"]
        )
    )
    assert assign_split(target, config).tolist() == ["train", "validation", "test"]
    modeling = pd.DataFrame(
        {
            "station_code": ["STA"] * 3,
            "issue_time_utc": target - pd.Timedelta(hours=1),
            "target_time_utc": target,
            "forecast_hour": [1] * 3,
            "target_pm25_ug_m3": [10.0, 11.0, 12.0],
            "cams_pm25_ug_m3": [9.0, 10.0, 11.0],
            "cams_temperature_c": [np.nan] * 3,
            "pm25_lag_0h": [9.0, 10.0, 11.0],
            "split": ["train", "validation", "test"],
        }
    )
    report = validate_modeling_table(modeling, config)
    assert report["cams_complete_pct"] == 100.0
    assert report["target_before_or_at_issue"] == 0
