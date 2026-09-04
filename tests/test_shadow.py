from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pm25ml.data import ExperimentPaths  # noqa: E402
from pm25ml.shadow import parse_dashboard_observations, verify_shadow_forecasts  # noqa: E402


def test_dashboard_parser_converts_local_time_and_applies_pm25_qc() -> None:
    metadata = pd.DataFrame(
        {
            "source_file": ["example_station.csv"],
            "station_code": ["EXAMPLE1"],
            "station_name": ["Example Station"],
            "timezone": ["WITA"],
            "utc_offset_hours": [8],
        }
    )
    payload = {
        "locations": {
            "EXAMPLE STATION": {
                "latest": {"timezone": "WITA"},
                "timeseries": {
                    "labels": ["2026-09-04 08:00", "2026-09-04 09:00"],
                    "values": [25.0, 99999.0],
                },
            }
        }
    }
    parsed = parse_dashboard_observations(
        payload, metadata, pd.Timestamp("2026-09-04T02:00:00Z")
    )
    assert parsed.timestamp_utc.iloc[0] == pd.Timestamp("2026-09-04T00:00:00Z")
    assert parsed.pm25_ug_m3.iloc[0] == 25.0
    assert pd.isna(parsed.pm25_ug_m3.iloc[1])
    assert parsed.pm25_qc.iloc[1] == "at_or_above_985"


def test_shadow_verification_matches_target_time_and_computes_skill(tmp_path: Path) -> None:
    paths = ExperimentPaths(tmp_path)
    forecast_dir = tmp_path / "shadow" / "forecasts"
    forecast_dir.mkdir(parents=True)
    issue = pd.Timestamp("2026-09-01T00:00:00Z")
    forecast = pd.DataFrame(
        {
            "station_code": ["A"],
            "issue_time_utc": [issue],
            "target_time_utc": [issue + pd.Timedelta(hours=24)],
            "forecast_hour": [24],
            "forecast_pm25_ug_m3": [20.0],
            "pm25_lag_0h": [10.0],
            "cams_pm25_ug_m3": [18.0],
            "prediction_q10_ug_m3": [15.0],
            "prediction_q50_ug_m3": [20.0],
            "prediction_q90_ug_m3": [25.0],
            "forecast_status": ["primary"],
            "generated_utc": [issue + pd.Timedelta(hours=6)],
            "generation_status": ["prospective"],
        }
    )
    forecast.to_csv(forecast_dir / "one.csv", index=False)
    observations = pd.DataFrame(
        {
            "station_code": ["A"],
            "timestamp_utc": [issue + pd.Timedelta(hours=24)],
            "pm25_ug_m3": [22.0],
        }
    )
    matched, scorecard = verify_shadow_forecasts(paths, observations)
    assert len(matched) == 1
    assert matched.forecast_absolute_error_ug_m3.iloc[0] == 2.0
    assert scorecard.forecast_mae_ug_m3.iloc[0] == 2.0
    assert scorecard.persistence_mae_ug_m3.iloc[0] == 12.0
