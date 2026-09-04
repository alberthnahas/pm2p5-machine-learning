from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pm25ml.data import ExperimentPaths  # noqa: E402
from pm25ml.deployment import run_operational_forecast  # noqa: E402
from pm25ml.modeling import TrainedModel  # noqa: E402


def _constant_model(value: float, horizon: int) -> TrainedModel:
    estimator = DummyRegressor(strategy="constant", constant=value)
    estimator.fit(pd.DataFrame({"pm25_lag_0h": [1.0, 2.0]}), [value, value])
    return TrainedModel(
        model_name="dummy",
        forecast_hour=horizon,
        feature_variant="test",
        feature_columns=["pm25_lag_0h"],
        estimator=estimator,
        best_iteration=1,
        training_rows=2,
        validation_rows=0,
        training_seconds=0.0,
    )


def test_operational_primary_and_fallback_statuses(tmp_path: Path) -> None:
    paths = ExperimentPaths(tmp_path)
    (paths.derived).mkdir(parents=True)
    paths.provenance.mkdir(parents=True)
    model_dir = tmp_path / "models" / "deployment"
    model_dir.mkdir(parents=True)
    config = {
        "experiment_name": "test",
        "forecast_hours": [1],
        "forecast_cycle_hours_utc": [0],
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    models = []
    for role, value in (
        ("primary_point", 20.0),
        ("observation_only_fallback", 15.0),
        ("quantile_10", 10.0),
        ("quantile_50", 20.0),
        ("quantile_90", 30.0),
    ):
        path = model_dir / f"{role}_001h.joblib"
        joblib.dump(_constant_model(value, 1), path)
        models.append(
            {
                "role": role,
                "forecast_hour": 1,
                "path": str(path.relative_to(tmp_path)),
                "source_columns": ["pm25_lag_0h"],
            }
        )
    manifest = {
        "primary_requires_cams_pm25": True,
        "interval_calibration": {"correction_ug_m3": {"1": 2.0}},
        "models": models,
    }
    (paths.provenance / "deployment_model_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    issue = pd.Timestamp("2026-01-01T00:00:00Z")
    features = pd.DataFrame(
        {
            "timestamp_utc": [issue, issue],
            "station_code": ["A", "B"],
            "station_name": ["A station", "B station"],
            "province": ["X", "Y"],
            "region": ["west", "east"],
            "timezone": ["WIB", "WITA"],
            "utc_offset_hours": [7, 8],
            "pm25_lag_0h": [12.0, 13.0],
            "latest_pm25_age_hours": [0.0, 0.0],
        }
    )
    features.to_csv(paths.derived / "issue_time_observation_features.csv.gz", index=False)
    cams = pd.DataFrame(
        {
            "station_code": ["A"],
            "issue_time_utc": [issue],
            "valid_time_utc": [issue + pd.Timedelta(hours=1)],
            "forecast_hour": [1],
            "cams_pm25_ug_m3": [18.0],
        }
    )
    cams.to_csv(paths.derived / "cams_station_forecasts.csv.gz", index=False)
    output, metadata = run_operational_forecast(
        issue.isoformat(), paths=paths
    )
    output = output.set_index("station_code")
    assert output.loc["A", "forecast_pm25_ug_m3"] == 20.0
    assert output.loc["A", "prediction_q10_ug_m3"] == 8.0
    assert output.loc["A", "prediction_q90_ug_m3"] == 32.0
    assert output.loc["A", "forecast_status"] == "primary"
    assert output.loc["B", "forecast_pm25_ug_m3"] == 15.0
    assert np.isnan(output.loc["B", "prediction_q50_ug_m3"])
    assert output.loc["B", "forecast_status"] == "observation_only_fallback"
    assert metadata["primary_rows"] == 1
    assert metadata["degraded_rows"] == 1
