from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pm25ml.modeling import (  # noqa: E402
    _conformal_quantile,
    _interval_period_mask,
    choose_champion,
    encode_features,
    metric_values,
)


def test_encoding_aligns_unknown_category_without_changing_schema() -> None:
    train = pd.DataFrame(
        {
            "station_code": ["A", "B"],
            "region": ["west", "east"],
            "timezone": ["WIB", "WITA"],
            "latitude": [0.0, 1.0],
        }
    )
    test = pd.DataFrame(
        {
            "station_code": ["C"],
            "region": ["east"],
            "timezone": ["WIT"],
            "latitude": [2.0],
        }
    )
    columns = ["station_code", "region", "timezone", "latitude"]
    encoded_train = encode_features(train, columns)
    encoded_test = encode_features(test, columns, expected_columns=encoded_train.columns)
    assert encoded_test.columns.tolist() == encoded_train.columns.tolist()
    assert encoded_test.shape == (1, encoded_train.shape[1])


def test_champion_selection_uses_validation_only_and_operational_tie_break() -> None:
    rows = []
    for model, mae in (("obs_lgbm", 10.0), ("cams_lgbm", 10.05), ("cams_xgboost", 11.0)):
        for horizon in (1, 6):
            rows.append(
                {
                    "split": "validation",
                    "scope": "station_balanced_common_cases",
                    "model": model,
                    "forecast_hour": horizon,
                    "mae_ug_m3": mae,
                    "skill_vs_persistence_pct": 5.0,
                }
            )
    summary = pd.DataFrame(rows)
    champion, ranking = choose_champion(summary)
    assert champion == "cams_lgbm"
    assert ranking.selected_champion.sum() == 1


def test_conformal_correction_is_expansion_only() -> None:
    assert _conformal_quantile(np.array([-5.0, -4.0, -3.0]), 0.8) == 0.0
    assert _conformal_quantile(np.array([1.0, 2.0, 3.0]), 0.8) >= 0.0


def test_interval_tuning_precedes_disjoint_calibration() -> None:
    frame = pd.DataFrame(
        {
            "target_time_utc": pd.to_datetime(
                [
                    "2025-06-30T23:00:00Z",
                    "2025-07-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                ],
                utc=True,
            )
        }
    )
    config = {
        "prediction_intervals": {
            "quantile_tuning_target_start_utc": "2025-01-01T00:00:00Z",
            "quantile_tuning_target_end_utc": "2025-06-30T23:00:00Z",
            "conformal_calibration_target_start_utc": "2025-07-01T00:00:00Z",
            "conformal_calibration_target_end_utc": "2025-12-31T23:00:00Z",
        }
    }
    tuning = _interval_period_mask(frame, config, "tuning")
    calibration = _interval_period_mask(frame, config, "calibration")
    assert tuning.tolist() == [True, False, False]
    assert calibration.tolist() == [False, True, False]
    assert not (tuning & calibration).any()


def test_metrics_have_expected_sign_convention() -> None:
    observed = pd.Series([10.0, 20.0, 30.0])
    predicted = pd.Series([12.0, 22.0, 32.0])
    values = metric_values(observed, predicted)
    assert values["mae_ug_m3"] == 2.0
    assert values["bias_ug_m3"] == 2.0
    assert values["correlation"] == 1.0
