from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pm25ml.data import ExperimentPaths, file_sha256, load_config  # noqa: E402
import pm25ml.shadow as shadow_module  # noqa: E402
from pm25ml.shadow import (  # noqa: E402
    _attempt_candidate_shadow,
    _add_shadow_evidence_fields,
    _run_candidate_shadow,
    build_shadow_issue_features,
    evaluate_promotion_gate,
    parse_dashboard_observations,
    run_daily_shadow,
    verify_shadow_forecasts,
)


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
    (tmp_path / "operational_config.json").write_text(
        '{"schema_version":1,"observation_fresh_max_hours":6,'
        '"frozen_service_initialization_horizons_hours":[24],"promotion_gate":{}}',
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        '{"forecast_hours":[24]}', encoding="utf-8"
    )
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
            "latest_pm25_age_hours": [0.0],
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
    (forecast_dir / "one.json").write_text(
        '{"generation_timestamp_semantics":"forecast_write_completion",'
        '"generation_completed_utc":"2026-09-01T06:00:00Z",'
        '"deployment_manifest_sha256":"test-bundle"}',
        encoding="utf-8",
    )
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
    all_rows = scorecard.loc[scorecard.eligibility_scope.eq("all_research_records")]
    assert all_rows.station_balanced_forecast_mae_ug_m3.iloc[0] == 2.0
    assert all_rows.station_balanced_persistence_mae_ug_m3.iloc[0] == 12.0


def test_retrospective_and_stale_rows_are_retained_but_ineligible() -> None:
    issue = pd.Timestamp("2026-09-01T00:00:00Z")
    frame = pd.DataFrame(
        {
            "issue_time_utc": [issue, issue],
            "target_time_utc": [
                issue + pd.Timedelta(hours=6),
                issue + pd.Timedelta(hours=24),
            ],
            "latest_pm25_age_hours": [0.0, 1.0],
            "forecast_pm25_ug_m3": [10.0, 20.0],
        }
    )
    result = _add_shadow_evidence_fields(
        frame, issue + pd.Timedelta(hours=10), "exact_completion_timestamp"
    )
    assert len(result) == 2
    assert not result.prospective_evaluation_eligible.any()
    assert result.evaluation_eligibility_reason.tolist() == [
        "target_reached_before_completion",
        "observation_stale_or_unavailable",
    ]
    assert not result.duty_service_eligible.any()


def test_sparse_support_gate_cannot_pass() -> None:
    issue = pd.Timestamp("2026-09-01T00:00:00Z")
    matched = pd.DataFrame(
        {
            "model_bundle_id": ["v1"],
            "point_model_role": ["primary_point"],
            "prospective_evaluation_eligible": [True],
            "issue_time_utc": [issue],
        }
    )
    scorecard = pd.DataFrame(
        {
            "model_bundle_id": ["v1"],
            "point_model_role": ["primary_point"],
            "eligibility_scope": ["prospective_eligible"],
            "forecast_hour": [24],
            "stations": [1],
            "minimum_rows_per_station": [1],
            "skill_from_station_balanced_mae_pct": [50.0],
            "improvement_ci95_lower_ug_m3": [1.0],
            "stations_harmed": [0],
            "interval_coverage_pct": [80.0],
        }
    )
    config = {
        "forecast_hours": [24],
        "operational": {
            "frozen_service_initialization_horizons_hours": [24],
            "promotion_gate": {},
        },
    }
    gate = evaluate_promotion_gate(matched, scorecard, [], config)
    assert gate["status"] == "fail"
    reasons = gate["bundle_results"][0]["failure_reasons"]
    assert any("eligible_days" in reason for reason in reasons)
    assert any("station_support" in reason for reason in reasons)


def test_candidate_verification_preserves_10utc_timing_and_version_bundle(
    tmp_path: Path,
) -> None:
    paths = ExperimentPaths(tmp_path)
    (tmp_path / "operational_config.json").write_text(
        '{"schema_version":1,"observation_fresh_max_hours":6,'
        '"frozen_service_initialization_horizons_hours":[24],"promotion_gate":{}}',
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        '{"forecast_hours":[24]}', encoding="utf-8"
    )
    issue = pd.Timestamp("2026-09-08T00:00:00Z")
    target = issue + pd.Timedelta(hours=24)
    primary_dir = tmp_path / "shadow" / "forecasts"
    candidate_dir = tmp_path / "shadow" / "candidate_forecasts"
    primary_dir.mkdir(parents=True)
    candidate_dir.mkdir(parents=True)
    common = {
        "station_code": ["A"],
        "issue_time_utc": [issue],
        "target_time_utc": [target],
        "forecast_hour": [24],
        "forecast_pm25_ug_m3": [20.0],
        "pm25_lag_0h": [10.0],
        "latest_pm25_age_hours": [0.0],
        "cams_pm25_ug_m3": [18.0],
        "forecast_status": ["primary"],
    }
    primary = pd.DataFrame(
        {
            **common,
            "prediction_q10_ug_m3": [15.0],
            "prediction_q50_ug_m3": [20.0],
            "prediction_q90_ug_m3": [25.0],
            "generated_utc": [issue + pd.Timedelta(hours=6)],
        }
    )
    primary_path = primary_dir / "primary.csv"
    primary.to_csv(primary_path, index=False)
    (primary_dir / "primary.json").write_text(
        json.dumps(
            {
                "generation_timestamp_semantics": "forecast_write_completion",
                "generation_completed_utc": "2026-09-08T06:00:00Z",
                "deployment_manifest_sha256": "frozen-v1",
                "output_sha256": file_sha256(primary_path),
            }
        ),
        encoding="utf-8",
    )
    candidate = pd.DataFrame(
        {
            **common,
            "asof_time_utc": [issue + pd.Timedelta(hours=10)],
            "prediction_lower_ug_m3": [14.0],
            "prediction_upper_ug_m3": [26.0],
            "forecast_status": ["candidate_shadow"],
            "candidate_version": ["point-v1"],
            "interval_version": ["interval-v2"],
            "generated_utc": [issue + pd.Timedelta(hours=10.5)],
            "prospective_evaluation_eligible": [True],
            "evaluation_eligibility_reason": [
                "eligible_for_candidate_shadow_evaluation"
            ],
        }
    )
    candidate_path = candidate_dir / "candidate.csv"
    candidate.to_csv(candidate_path, index=False)
    (candidate_dir / "candidate.json").write_text(
        json.dumps(
            {
                "generation_completed_utc": "2026-09-08T10:30:00Z",
                "generation_timestamp_semantics": "forecast_write_completion",
                "output_sha256": file_sha256(candidate_path),
            }
        ),
        encoding="utf-8",
    )
    observations = pd.DataFrame(
        {
            "station_code": ["A"],
            "timestamp_utc": [target],
            "pm25_ug_m3": [22.0],
        }
    )
    matched, scorecard = verify_shadow_forecasts(paths, observations)
    candidate_matched = matched.loc[matched.point_model_role.eq("candidate_point")]
    assert len(candidate_matched) == 1
    assert candidate_matched.observation_age_at_completion_hours.iloc[0] == 0.5
    assert candidate_matched.prospective_evaluation_eligible.iloc[0]
    assert candidate_matched.interval_covered.iloc[0]
    assert set(scorecard.model_bundle_id) == {
        "frozen-v1",
        "candidate::point-v1::interval::interval-v2",
    }


def test_candidate_failure_is_isolated_and_next_attempt_can_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = ExperimentPaths(tmp_path)
    calls = 0

    def fail_then_succeed(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("isolated candidate failure")
        return {"status": "candidate_forecast_generated"}

    monkeypatch.setattr(shadow_module, "_run_candidate_shadow", fail_then_succeed)
    issue = pd.Timestamp("2026-09-01T00:00:00Z")
    first = _attempt_candidate_shadow(paths, issue, pd.DataFrame(), {})
    second = _attempt_candidate_shadow(paths, issue, pd.DataFrame(), {})
    assert first["status"] == "candidate_forecast_failed"
    assert second["status"] == "candidate_forecast_generated"
    assert len(list((tmp_path / "shadow" / "candidate_state").glob("failed_*.json"))) == 1


def test_existing_frozen_forecast_does_not_skip_candidate_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = ExperimentPaths(tmp_path)
    issue = pd.Timestamp("2026-09-01T00:00:00Z")
    forecast_dir = tmp_path / "shadow" / "forecasts"
    forecast_dir.mkdir(parents=True)
    forecast_path = forecast_dir / "pm25_shadow_20260901T0000Z.csv"
    pd.DataFrame(
        {
            "station_code": ["A"],
            "issue_time_utc": [issue],
            "target_time_utc": [issue + pd.Timedelta(hours=24)],
            "generated_utc": [issue + pd.Timedelta(hours=1)],
        }
    ).to_csv(forecast_path, index=False)
    forecast_path.with_suffix(".json").write_text('{"warnings":[]}', encoding="utf-8")
    monkeypatch.setattr(
        shadow_module,
        "acquire_dashboard_observations",
        lambda paths: (
            pd.DataFrame(),
            {"retrieved_utc": "2026-09-01T10:15:00Z"},
        ),
    )
    attempted: list[bool] = []
    monkeypatch.setattr(
        shadow_module,
        "_attempt_candidate_shadow",
        lambda *args, **kwargs: attempted.append(True)
        or {"status": "candidate_forecast_generated"},
    )
    monkeypatch.setattr(
        shadow_module,
        "verify_shadow_forecasts",
        lambda paths, observations: (pd.DataFrame(), pd.DataFrame()),
    )
    result = run_daily_shadow("2026-09-01", paths)
    assert attempted == [True]
    assert result["status"] == "forecast_already_exists"
    assert result["candidate_shadow"]["status"] == "candidate_forecast_generated"


def test_shadow_feature_cutoff_excludes_later_dashboard_arrival(tmp_path: Path) -> None:
    paths = ExperimentPaths(tmp_path)
    paths.derived.mkdir(parents=True)
    issue = pd.Timestamp("2026-09-08T10:00:00Z")
    historical = pd.DataFrame(
        {
            "station_code": ["A", "A"],
            "timestamp_utc": [issue - pd.Timedelta(hours=1), issue],
            "pm25_ug_m3": [8.0, 5.0],
            "relative_humidity_pct": [50.0, 50.0],
            "temperature_c": [25.0, 25.0],
        }
    )
    historical.to_csv(
        paths.derived / "observations_quality_controlled.csv.gz", index=False
    )
    pd.DataFrame(
        {
            "source_file": ["a.csv"],
            "station_code": ["A"],
            "station_name": ["A station"],
            "province": ["X"],
            "region": ["Sumatra"],
            "latitude": [0.0],
            "longitude": [100.0],
            "timezone": ["WIB"],
            "utc_offset_hours": [7],
            "coordinate_source": ["test"],
            "coordinate_source_id": ["test"],
        }
    ).to_csv(paths.station_metadata, index=False)
    dashboard = pd.DataFrame(
        {
            "station_code": ["A", "A"],
            "timestamp_utc": [issue, issue],
            "pm25_ug_m3": [10.0, 99.0],
            "source_retrieved_utc": [
                issue + pd.Timedelta(minutes=5),
                issue + pd.Timedelta(minutes=20),
            ],
        }
    )
    config = load_config(ROOT / "config.json")
    config["forecast_cycle_hours_utc"] = [10]
    config["forecast_hours"] = [2]
    features, manifest = build_shadow_issue_features(
        paths,
        dashboard,
        issue,
        availability_cutoff_utc=issue + pd.Timedelta(minutes=10),
        feature_config=config,
    )
    assert features.pm25_lag_0h.iloc[0] == 10.0
    assert manifest["dashboard_rows_excluded_after_cutoff"] == 1


def test_existing_candidate_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    paths = ExperimentPaths(tmp_path)
    output = (
        tmp_path
        / "shadow"
        / "candidate_forecasts"
        / "pm25_candidate_shadow_20260901T0000Z.csv"
    )
    output.parent.mkdir(parents=True)
    output.write_text("x\n1\n", encoding="utf-8")
    output.with_suffix(".json").write_text(
        '{"output_sha256":"wrong"}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        _run_candidate_shadow(
            paths,
            pd.Timestamp("2026-09-01T00:00:00Z"),
            pd.DataFrame(),
            {},
        )
