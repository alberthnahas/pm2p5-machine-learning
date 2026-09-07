"""Chronological development experiment for availability-aligned PM2.5 forecasts."""

from __future__ import annotations

import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data import (
    ExperimentPaths,
    build_issue_features,
    file_sha256,
    load_config,
    write_json,
)
from .deployment import _validated_deployment_bundle
from .modeling import (
    TrainedModel,
    _climatology_predictor,
    _conformal_quantile,
    _fit_lightgbm,
    feature_columns,
    load_modeling_table,
    predict_model,
)


METEOROLOGY_PREFIXES = ("rh_lag_", "temperature_lag_")


def _candidate_config(paths: ExperimentPaths) -> dict[str, Any]:
    return json.loads((paths.root / "candidate_config.json").read_text(encoding="utf-8"))


def _no_met_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if not column.startswith(METEOROLOGY_PREFIXES)]


def _asof_10_table(
    paths: ExperimentPaths, base_config: dict[str, Any], candidate: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    observation_path = paths.derived / "observations_quality_controlled.csv.gz"
    observations = pd.read_csv(
        observation_path,
        parse_dates=["timestamp_utc"],
        low_memory=False,
    )
    observations["timestamp_utc"] = pd.to_datetime(observations.timestamp_utc, utc=True)
    metadata = pd.read_csv(paths.station_metadata)
    mapping = {int(key): int(value) for key, value in candidate["lead_mapping_hours"].items()}
    feature_config = json.loads(json.dumps(base_config))
    feature_config["forecast_cycle_hours_utc"] = [int(candidate["observation_asof_hour_utc"])]
    feature_config["forecast_hours"] = sorted(mapping.values())
    issues = build_issue_features(observations, metadata, feature_config)

    cams_paths = sorted(paths.cams_earth_engine.glob("cams_station_forecasts_*.csv.gz"))
    if len(cams_paths) != 1:
        raise ValueError("Expected one combined historical CAMS station archive")
    cams = pd.read_csv(
        cams_paths[0],
        usecols=[
            "station_code",
            "issue_time_utc",
            "valid_time_utc",
            "forecast_hour",
            "cams_pm25_ug_m3",
            "source_image_id",
        ],
        parse_dates=["issue_time_utc", "valid_time_utc"],
        low_memory=False,
    )
    cams["issue_time_utc"] = pd.to_datetime(cams.issue_time_utc, utc=True)
    cams["valid_time_utc"] = pd.to_datetime(cams.valid_time_utc, utc=True)
    frames: list[pd.DataFrame] = []
    for initialization_horizon, remaining_horizon in mapping.items():
        frame = issues.copy()
        frame["asof_time_utc"] = frame.timestamp_utc
        frame["issue_time_utc"] = frame.timestamp_utc.dt.floor("D")
        frame["forecast_hour"] = initialization_horizon
        frame["remaining_horizon_hours"] = remaining_horizon
        frame["target_time_utc"] = frame.asof_time_utc + pd.Timedelta(
            hours=remaining_horizon
        )
        frame["valid_time_utc"] = frame.target_time_utc
        frame["target_pm25_ug_m3"] = frame[f"target_pm25_{remaining_horizon}h"]
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table = table.merge(
        cams,
        on=["station_code", "issue_time_utc", "forecast_hour", "valid_time_utc"],
        how="left",
        validate="many_to_one",
    )
    if not table.valid_time_utc.eq(table.target_time_utc).all():
        raise ValueError("+10 UTC target times do not match CAMS valid times")
    local_target = table.target_time_utc + pd.to_timedelta(
        table.utc_offset_hours, unit="h"
    )
    table["target_hour_local"] = local_target.dt.hour
    table["target_hour_local_sin"] = np.sin(2 * np.pi * local_target.dt.hour / 24.0)
    table["target_hour_local_cos"] = np.cos(2 * np.pi * local_target.dt.hour / 24.0)
    day = local_target.dt.dayofyear
    table["target_day_of_year_sin"] = np.sin(2 * np.pi * day / 365.25)
    table["target_day_of_year_cos"] = np.cos(2 * np.pi * day / 365.25)
    table["target_month"] = local_target.dt.month
    table = table.loc[
        table.target_time_utc.between(
            pd.Timestamp(base_config["splits"]["training_target_start_utc"]),
            pd.Timestamp(base_config["splits"]["test_target_end_utc"]),
            inclusive="both",
        )
    ].copy()
    audit = {
        "observation_source": str(observation_path.relative_to(paths.root)),
        "observation_sha256": file_sha256(observation_path),
        "cams_source": str(cams_paths[0].relative_to(paths.root)),
        "cams_sha256": file_sha256(cams_paths[0]),
        "rows": len(table),
        "stations": int(table.station_code.nunique()),
        "target_start_utc": table.target_time_utc.min().isoformat(),
        "target_end_utc": table.target_time_utc.max().isoformat(),
        "mapping": mapping,
        "target_time_mismatches": int((table.valid_time_utc != table.target_time_utc).sum()),
        "cams_missing_rows": int(table.cams_pm25_ug_m3.isna().sum()),
    }
    return table, audit


def _period(frame: pd.DataFrame, start: str | None, end: str) -> pd.DataFrame:
    mask = frame.target_time_utc.le(pd.Timestamp(end))
    if start is not None:
        mask &= frame.target_time_utc.ge(pd.Timestamp(start))
    return frame.loc[mask & frame.target_pm25_ug_m3.notna() & frame.cams_pm25_ug_m3.notna()].copy()


def _fit_refit_predict(
    name: str,
    horizon: int,
    columns: list[str],
    fit: pd.DataFrame,
    tune: pd.DataFrame,
    calibration: pd.DataFrame,
    assessment: pd.DataFrame,
    config: dict[str, Any],
    missing_met_stress: bool = False,
) -> tuple[np.ndarray, np.ndarray, TrainedModel, int]:
    selected = _fit_lightgbm(
        name,
        horizon,
        "availability_development",
        columns,
        fit,
        tune,
        config,
    )
    refit = _fit_lightgbm(
        name,
        horizon,
        "availability_development_refit",
        columns,
        pd.concat([fit, tune], ignore_index=True),
        tune.iloc[0:0],
        config,
        n_estimators=selected.best_iteration,
        use_early_stopping=False,
    )
    assessment_input = assessment.copy()
    calibration_input = calibration.copy()
    if missing_met_stress:
        met_columns = [column for column in columns if column.startswith(METEOROLOGY_PREFIXES)]
        assessment_input.loc[:, met_columns] = np.nan
        calibration_input.loc[:, met_columns] = np.nan
    return (
        predict_model(refit, assessment_input, columns),
        predict_model(refit, calibration_input, columns),
        refit,
        selected.best_iteration,
    )


def _frozen_predict(
    frame: pd.DataFrame,
    model: TrainedModel,
    columns: list[str],
    missing_met: bool,
) -> np.ndarray:
    source = frame.copy()
    if missing_met:
        met_columns = [column for column in columns if column.startswith(METEOROLOGY_PREFIXES)]
        source.loc[:, met_columns] = np.nan
    return predict_model(model, source, columns)


def run_candidate_forecast(
    asof_time: pd.Timestamp,
    issue_features: pd.DataFrame,
    cams: pd.DataFrame,
    output_path: Path,
    paths: ExperimentPaths | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the frozen +10 UTC candidate without modifying the primary bundle."""

    paths = paths or ExperimentPaths()
    started = time.perf_counter()
    candidate_config_path = paths.root / "candidate_config.json"
    candidate = _candidate_config(paths)
    point_manifest_path = paths.provenance / "candidate_availability_manifest.json"
    interval_manifest_path = paths.provenance / "candidate_interval_v2_manifest.json"
    point_manifest = json.loads(point_manifest_path.read_text(encoding="utf-8"))
    interval_manifest = json.loads(interval_manifest_path.read_text(encoding="utf-8"))
    if int(point_manifest.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported candidate point-manifest schema")
    if int(interval_manifest.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported candidate interval-manifest schema")
    if point_manifest["candidate_config_sha256"] != file_sha256(candidate_config_path):
        raise ValueError("Candidate configuration checksum mismatch")
    if interval_manifest["candidate_config_sha256"] != file_sha256(candidate_config_path):
        raise ValueError("Candidate interval configuration checksum mismatch")
    if point_manifest["base_config_sha256"] != file_sha256(paths.config):
        raise ValueError("Candidate base configuration checksum mismatch")
    if interval_manifest.get("base_config_sha256") != file_sha256(paths.config):
        raise ValueError("Candidate interval base-configuration checksum mismatch")
    if interval_manifest.get("point_manifest_sha256") != file_sha256(
        point_manifest_path
    ):
        raise ValueError("Candidate interval point-manifest checksum mismatch")
    correction_config_path = paths.root / "candidate_interval_v2_config.json"
    if interval_manifest["correction_config_sha256"] != file_sha256(
        correction_config_path
    ):
        raise ValueError("Candidate interval-v2 configuration checksum mismatch")
    if interval_manifest["point_candidate_version"] != point_manifest["candidate_version"]:
        raise ValueError("Candidate point and interval versions are inconsistent")
    if candidate.get("candidate_version") != point_manifest["candidate_version"]:
        raise ValueError("Candidate configuration and point version are inconsistent")
    asof_time = pd.Timestamp(asof_time)
    if asof_time.tzinfo is None:
        raise ValueError("Candidate as-of time must be timezone-aware")
    asof_time = asof_time.tz_convert("UTC")
    if asof_time.hour != int(candidate["observation_asof_hour_utc"]):
        raise ValueError("Candidate supports the predeclared 10 UTC observation as-of only")
    issue_time = asof_time.floor("D")
    features = issue_features.copy()
    features["timestamp_utc"] = pd.to_datetime(features.timestamp_utc, utc=True)
    base = features.loc[features.timestamp_utc.eq(asof_time)].copy()
    if base.empty or base.duplicated(["station_code", "timestamp_utc"]).any():
        raise ValueError("Candidate requires one feature row per station at 10 UTC")
    registry = pd.read_csv(paths.station_metadata)
    if set(base.station_code.astype(str)) != set(registry.station_code.astype(str)):
        raise ValueError("Candidate features do not contain the complete station registry")
    if len(base) != len(registry):
        raise ValueError("Candidate features do not contain exactly one row per station")
    for column, lower, upper in (
        ("latitude", -90.0, 90.0),
        ("longitude", -180.0, 180.0),
        ("utc_offset_hours", -12.0, 14.0),
    ):
        values = pd.to_numeric(base[column], errors="coerce")
        if not np.isfinite(values).all() or not values.between(lower, upper).all():
            raise ValueError(f"Candidate feature {column} is invalid")
    cams = cams.copy()
    cams["issue_time_utc"] = pd.to_datetime(cams.issue_time_utc, utc=True)
    cams["valid_time_utc"] = pd.to_datetime(cams.valid_time_utc, utc=True)
    if cams.duplicated(["station_code", "issue_time_utc", "forecast_hour"]).any():
        raise ValueError("Candidate CAMS inputs contain duplicate keys")
    coefficient_lookup = {
        int(row["forecast_hour"]): row
        for row in interval_manifest["future_coefficients"]
    }
    expected_mapping = {
        int(horizon): int(remaining)
        for horizon, remaining in candidate["lead_mapping_hours"].items()
    }
    model_mapping = {
        int(row["forecast_hour"]): int(row["remaining_horizon_hours"])
        for row in point_manifest["models"]
    }
    if (
        len(model_mapping) != len(point_manifest["models"])
        or len(coefficient_lookup) != len(interval_manifest["future_coefficients"])
        or model_mapping != expected_mapping
        or set(coefficient_lookup) != set(expected_mapping)
    ):
        raise ValueError("Candidate model or interval lead mapping is incomplete")
    outputs: list[pd.DataFrame] = []
    validated_model_paths: dict[int, Path] = {}
    root = paths.root.resolve()
    for model_meta in point_manifest["models"]:
        horizon = int(model_meta["forecast_hour"])
        remaining = int(model_meta["remaining_horizon_hours"])
        relative_model_path = Path(str(model_meta["path"]))
        model_path = (paths.root / relative_model_path).resolve()
        if (
            not model_path.is_relative_to(root)
            or not model_path.is_file()
            or model_path.stat().st_size != int(model_meta["bytes"])
            or file_sha256(model_path) != model_meta["sha256"]
        ):
            raise ValueError(f"Candidate model integrity failed at lead {horizon}")
        if not model_meta.get("source_columns") or not model_meta.get("encoded_columns"):
            raise ValueError(f"Candidate feature contract is invalid at lead {horizon}")
        validated_model_paths[horizon] = model_path
        coefficient_meta = coefficient_lookup[horizon]
        if coefficient_meta["point_model_sha256"] != model_meta["sha256"]:
            raise ValueError("Candidate interval coefficient points to another model")
        if int(coefficient_meta["remaining_horizon_hours"]) != remaining:
            raise ValueError("Candidate interval remaining-horizon mapping mismatch")
        coefficient = float(coefficient_meta["adaptive_interval_coefficient"])
        if not np.isfinite(coefficient) or coefficient < 0:
            raise ValueError("Candidate interval coefficient is invalid")
    # Deserialize only after every candidate model has passed integrity checks.
    for model_meta in point_manifest["models"]:
        horizon = int(model_meta["forecast_hour"])
        remaining = int(model_meta["remaining_horizon_hours"])
        target_time = asof_time + pd.Timedelta(hours=remaining)
        frame = base.copy()
        frame["issue_time_utc"] = issue_time
        frame["asof_time_utc"] = asof_time
        frame["forecast_hour"] = horizon
        frame["remaining_horizon_hours"] = remaining
        frame["target_time_utc"] = target_time
        frame = frame.drop(
            columns=["valid_time_utc", "cams_pm25_ug_m3"], errors="ignore"
        )
        subset = cams.loc[
            cams.issue_time_utc.eq(issue_time) & cams.forecast_hour.eq(horizon),
            ["station_code", "issue_time_utc", "forecast_hour", "valid_time_utc", "cams_pm25_ug_m3"],
        ]
        expected_stations = set(registry.station_code.astype(str))
        if (
            len(subset) != len(registry)
            or set(subset.station_code.astype(str)) != expected_stations
        ):
            raise ValueError(f"Candidate CAMS coverage is incomplete at lead {horizon}")
        cams_values = pd.to_numeric(subset.cams_pm25_ug_m3, errors="coerce")
        if not np.isfinite(cams_values).all() or cams_values.lt(0).any():
            raise ValueError(f"Candidate CAMS PM2.5 is invalid at lead {horizon}")
        frame = frame.merge(
            subset,
            on=["station_code", "issue_time_utc", "forecast_hour"],
            how="left",
            validate="one_to_one",
        )
        valid_time = frame.issue_time_utc + pd.to_timedelta(frame.forecast_hour, unit="h")
        if frame.valid_time_utc.isna().any() or not frame.valid_time_utc.eq(valid_time).all():
            raise ValueError("Candidate CAMS valid-time mismatch")
        if target_time != issue_time + pd.Timedelta(hours=horizon):
            raise ValueError("Candidate target mapping mismatch")
        local_target = frame.target_time_utc + pd.to_timedelta(frame.utc_offset_hours, unit="h")
        frame["target_hour_local"] = local_target.dt.hour
        frame["target_hour_local_sin"] = np.sin(2 * np.pi * local_target.dt.hour / 24.0)
        frame["target_hour_local_cos"] = np.cos(2 * np.pi * local_target.dt.hour / 24.0)
        day = local_target.dt.dayofyear
        frame["target_day_of_year_sin"] = np.sin(2 * np.pi * day / 365.25)
        frame["target_day_of_year_cos"] = np.cos(2 * np.pi * day / 365.25)
        frame["target_month"] = local_target.dt.month
        missing = set(model_meta["source_columns"]).difference(frame.columns)
        if missing:
            raise ValueError(f"Candidate inference features are absent: {sorted(missing)}")
        available = np.isfinite(pd.to_numeric(frame.cams_pm25_ug_m3, errors="coerce"))
        prediction = np.full(len(frame), np.nan)
        if available.any():
            model: TrainedModel = joblib.load(validated_model_paths[horizon])
            if model.forecast_hour != horizon or model.feature_columns != model_meta["encoded_columns"]:
                raise ValueError("Candidate deserialized model identity mismatch")
            prediction[available] = predict_model(
                model, frame.loc[available], model_meta["source_columns"]
            )
        coefficient = float(coefficient_lookup[horizon]["adaptive_interval_coefficient"])
        scale = np.maximum.reduce(
            [
                np.full(len(frame), 10.0),
                np.nan_to_num(prediction, nan=0.0),
                frame.pm25_lag_0h.fillna(0.0).clip(lower=0).to_numpy(float),
            ]
        )
        output = frame[
            [
                "station_code", "station_name", "province", "region", "timezone",
                "issue_time_utc", "asof_time_utc", "target_time_utc", "forecast_hour",
                "remaining_horizon_hours", "cams_pm25_ug_m3", "pm25_lag_0h",
                "latest_pm25_age_hours",
            ]
        ].copy()
        output["forecast_pm25_ug_m3"] = prediction
        output["prediction_lower_ug_m3"] = np.where(
            available, np.maximum(prediction - coefficient * scale, 0.0), np.nan
        )
        output["prediction_upper_ug_m3"] = np.where(
            available, prediction + coefficient * scale, np.nan
        )
        output["forecast_status"] = np.where(
            available, "candidate_shadow", "candidate_cams_unavailable"
        )
        output["candidate_version"] = point_manifest["candidate_version"]
        output["interval_version"] = interval_manifest["interval_version"]
        output["model_bundle_id"] = (
            f"candidate::{point_manifest['candidate_version']}::interval::"
            f"{interval_manifest['interval_version']}"
        )
        output["point_model_role"] = "candidate_point"
        output["point_model_sha256"] = model_meta["sha256"]
        outputs.append(output)
    result = pd.concat(outputs, ignore_index=True).sort_values(
        ["station_code", "forecast_hour"]
    )
    generated = pd.Timestamp.now(tz="UTC")
    frozen_utc = max(
        pd.Timestamp(point_manifest["generated_utc"]),
        pd.Timestamp(interval_manifest["generated_utc"]),
    )
    result["generated_utc"] = generated
    result["lead_remaining_at_generation_hours"] = (
        result.target_time_utc - generated
    ) / pd.Timedelta(hours=1)
    result["observation_age_at_generation_hours"] = (
        result.latest_pm25_age_hours
        + (generated - asof_time) / pd.Timedelta(hours=1)
    )
    result["generation_status"] = np.where(
        result.lead_remaining_at_generation_hours.gt(0),
        "prospective_target",
        "target_reached_before_completion",
    )
    fresh = (
        np.isfinite(result.observation_age_at_generation_hours)
        & result.observation_age_at_generation_hours.ge(0)
        & result.observation_age_at_generation_hours.le(6)
    )
    post_freeze = asof_time > frozen_utc
    asof_reached = asof_time <= generated
    result["prospective_evaluation_eligible"] = (
        result.lead_remaining_at_generation_hours.gt(0)
        & fresh
        & result.forecast_pm25_ug_m3.notna()
        & post_freeze
        & asof_reached
    )
    reasons = np.full(len(result), "eligible_for_candidate_shadow_evaluation", dtype=object)
    reasons[~result.forecast_pm25_ug_m3.notna().to_numpy()] = "cams_or_forecast_unavailable"
    reasons[(~fresh).to_numpy()] = "observation_stale_or_unavailable"
    reasons[result.lead_remaining_at_generation_hours.le(0).to_numpy()] = "target_reached_before_completion"
    if not post_freeze:
        reasons[:] = "engineering_replay_before_candidate_freeze"
    if not asof_reached:
        reasons[:] = "observation_asof_time_not_reached"
    result["evaluation_eligibility_reason"] = reasons
    result["duty_service_eligible"] = False
    result["public_service_eligible"] = False
    result["service_eligibility_reason"] = "candidate_requires_post_freeze_acceptance"
    if len(result) != len(registry) * len(expected_mapping):
        raise ValueError("Candidate output row coverage is incomplete")
    if (
        not np.isfinite(result.forecast_pm25_ug_m3).all()
        or result.forecast_pm25_ug_m3.lt(0).any()
        or not np.isfinite(result.prediction_lower_ug_m3).all()
        or not np.isfinite(result.prediction_upper_ug_m3).all()
        or not result.prediction_lower_ug_m3.le(result.forecast_pm25_ug_m3).all()
        or not result.forecast_pm25_ug_m3.le(result.prediction_upper_ug_m3).all()
    ):
        raise ValueError("Candidate output contains invalid point or interval values")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    result.to_csv(temporary, index=False)
    temporary.replace(output_path)
    generation_completed = pd.Timestamp.now(tz="UTC")
    metadata = {
        "generated_utc": generated.isoformat(),
        "generation_completed_utc": generation_completed.isoformat(),
        "generation_timestamp_semantics": "forecast_write_completion",
        "candidate_version": point_manifest["candidate_version"],
        "interval_version": interval_manifest["interval_version"],
        "candidate_manifest_sha256": file_sha256(point_manifest_path),
        "interval_manifest_sha256": file_sha256(interval_manifest_path),
        "base_config_sha256": file_sha256(paths.config),
        "candidate_config_sha256": file_sha256(candidate_config_path),
        "candidate_freeze_utc": frozen_utc.isoformat(),
        "rows": len(result),
        "available_rows": int(result.forecast_pm25_ug_m3.notna().sum()),
        "output": str(output_path.relative_to(paths.root)),
        "output_sha256": file_sha256(output_path),
        "elapsed_seconds": time.perf_counter() - started,
        "promotion_authorized": False,
        "prospective_evaluation_eligible_rows": int(
            result.prospective_evaluation_eligible.sum()
        ),
    }
    write_json(output_path.with_suffix(".json"), metadata)
    return result, metadata


def _station_balanced(frame: pd.DataFrame, prediction: str) -> dict[str, float]:
    per_station = frame.groupby("station_code", observed=True).apply(
        lambda group: pd.Series(
            {
                "n": len(group),
                "model_mae": (group[prediction] - group.target_pm25_ug_m3).abs().mean(),
                "persistence_mae": (group.persistence_10utc - group.target_pm25_ug_m3).abs().mean(),
                "frozen_mae": (group.frozen_deployment - group.target_pm25_ug_m3).abs().mean(),
            }
        ),
        include_groups=False,
    )
    model_mae = float(per_station.model_mae.mean())
    persistence_mae = float(per_station.persistence_mae.mean())
    frozen_mae = float(per_station.frozen_mae.mean())
    return {
        "n": int(per_station.n.sum()),
        "stations": len(per_station),
        "minimum_rows_per_station": int(per_station.n.min()),
        "station_balanced_mae_ug_m3": model_mae,
        "persistence_station_balanced_mae_ug_m3": persistence_mae,
        "frozen_station_balanced_mae_ug_m3": frozen_mae,
        "skill_vs_10utc_persistence_pct": 100.0 * (1.0 - model_mae / persistence_mae),
        "skill_vs_frozen_pct": 100.0 * (1.0 - model_mae / frozen_mae),
        "mean_per_station_skill_vs_persistence_pct": float(
            (100.0 * (1.0 - per_station.model_mae / per_station.persistence_mae)).mean()
        ),
        "stations_harmed_vs_persistence": int((per_station.model_mae > per_station.persistence_mae).sum()),
        "stations_harmed_vs_frozen": int((per_station.model_mae > per_station.frozen_mae).sum()),
    }


def _bootstrap(
    frame: pd.DataFrame,
    prediction: str,
    baseline: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    working = frame.copy()
    working["week"] = working.target_time_utc.dt.strftime("%G-W%V")
    weeks = working.week.unique()
    stations = working.station_code.unique()
    weekly = (
        working.assign(
            model_error=(working[prediction] - working.target_pm25_ug_m3).abs(),
            baseline_error=(working[baseline] - working.target_pm25_ug_m3).abs(),
        )
        .groupby(["week", "station_code"], observed=True)
        .agg(model_sum=("model_error", "sum"), baseline_sum=("baseline_error", "sum"), n=("model_error", "size"))
    )
    index = pd.MultiIndex.from_product([weeks, stations], names=["week", "station_code"])
    weekly = weekly.reindex(index, fill_value=0)
    model_sums = weekly.model_sum.to_numpy().reshape(len(weeks), len(stations))
    baseline_sums = weekly.baseline_sum.to_numpy().reshape(len(weeks), len(stations))
    counts = weekly.n.to_numpy().reshape(len(weeks), len(stations))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        weights = rng.multinomial(len(weeks), np.full(len(weeks), 1.0 / len(weeks)))
        sampled_counts = weights @ counts
        valid = sampled_counts > 0
        sampled_model = (weights @ model_sums)[valid] / sampled_counts[valid]
        sampled_baseline = (weights @ baseline_sums)[valid] / sampled_counts[valid]
        values.append(float(np.mean(sampled_baseline - sampled_model)))
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "improvement_ug_m3": float(np.mean(values)),
        "ci95_lower_ug_m3": float(lower),
        "ci95_upper_ug_m3": float(upper),
        "calendar_weeks": len(weeks),
        "bootstrap_replicates": replicates,
    }


def run_candidate_experiment(paths: ExperimentPaths | None = None) -> dict[str, Any]:
    paths = paths or ExperimentPaths()
    started = time.perf_counter()
    base_config = load_config(paths.config)
    candidate = _candidate_config(paths)
    manifest_path = paths.provenance / "candidate_availability_manifest.json"
    candidate_dir = paths.root / "models" / "candidate" / candidate["candidate_version"]
    if manifest_path.exists() or candidate_dir.exists():
        raise FileExistsError(
            "Candidate version already exists; choose a new candidate_version instead "
            "of overwriting frozen development evidence"
        )
    table00 = load_modeling_table(paths)
    table10, asof_audit = _asof_10_table(paths, base_config, candidate)
    all_columns = feature_columns(table00, include_cams=True)
    no_met_columns = _no_met_columns(all_columns)
    manifest, frozen_bundle_id, frozen_paths = _validated_deployment_bundle(paths, base_config)
    frozen_meta = {
        int(row["forecast_hour"]): row
        for row in manifest["models"]
        if row["role"] == "primary_point"
    }
    frozen_models = {
        horizon: joblib.load(frozen_paths[("primary_point", horizon)])
        for horizon in map(int, candidate["lead_mapping_hours"])
    }

    predictions: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    selected_iterations: dict[int, list[int]] = {
        int(horizon): [] for horizon in candidate["lead_mapping_hours"]
    }
    training_start = base_config["splits"]["training_target_start_utc"]
    for fold in candidate["folds"]:
        fold_number = int(fold["fold"])
        for horizon_text, remaining in candidate["lead_mapping_hours"].items():
            horizon = int(horizon_text)
            remaining = int(remaining)
            source00 = table00.loc[table00.forecast_hour.eq(horizon)].copy()
            source10 = table10.loc[table10.forecast_hour.eq(horizon)].copy()
            fit00 = _period(source00, training_start, fold["fit_end"])
            tune00 = _period(source00, fold["tune_start"], fold["tune_end"])
            calibration00 = _period(source00, fold["calibration_start"], fold["calibration_end"])
            assessment00 = _period(source00, fold["assessment_start"], fold["assessment_end"])
            fit10 = _period(source10, training_start, fold["fit_end"])
            tune10 = _period(source10, fold["tune_start"], fold["tune_end"])
            calibration10 = _period(source10, fold["calibration_start"], fold["calibration_end"])
            assessment10 = _period(source10, fold["assessment_start"], fold["assessment_end"])
            if any(frame.empty for frame in (fit00, tune00, calibration00, assessment00, fit10, tune10, calibration10, assessment10)):
                raise ValueError(f"Empty chronological block for fold={fold_number}, lead={horizon}")

            key = ["station_code", "target_time_utc", "forecast_hour"]
            common_keys = (
                assessment10[key]
                .merge(assessment00[key], on=key, how="inner", validate="one_to_one")
                .sort_values(key)
                .reset_index(drop=True)
            )
            common_index = pd.MultiIndex.from_frame(common_keys)
            assessment10 = assessment10.set_index(key).loc[common_index].reset_index()
            assessment00 = assessment00.set_index(key).loc[common_index].reset_index()
            assessment = assessment10
            if not assessment10.target_time_utc.equals(assessment00.target_time_utc):
                raise ValueError("Aligned +00 and +10 UTC assessment targets differ")

            fit_started = time.perf_counter()
            all_pred, _, _, all_iteration = _fit_refit_predict(
                "all_features_00utc_expanding", horizon, all_columns,
                fit00, tune00, calibration00, assessment00, base_config,
            )
            all_missing_pred, _, _, _ = _fit_refit_predict(
                "all_features_00utc_expanding", horizon, all_columns,
                fit00, tune00, calibration00, assessment00, base_config,
                missing_met_stress=True,
            )
            no00_pred, _, _, no00_iteration = _fit_refit_predict(
                "no_met_00utc_expanding", horizon, no_met_columns,
                fit00, tune00, calibration00, assessment00, base_config,
            )
            no10_pred, no10_calibration_pred, _, no10_iteration = _fit_refit_predict(
                "no_met_10utc_expanding", horizon, no_met_columns,
                fit10, tune10, calibration10, assessment10, base_config,
            )
            selected_iterations[horizon].append(no10_iteration)
            runtime_rows.append(
                {
                    "fold": fold_number,
                    "forecast_hour": horizon,
                    "fit_seconds_all_variants": time.perf_counter() - fit_started,
                    "all_features_best_iteration": all_iteration,
                    "no_met_00_best_iteration": no00_iteration,
                    "no_met_10_best_iteration": no10_iteration,
                    "fit_rows_10utc": len(fit10),
                    "tune_rows_10utc": len(tune10),
                    "calibration_rows_10utc": len(calibration10),
                    "assessment_rows_common": len(assessment),
                }
            )
            frozen_columns = frozen_meta[horizon]["source_columns"]
            frozen_normal = _frozen_predict(
                assessment00, frozen_models[horizon], frozen_columns, False
            )
            frozen_missing = _frozen_predict(
                assessment00, frozen_models[horizon], frozen_columns, True
            )
            climatology = _climatology_predictor(
                pd.concat([fit10, tune10], ignore_index=True), [assessment10]
            )[0]
            calibration_errors = np.abs(
                no10_calibration_pred - calibration10.target_pm25_ug_m3.to_numpy(float)
            )
            interval_half_width = _conformal_quantile(
                calibration_errors,
                coverage=float(candidate["nominal_interval_coverage"]),
            )
            threshold = (
                pd.concat([fit10, tune10])
                .groupby("station_code", observed=True)
                .target_pm25_ug_m3.quantile(0.9)
            )
            output = assessment[key + [
                "station_name", "region", "remaining_horizon_hours",
                "asof_time_utc", "target_pm25_ug_m3", "pm25_lag_0h",
                "cams_pm25_ug_m3", "latest_pm25_age_hours",
            ]].copy()
            output["fold"] = fold_number
            output["persistence_10utc"] = output.pm25_lag_0h
            output["persistence_00utc"] = assessment00.pm25_lag_0h.to_numpy()
            output["climatology_training_only"] = climatology
            output["raw_cams"] = output.cams_pm25_ug_m3
            output["frozen_deployment"] = frozen_normal
            output["frozen_deployment_missing_met"] = frozen_missing
            output["all_features_00utc_expanding"] = all_pred
            output["all_features_00utc_missing_met"] = all_missing_pred
            output["no_met_00utc_expanding"] = no00_pred
            output["no_met_10utc_expanding"] = no10_pred
            output["candidate_interval_lower"] = np.maximum(no10_pred - interval_half_width, 0)
            output["candidate_interval_upper"] = no10_pred + interval_half_width
            output["candidate_interval_half_width_ug_m3"] = interval_half_width
            output["high_event_training_p90"] = output.apply(
                lambda row: row.target_pm25_ug_m3 >= threshold.get(row.station_code, np.inf),
                axis=1,
            )
            predictions.append(output)

    result = pd.concat(predictions, ignore_index=True)
    model_columns = [
        "persistence_10utc", "persistence_00utc", "climatology_training_only",
        "raw_cams", "frozen_deployment", "frozen_deployment_missing_met",
        "all_features_00utc_expanding", "all_features_00utc_missing_met",
        "no_met_00utc_expanding", "no_met_10utc_expanding",
    ]
    common = result.target_pm25_ug_m3.notna()
    for column in model_columns:
        common &= result[column].notna() & np.isfinite(result[column])
    result["common_case"] = common
    common_result = result.loc[common].copy()

    summary_rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []
    for (horizon, event_scope), group in pd.concat(
        [
            common_result.assign(event_scope="all_common_cases"),
            common_result.loc[common_result.high_event_training_p90].assign(event_scope="training_p90_high_events"),
        ]
    ).groupby(["forecast_hour", "event_scope"], observed=True):
        for model_name in model_columns:
            values = _station_balanced(group, model_name)
            values.update(
                {"forecast_hour": int(horizon), "event_scope": event_scope, "model": model_name}
            )
            if model_name == "no_met_10utc_expanding":
                covered = group.target_pm25_ug_m3.between(
                    group.candidate_interval_lower, group.candidate_interval_upper
                )
                values["interval_coverage_pct"] = 100.0 * float(covered.mean())
            else:
                values["interval_coverage_pct"] = np.nan
            summary_rows.append(values)
            for station_code, station in group.groupby("station_code", observed=True):
                station_rows.append(
                    {
                        "forecast_hour": int(horizon),
                        "event_scope": event_scope,
                        "model": model_name,
                        "station_code": station_code,
                        "n": len(station),
                        "mae_ug_m3": float((station[model_name] - station.target_pm25_ug_m3).abs().mean()),
                        "persistence_mae_ug_m3": float((station.persistence_10utc - station.target_pm25_ug_m3).abs().mean()),
                        "frozen_mae_ug_m3": float((station.frozen_deployment - station.target_pm25_ug_m3).abs().mean()),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    station_summary = pd.DataFrame(station_rows)
    bootstrap_rows: list[dict[str, Any]] = []
    for horizon, group in common_result.groupby("forecast_hour", observed=True):
        for baseline in ("persistence_10utc", "frozen_deployment"):
            values = _bootstrap(
                group, "no_met_10utc_expanding", baseline,
                int(candidate["bootstrap_replicates"]),
                int(candidate["random_seed"]) + int(horizon),
            )
            values.update({"forecast_hour": int(horizon), "candidate": "no_met_10utc_expanding", "baseline": baseline})
            bootstrap_rows.append(values)
    bootstrap = pd.DataFrame(bootstrap_rows)

    # Freeze a separately versioned future candidate; July-August remains calibration only.
    candidate_dir.mkdir(parents=True, exist_ok=True)
    model_manifest_rows: list[dict[str, Any]] = []
    for horizon_text in candidate["lead_mapping_hours"]:
        horizon = int(horizon_text)
        source = table10.loc[table10.forecast_hour.eq(horizon)].copy()
        fit = _period(source, training_start, candidate["final_fit_target_end"])
        calibration = _period(
            source, candidate["final_calibration_start"], candidate["final_calibration_end"]
        )
        iteration = int(np.median(selected_iterations[horizon]))
        model = _fit_lightgbm(
            candidate["candidate_version"], horizon, "no_met_10utc",
            no_met_columns, fit, fit.iloc[0:0], base_config,
            n_estimators=iteration, use_early_stopping=False,
        )
        calibration_prediction = predict_model(model, calibration, no_met_columns)
        correction = _conformal_quantile(
            np.abs(calibration_prediction - calibration.target_pm25_ug_m3.to_numpy(float)),
            coverage=float(candidate["nominal_interval_coverage"]),
        )
        path = candidate_dir / f"candidate_{horizon:03d}h.joblib"
        joblib.dump(model, path, compress=3)
        model_manifest_rows.append(
            {
                "forecast_hour": horizon,
                "remaining_horizon_hours": int(candidate["lead_mapping_hours"][horizon_text]),
                "path": str(path.relative_to(paths.root)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "source_columns": no_met_columns,
                "encoded_columns": model.feature_columns,
                "training_rows": len(fit),
                "training_target_end_utc": candidate["final_fit_target_end"],
                "fixed_iterations": iteration,
                "calibration_rows": len(calibration),
                "calibration_period": [candidate["final_calibration_start"], candidate["final_calibration_end"]],
                "symmetric_interval_half_width_ug_m3": correction,
            }
        )

    paths.tables.mkdir(parents=True, exist_ok=True)
    paths.provenance.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "predictions": paths.derived / "candidate_availability_predictions.csv.gz",
        "metrics": paths.tables / "candidate_availability_metrics.csv",
        "station_metrics": paths.tables / "candidate_availability_metrics_by_station.csv",
        "week_bootstrap": paths.tables / "candidate_availability_week_bootstrap.csv",
        "runtime": paths.tables / "candidate_availability_runtime.csv",
    }
    result.to_csv(output_paths["predictions"], index=False, compression="gzip")
    summary.to_csv(output_paths["metrics"], index=False)
    station_summary.to_csv(output_paths["station_metrics"], index=False)
    bootstrap.to_csv(output_paths["week_bootstrap"], index=False)
    pd.DataFrame(runtime_rows).to_csv(output_paths["runtime"], index=False)
    elapsed = time.perf_counter() - started
    provenance = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_version": candidate["candidate_version"],
        "development_only": True,
        "promotion_authorized": False,
        "independent_acceptance_start": candidate["independent_acceptance_start"],
        "candidate_config_sha256": file_sha256(paths.root / "candidate_config.json"),
        "base_config_sha256": file_sha256(paths.config),
        "frozen_deployment_bundle_id": frozen_bundle_id,
        "asof_10utc_audit": asof_audit,
        "folds": candidate["folds"],
        "models": model_manifest_rows,
        "artifacts": {
            name: {
                "path": str(path.relative_to(paths.root)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in output_paths.items()
        },
        "prediction_rows": len(result),
        "common_prediction_rows": int(result.common_case.sum()),
        "runtime_seconds": elapsed,
        "maximum_resident_set_size_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "future_inference_contract": {
            "cams_initialization_hour_utc": candidate["cams_initialization_hour_utc"],
            "observation_asof_hour_utc": candidate["observation_asof_hour_utc"],
            "lead_mapping_hours": candidate["lead_mapping_hours"],
            "meteorology_predictors_required": False,
            "scheduler_changed": False,
        },
        "limitations": [
            "January-August 2026 was reused for development and is not independent acceptance evidence.",
            "The historical archive reconstructs 10 UTC features from final files, not first-arrival snapshots.",
            "Future promotion requires post-freeze prospective evidence under the predeclared operational gate.",
        ],
    }
    write_json(manifest_path, provenance)
    return provenance


def run_interval_scale_correction(
    paths: ExperimentPaths | None = None,
) -> dict[str, Any]:
    """Run the single predeclared concentration-scaled interval correction."""

    paths = paths or ExperimentPaths()
    started = time.perf_counter()
    base_config = load_config(paths.config)
    candidate = _candidate_config(paths)
    correction_config_path = paths.root / "candidate_interval_v2_config.json"
    correction_config = json.loads(correction_config_path.read_text(encoding="utf-8"))
    manifest_path = paths.provenance / "candidate_interval_v2_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Interval-v2 evidence already exists and is immutable")
    table10, audit = _asof_10_table(paths, base_config, candidate)
    table00 = load_modeling_table(paths)
    no_met_columns = _no_met_columns(feature_columns(table00, include_cams=True))
    prediction_path = paths.derived / "candidate_availability_predictions.csv.gz"
    initial = pd.read_csv(
        prediction_path,
        parse_dates=["asof_time_utc", "target_time_utc"],
        low_memory=False,
    )
    pieces: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, Any]] = []
    training_start = base_config["splits"]["training_target_start_utc"]
    for fold in candidate["folds"]:
        fold_number = int(fold["fold"])
        for horizon_text in candidate["lead_mapping_hours"]:
            horizon = int(horizon_text)
            source = table10.loc[table10.forecast_hour.eq(horizon)].copy()
            fit = _period(source, training_start, fold["fit_end"])
            tune = _period(source, fold["tune_start"], fold["tune_end"])
            calibration = _period(
                source, fold["calibration_start"], fold["calibration_end"]
            )
            assessment = _period(
                source, fold["assessment_start"], fold["assessment_end"]
            ).sort_values(["station_code", "target_time_utc"])
            assessment_saved = initial.loc[
                initial.fold.eq(fold_number) & initial.forecast_hour.eq(horizon)
            ].sort_values(["station_code", "target_time_utc"])
            common_keys = ["station_code", "target_time_utc", "forecast_hour"]
            assessment = assessment.set_index(common_keys).loc[
                pd.MultiIndex.from_frame(assessment_saved[common_keys])
            ].reset_index()
            predicted, calibration_predicted, _, iteration = _fit_refit_predict(
                "no_met_10utc_interval_v2",
                horizon,
                no_met_columns,
                fit,
                tune,
                calibration,
                assessment,
                base_config,
            )
            if not np.allclose(
                predicted,
                assessment_saved.no_met_10utc_expanding,
                rtol=0,
                atol=1.0e-6,
            ):
                raise ValueError("Interval correction did not reproduce v1 point predictions")
            calibration_scale = np.maximum.reduce(
                [
                    np.full(len(calibration), 10.0),
                    np.maximum(calibration_predicted, 0.0),
                    calibration.pm25_lag_0h.fillna(0.0).clip(lower=0).to_numpy(float),
                ]
            )
            coefficient = _conformal_quantile(
                np.abs(
                    calibration_predicted
                    - calibration.target_pm25_ug_m3.to_numpy(float)
                )
                / calibration_scale,
                coverage=float(correction_config["nominal_coverage"]),
            )
            scale = np.maximum.reduce(
                [
                    np.full(len(assessment_saved), 10.0),
                    np.maximum(
                        assessment_saved.no_met_10utc_expanding.to_numpy(float), 0.0
                    ),
                    assessment_saved.pm25_lag_0h.fillna(0.0).clip(lower=0).to_numpy(float),
                ]
            )
            output = assessment_saved[
                [
                    "station_code",
                    "target_time_utc",
                    "forecast_hour",
                    "remaining_horizon_hours",
                    "fold",
                    "target_pm25_ug_m3",
                    "pm25_lag_0h",
                    "no_met_10utc_expanding",
                    "candidate_interval_lower",
                    "candidate_interval_upper",
                    "high_event_training_p90",
                    "common_case",
                ]
            ].copy()
            output["interval_scale_ug_m3"] = scale
            output["adaptive_interval_coefficient"] = coefficient
            output["adaptive_interval_lower"] = np.maximum(
                output.no_met_10utc_expanding - coefficient * scale, 0.0
            )
            output["adaptive_interval_upper"] = (
                output.no_met_10utc_expanding + coefficient * scale
            )
            pieces.append(output)
            coefficient_rows.append(
                {
                    "fold": fold_number,
                    "forecast_hour": horizon,
                    "calibration_rows": len(calibration),
                    "best_iteration": iteration,
                    "adaptive_interval_coefficient": coefficient,
                }
            )
    result = pd.concat(pieces, ignore_index=True)
    alpha = 1.0 - float(correction_config["nominal_coverage"])
    metrics_rows: list[dict[str, Any]] = []
    for (horizon, event_scope), group in pd.concat(
        [
            result.loc[result.common_case].assign(event_scope="all_common_cases"),
            result.loc[result.common_case & result.high_event_training_p90].assign(
                event_scope="training_p90_high_events"
            ),
        ]
    ).groupby(["forecast_hour", "event_scope"], observed=True):
        for version, lower_name, upper_name in (
            ("symmetric_v1", "candidate_interval_lower", "candidate_interval_upper"),
            ("input_scaled_v2", "adaptive_interval_lower", "adaptive_interval_upper"),
        ):
            lower = group[lower_name]
            upper = group[upper_name]
            target = group.target_pm25_ug_m3
            covered = target.between(lower, upper)
            width = upper - lower
            interval_score = width + (2.0 / alpha) * (
                (lower - target).clip(lower=0) + (target - upper).clip(lower=0)
            )
            metrics_rows.append(
                {
                    "forecast_hour": int(horizon),
                    "remaining_horizon_hours": int(group.remaining_horizon_hours.iloc[0]),
                    "event_scope": event_scope,
                    "interval_version": version,
                    "n": len(group),
                    "coverage_pct": 100.0 * float(covered.mean()),
                    "mean_width_ug_m3": float(width.mean()),
                    "mean_interval_score_ug_m3": float(interval_score.mean()),
                }
            )
    metrics = pd.DataFrame(metrics_rows)

    initial_manifest = json.loads(
        (paths.provenance / "candidate_availability_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    final_rows: list[dict[str, Any]] = []
    for model_meta in initial_manifest["models"]:
        horizon = int(model_meta["forecast_hour"])
        model_path = paths.root / model_meta["path"]
        if model_path.stat().st_size != model_meta["bytes"] or file_sha256(model_path) != model_meta["sha256"]:
            raise ValueError("Frozen candidate model checksum mismatch")
        model: TrainedModel = joblib.load(model_path)
        source = table10.loc[table10.forecast_hour.eq(horizon)].copy()
        calibration = _period(
            source,
            candidate["final_calibration_start"],
            candidate["final_calibration_end"],
        )
        predicted = predict_model(model, calibration, model_meta["source_columns"])
        scale = np.maximum.reduce(
            [
                np.full(len(calibration), 10.0),
                np.maximum(predicted, 0.0),
                calibration.pm25_lag_0h.fillna(0.0).clip(lower=0).to_numpy(float),
            ]
        )
        coefficient = _conformal_quantile(
            np.abs(predicted - calibration.target_pm25_ug_m3.to_numpy(float)) / scale,
            coverage=float(correction_config["nominal_coverage"]),
        )
        final_rows.append(
            {
                "forecast_hour": horizon,
                "remaining_horizon_hours": model_meta["remaining_horizon_hours"],
                "point_model_sha256": model_meta["sha256"],
                "calibration_rows": len(calibration),
                "adaptive_interval_coefficient": coefficient,
            }
        )

    output_path = paths.derived / "candidate_interval_v2_predictions.csv.gz"
    metrics_path = paths.tables / "candidate_interval_comparison.csv"
    coefficient_path = paths.tables / "candidate_interval_v2_coefficients.csv"
    result.to_csv(output_path, index=False, compression="gzip")
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(coefficient_rows).to_csv(coefficient_path, index=False)
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "interval_version": correction_config["interval_version"],
        "point_candidate_version": correction_config["point_candidate_version"],
        "development_only": True,
        "promotion_authorized": False,
        "correction_config_sha256": file_sha256(correction_config_path),
        "candidate_config_sha256": file_sha256(paths.root / "candidate_config.json"),
        "base_config_sha256": file_sha256(paths.config),
        "point_manifest_sha256": file_sha256(
            paths.provenance / "candidate_availability_manifest.json"
        ),
        "v1_predictions_sha256": file_sha256(prediction_path),
        "asof_10utc_audit": audit,
        "fold_coefficients": coefficient_rows,
        "future_coefficients": final_rows,
        "runtime_seconds": time.perf_counter() - started,
        "maximum_resident_set_size_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "artifacts": {
            name: {
                "path": str(path.relative_to(paths.root)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in {
                "predictions": output_path,
                "metrics": metrics_path,
                "coefficients": coefficient_path,
            }.items()
        },
        "limitations": [
            "This is the single predeclared adaptive interval correction motivated by v1 high-event undercoverage.",
            "It changes uncertainty scaling only; point forecasts and station-level harms are unchanged.",
            "January-August 2026 remains development evidence rather than independent acceptance evidence.",
        ],
    }
    write_json(manifest_path, manifest)
    return manifest
