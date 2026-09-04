"""Deployment refit, serialization, and batch operational inference."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data import ExperimentPaths, file_sha256, load_config, write_json
from .modeling import (
    TrainedModel,
    _conformal_quantile,
    _fit_lightgbm,
    _fit_xgboost,
    _interval_period_mask,
    feature_columns,
    load_modeling_table,
    predict_model,
)


def _research_model_lookup(manifest: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row["model"]), int(row["forecast_hour"])): row
        for row in manifest["models"]
    }


def _fit_fixed_point_model(
    model_name: str,
    horizon: int,
    source_columns: list[str],
    training: pd.DataFrame,
    best_iteration: int,
    config: dict[str, Any],
) -> TrainedModel:
    empty = training.iloc[0:0]
    if model_name == "cams_xgboost":
        return _fit_xgboost(
            horizon,
            source_columns,
            training,
            empty,
            config,
            n_estimators=best_iteration,
            use_early_stopping=False,
        )
    return _fit_lightgbm(
        model_name=f"deployment_{model_name}",
        forecast_hour=horizon,
        feature_variant=("cams" if model_name != "obs_lgbm" else "observation_only"),
        source_columns=source_columns,
        train=training,
        validation=empty,
        config=config,
        n_estimators=best_iteration,
        use_early_stopping=False,
    )


def _save_model(
    model: TrainedModel,
    output: Path,
    role: str,
    source_columns: list[str],
) -> dict[str, Any]:
    joblib.dump(model, output, compress=3)
    return {
        "role": role,
        "model_name": model.model_name,
        "forecast_hour": model.forecast_hour,
        "path": str(output.relative_to(output.parents[2])),
        "bytes": output.stat().st_size,
        "sha256": file_sha256(output),
        "best_iteration": model.best_iteration,
        "training_rows": model.training_rows,
        "source_columns": source_columns,
        "encoded_columns": model.feature_columns,
    }


def build_deployment_models(paths: ExperimentPaths | None = None) -> dict[str, Any]:
    """Refit frozen model specifications on training plus validation only."""

    paths = paths or ExperimentPaths()
    config = load_config(paths.config)
    modeling = load_modeling_table(paths)
    research_manifest_path = paths.provenance / "research_model_manifest.json"
    research_manifest = json.loads(research_manifest_path.read_text(encoding="utf-8"))
    lookup = _research_model_lookup(research_manifest)
    champion = str(research_manifest["champion"])
    include_cams = champion != "obs_lgbm"
    primary_columns = feature_columns(modeling, include_cams=include_cams)
    fallback_columns = feature_columns(modeling, include_cams=False)
    deployment_dir = paths.root / "models" / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)

    model_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    fitted_models: list[tuple[str, TrainedModel, list[str]]] = []
    deployment_corrections: dict[int, float] = {}
    calibration_rows_by_horizon: dict[int, int] = {}
    interval_config = config["prediction_intervals"]
    for horizon in config["forecast_hours"]:
        point_training = modeling.loc[
            modeling.forecast_hour.eq(horizon)
            & modeling.split.isin(["train", "validation"])
            & modeling.target_pm25_ug_m3.notna()
        ].copy()
        primary_training = point_training
        if include_cams:
            primary_training = primary_training.loc[
                primary_training.cams_pm25_ug_m3.notna()
            ]
        champion_iteration = int(lookup[(champion, int(horizon))]["best_iteration"])
        primary = _fit_fixed_point_model(
            champion,
            int(horizon),
            primary_columns,
            primary_training,
            champion_iteration,
            config,
        )
        primary_path = deployment_dir / f"primary_{int(horizon):03d}h.joblib"
        model_rows.append(
            _save_model(primary, primary_path, "primary_point", primary_columns)
        )
        timing_rows.append(
            {
                "stage": "deployment_refit",
                "role": "primary_point",
                "forecast_hour": int(horizon),
                "seconds": primary.training_seconds,
                "training_rows": primary.training_rows,
            }
        )
        fitted_models.append(("primary_point", primary, primary_columns))

        fallback_iteration = int(
            lookup[("obs_lgbm", int(horizon))]["best_iteration"]
        )
        fallback = _fit_fixed_point_model(
            "obs_lgbm",
            int(horizon),
            fallback_columns,
            point_training,
            fallback_iteration,
            config,
        )
        fallback_path = deployment_dir / f"fallback_obs_{int(horizon):03d}h.joblib"
        model_rows.append(
            _save_model(
                fallback, fallback_path, "observation_only_fallback", fallback_columns
            )
        )
        timing_rows.append(
            {
                "stage": "deployment_refit",
                "role": "observation_only_fallback",
                "forecast_hour": int(horizon),
                "seconds": fallback.training_seconds,
                "training_rows": fallback.training_rows,
            }
        )
        fitted_models.append(("observation_only_fallback", fallback, fallback_columns))

        interval_fit = modeling.loc[
            modeling.forecast_hour.eq(horizon)
            & modeling.target_pm25_ug_m3.notna()
            & (
                modeling.split.eq("train")
                | (
                    modeling.split.eq("validation")
                    & _interval_period_mask(modeling, config, "tuning")
                )
            )
        ].copy()
        interval_calibration = modeling.loc[
            modeling.forecast_hour.eq(horizon)
            & modeling.target_pm25_ug_m3.notna()
            & modeling.split.eq("validation")
            & _interval_period_mask(modeling, config, "calibration")
        ].copy()
        if include_cams:
            interval_fit = interval_fit.loc[interval_fit.cams_pm25_ug_m3.notna()]
            interval_calibration = interval_calibration.loc[
                interval_calibration.cams_pm25_ug_m3.notna()
            ]
        if interval_fit.empty or interval_calibration.empty:
            raise ValueError(
                f"Deployment interval fit or calibration is empty for horizon {horizon}"
            )
        quantile_calibration_predictions: dict[float, np.ndarray] = {}
        for quantile in config["quantiles"]:
            research_name = f"quantile_{float(quantile):.1f}"
            iteration = int(lookup[(research_name, int(horizon))]["best_iteration"])
            quantile_model = _fit_lightgbm(
                model_name=f"deployment_{research_name}",
                forecast_hour=int(horizon),
                feature_variant="cams" if include_cams else "observation_only",
                source_columns=primary_columns,
                train=interval_fit,
                validation=interval_fit.iloc[0:0],
                config=config,
                objective="quantile",
                alpha=float(quantile),
                n_estimators=iteration,
                use_early_stopping=False,
            )
            role = f"quantile_{int(float(quantile) * 100):02d}"
            output = deployment_dir / f"{role}_{int(horizon):03d}h.joblib"
            model_rows.append(
                _save_model(quantile_model, output, role, primary_columns)
            )
            timing_rows.append(
                {
                    "stage": "deployment_refit",
                    "role": role,
                    "forecast_hour": int(horizon),
                    "seconds": quantile_model.training_seconds,
                    "training_rows": quantile_model.training_rows,
                }
            )
            fitted_models.append((role, quantile_model, primary_columns))
            quantile_calibration_predictions[float(quantile)] = predict_model(
                quantile_model, interval_calibration, primary_columns
            )

        ordered_calibration = np.sort(
            np.column_stack(
                [
                    quantile_calibration_predictions[0.1],
                    quantile_calibration_predictions[0.5],
                    quantile_calibration_predictions[0.9],
                ]
            ),
            axis=1,
        )
        calibration_target = interval_calibration.target_pm25_ug_m3.to_numpy(float)
        calibration_scores = np.maximum(
            ordered_calibration[:, 0] - calibration_target,
            calibration_target - ordered_calibration[:, 2],
        )
        deployment_corrections[int(horizon)] = _conformal_quantile(
            calibration_scores,
            coverage=float(interval_config["nominal_coverage"]),
        )
        calibration_rows_by_horizon[int(horizon)] = len(interval_calibration)

    # Benchmark warm prediction on one daily national batch per lead.
    benchmark_rows: list[dict[str, Any]] = []
    repetitions = 30
    for role, model, columns in fitted_models:
        sample = (
            modeling.loc[
                modeling.forecast_hour.eq(model.forecast_hour)
                & modeling.split.eq("test")
            ]
            .sort_values("issue_time_utc")
            .groupby("station_code", observed=True)
            .tail(1)
        )
        elapsed = []
        for _ in range(repetitions):
            started = time.perf_counter()
            predict_model(model, sample, columns)
            elapsed.append(time.perf_counter() - started)
        benchmark_rows.append(
            {
                "stage": "warm_batch_inference",
                "role": role,
                "forecast_hour": model.forecast_hour,
                "rows_per_batch": len(sample),
                "repetitions": repetitions,
                "median_seconds_per_batch": float(np.median(elapsed)),
                "p95_seconds_per_batch": float(np.quantile(elapsed, 0.95)),
            }
        )

    timing = pd.concat(
        [pd.DataFrame(timing_rows), pd.DataFrame(benchmark_rows)],
        ignore_index=True,
        sort=False,
    )
    timing.to_csv(paths.tables / "deployment_runtime_benchmark.csv", index=False)
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": config["experiment_name"],
        "research_manifest_sha256": file_sha256(research_manifest_path),
        "config_sha256": file_sha256(paths.config),
        "research_champion": champion,
        "primary_requires_cams_pm25": include_cams,
        "historically_validated_cycle_hours_utc": config[
            "forecast_cycle_hours_utc"
        ],
        "forecast_hours": config["forecast_hours"],
        "deployment_training_target_period": {
            "start_utc": config["splits"]["training_target_start_utc"],
            "end_utc": config["splits"]["validation_target_end_utc"],
            "excluded_test_period": [
                config["splits"]["test_target_start_utc"],
                config["splits"]["test_target_end_utc"],
            ],
        },
        "interval_calibration": {
            "nominal_coverage_pct": 100.0
            * float(interval_config["nominal_coverage"]),
            "method": "lead-specific split-conformal expansion pooled across stations",
            "quantile_fit_target_period": {
                "start_utc": config["splits"]["training_target_start_utc"],
                "end_utc": interval_config["quantile_tuning_target_end_utc"],
            },
            "conformal_calibration_target_period": {
                "start_utc": interval_config[
                    "conformal_calibration_target_start_utc"
                ],
                "end_utc": interval_config[
                    "conformal_calibration_target_end_utc"
                ],
            },
            "calibration_rows_by_forecast_hour": {
                str(horizon): rows
                for horizon, rows in calibration_rows_by_horizon.items()
            },
            "correction_ug_m3": deployment_corrections,
            "note": (
                "Quantile models exclude the July-December 2025 calibration block; "
                "corrections are recomputed from that held block after refitting. "
                "Monitor empirical coverage before live use."
            ),
        },
        "models": model_rows,
    }
    write_json(paths.provenance / "deployment_model_manifest.json", manifest)
    return {
        "champion": champion,
        "models": len(model_rows),
        "deployment_bytes": int(sum(row["bytes"] for row in model_rows)),
        "refit_seconds": float(pd.DataFrame(timing_rows).seconds.sum()),
    }


def _build_inference_rows(
    issue_features: pd.DataFrame,
    cams: pd.DataFrame,
    issue_time: pd.Timestamp,
    config: dict[str, Any],
) -> pd.DataFrame:
    base = issue_features.loc[issue_features.timestamp_utc.eq(issue_time)].copy()
    if base.empty:
        raise ValueError(f"No issue-time observation features at {issue_time.isoformat()}")
    frames: list[pd.DataFrame] = []
    for horizon in config["forecast_hours"]:
        frame = base.copy()
        frame["issue_time_utc"] = frame.timestamp_utc
        frame["forecast_hour"] = int(horizon)
        frame["target_time_utc"] = issue_time + pd.Timedelta(hours=int(horizon))
        frames.append(frame)
    inference = pd.concat(frames, ignore_index=True)
    cams_subset = cams.loc[cams.issue_time_utc.eq(issue_time)].copy()
    inference = inference.merge(
        cams_subset.drop(columns=["valid_time_utc"], errors="ignore"),
        on=["station_code", "issue_time_utc", "forecast_hour"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_cams"),
    )
    local_target = inference.target_time_utc + pd.to_timedelta(
        inference.utc_offset_hours, unit="h"
    )
    inference["target_hour_local"] = local_target.dt.hour
    inference["target_hour_local_sin"] = np.sin(
        2 * np.pi * local_target.dt.hour / 24.0
    )
    inference["target_hour_local_cos"] = np.cos(
        2 * np.pi * local_target.dt.hour / 24.0
    )
    target_day = local_target.dt.dayofyear
    inference["target_day_of_year_sin"] = np.sin(
        2 * np.pi * target_day / 365.25
    )
    inference["target_day_of_year_cos"] = np.cos(
        2 * np.pi * target_day / 365.25
    )
    inference["target_month"] = local_target.dt.month
    return inference


def run_operational_forecast(
    issue_time: str | None = None,
    output_path: Path | None = None,
    paths: ExperimentPaths | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run one restartable national station forecast from prepared inputs."""

    started = time.perf_counter()
    paths = paths or ExperimentPaths()
    config = load_config(paths.config)
    manifest = json.loads(
        (paths.provenance / "deployment_model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    issue_features = pd.read_csv(
        paths.derived / "issue_time_observation_features.csv.gz",
        parse_dates=["timestamp_utc"],
        low_memory=False,
    )
    issue_features["timestamp_utc"] = pd.to_datetime(
        issue_features.timestamp_utc, utc=True
    )
    cams = pd.read_csv(
        paths.derived / "cams_station_forecasts.csv.gz",
        parse_dates=["issue_time_utc", "valid_time_utc"],
        low_memory=False,
    )
    cams["issue_time_utc"] = pd.to_datetime(cams.issue_time_utc, utc=True)
    if issue_time is None:
        common_times = set(issue_features.timestamp_utc.unique()).intersection(
            cams.issue_time_utc.unique()
        )
        if not len(common_times):
            raise ValueError("Prepared observation and CAMS inputs have no common issue time")
        selected_issue = pd.Timestamp(max(common_times)).tz_convert("UTC")
    else:
        selected_issue = pd.Timestamp(issue_time)
        if selected_issue.tzinfo is None:
            raise ValueError("--issue-time must include an explicit timezone, normally Z")
        selected_issue = selected_issue.tz_convert("UTC")
    if selected_issue.hour not in config["forecast_cycle_hours_utc"]:
        raise ValueError(
            f"Issue hour {selected_issue.hour:02d} UTC was not historically validated"
        )
    frame = _build_inference_rows(
        issue_features, cams, selected_issue, config
    )
    model_lookup = {
        (str(row["role"]), int(row["forecast_hour"])): row
        for row in manifest["models"]
    }
    corrections = {
        int(key): float(value)
        for key, value in manifest["interval_calibration"][
            "correction_ug_m3"
        ].items()
    }
    outputs: list[pd.DataFrame] = []
    for horizon, horizon_frame in frame.groupby("forecast_hour", observed=True):
        horizon = int(horizon)
        primary_meta = model_lookup[("primary_point", horizon)]
        fallback_meta = model_lookup[("observation_only_fallback", horizon)]
        primary: TrainedModel = joblib.load(paths.root / primary_meta["path"])
        fallback: TrainedModel = joblib.load(paths.root / fallback_meta["path"])
        primary_prediction = predict_model(
            primary, horizon_frame, primary_meta["source_columns"]
        )
        fallback_prediction = predict_model(
            fallback, horizon_frame, fallback_meta["source_columns"]
        )
        cams_available = horizon_frame.cams_pm25_ug_m3.notna().to_numpy()
        use_primary = (
            cams_available
            if bool(manifest["primary_requires_cams_pm25"])
            else np.ones(len(horizon_frame), dtype=bool)
        )
        point = np.where(use_primary, primary_prediction, fallback_prediction)
        q_values: list[np.ndarray] = []
        for role in ("quantile_10", "quantile_50", "quantile_90"):
            meta = model_lookup[(role, horizon)]
            model: TrainedModel = joblib.load(paths.root / meta["path"])
            q_values.append(predict_model(model, horizon_frame, meta["source_columns"]))
        ordered = np.sort(np.column_stack(q_values), axis=1)
        correction = corrections[horizon]
        q10 = np.maximum(ordered[:, 0] - correction, 0.0)
        q50 = ordered[:, 1]
        q90 = ordered[:, 2] + correction
        q10[~use_primary] = np.nan
        q50[~use_primary] = np.nan
        q90[~use_primary] = np.nan
        status = np.where(use_primary, "primary", "observation_only_fallback")
        stale = horizon_frame.latest_pm25_age_hours.gt(6).fillna(True).to_numpy()
        status = np.where(stale, np.char.add(status.astype(str), "_stale_obs"), status)
        output = horizon_frame[
            [
                "station_code",
                "station_name",
                "province",
                "region",
                "timezone",
                "issue_time_utc",
                "target_time_utc",
                "forecast_hour",
                "cams_pm25_ug_m3",
                "pm25_lag_0h",
                "latest_pm25_age_hours",
            ]
        ].copy()
        output["forecast_pm25_ug_m3"] = point
        output["prediction_q10_ug_m3"] = q10
        output["prediction_q50_ug_m3"] = q50
        output["prediction_q90_ug_m3"] = q90
        output["forecast_status"] = status
        outputs.append(output)
    result = pd.concat(outputs, ignore_index=True).sort_values(
        ["station_code", "forecast_hour"]
    )
    if (result.forecast_pm25_ug_m3 < 0).any():
        raise ValueError("Operational output contains a negative PM2.5 forecast")
    valid_intervals = result.dropna(
        subset=["prediction_q10_ug_m3", "prediction_q50_ug_m3", "prediction_q90_ug_m3"]
    )
    if not (
        valid_intervals.prediction_q10_ug_m3.le(
            valid_intervals.prediction_q50_ug_m3
        )
        & valid_intervals.prediction_q50_ug_m3.le(
            valid_intervals.prediction_q90_ug_m3
        )
    ).all():
        raise ValueError("Operational quantile forecasts are not ordered")

    output_dir = paths.root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_path or output_dir / (
        f"pm25_station_forecast_{selected_issue:%Y%m%dT%H%MZ}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "issue_time_utc": selected_issue.isoformat(),
        "rows": len(result),
        "stations": int(result.station_code.nunique()),
        "forecast_hours": sorted(result.forecast_hour.unique().tolist()),
        "primary_rows": int(result.forecast_status.str.startswith("primary").sum()),
        "degraded_rows": int(
            result.forecast_status.str.startswith("observation_only_fallback").sum()
        ),
        "stale_observation_rows": int(
            result.forecast_status.str.contains("stale_obs").sum()
        ),
        "deployment_manifest_sha256": file_sha256(
            paths.provenance / "deployment_model_manifest.json"
        ),
        "output": str(output_path.relative_to(paths.root)),
        "output_sha256": file_sha256(output_path),
        "elapsed_seconds_excluding_metadata_write": time.perf_counter() - started,
    }
    write_json(output_path.with_suffix(".json"), metadata)
    return result, metadata
