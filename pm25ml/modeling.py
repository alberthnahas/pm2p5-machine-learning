"""Leakage-safe training, evaluation, uncertainty, and deployment bundles."""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold

from .data import ExperimentPaths, file_sha256, load_config, write_json


CATEGORICAL_FEATURES = ["station_code", "region", "timezone"]
STATIC_NUMERIC_FEATURES = ["latitude", "longitude", "utc_offset_hours"]
TIME_FEATURES = [
    "issue_hour_local_sin",
    "issue_hour_local_cos",
    "issue_day_of_year_sin",
    "issue_day_of_year_cos",
    "target_hour_local_sin",
    "target_hour_local_cos",
    "target_day_of_year_sin",
    "target_day_of_year_cos",
]
NETWORK_FEATURES = [
    "network_pm25_mean_ug_m3",
    "nearby_pm25_weighted_mean_ug_m3",
    "nearby_station_count",
    "latest_pm25_age_hours",
]
CAMS_FEATURES = ["cams_pm25_ug_m3"]


@dataclass
class TrainedModel:
    model_name: str
    forecast_hour: int
    feature_variant: str
    feature_columns: list[str]
    estimator: Any
    best_iteration: int
    training_rows: int
    validation_rows: int
    training_seconds: float


def load_modeling_table(paths: ExperimentPaths | None = None) -> pd.DataFrame:
    paths = paths or ExperimentPaths()
    table = pd.read_csv(
        paths.derived / "modeling_table.csv.gz",
        parse_dates=["issue_time_utc", "target_time_utc", "valid_time_utc"],
        low_memory=False,
    )
    for column in ("issue_time_utc", "target_time_utc", "valid_time_utc"):
        table[column] = pd.to_datetime(table[column], utc=True)
    return table


def feature_columns(frame: pd.DataFrame, include_cams: bool) -> list[str]:
    prefixes = (
        "pm25_lag_",
        "rh_lag_",
        "temperature_lag_",
        "pm25_mean_",
        "pm25_std_",
        "pm25_max_",
        "pm25_available_fraction_",
    )
    columns = [column for column in frame.columns if column.startswith(prefixes)]
    columns.extend(STATIC_NUMERIC_FEATURES + TIME_FEATURES + NETWORK_FEATURES)
    columns.extend(CATEGORICAL_FEATURES)
    if include_cams:
        columns.extend(CAMS_FEATURES)
    absent = [column for column in columns if column not in frame.columns]
    if absent:
        raise ValueError(f"Configured model features are absent: {absent}")
    return list(dict.fromkeys(columns))


def encode_features(
    frame: pd.DataFrame,
    source_columns: Sequence[str],
    expected_columns: Sequence[str] | None = None,
    omit_station_identity: bool = False,
) -> pd.DataFrame:
    selected = list(source_columns)
    if omit_station_identity:
        selected = [column for column in selected if column != "station_code"]
    categorical = [column for column in CATEGORICAL_FEATURES if column in selected]
    encoded = pd.get_dummies(
        frame[selected], columns=categorical, prefix=categorical, dummy_na=True, dtype=float
    )
    encoded = encoded.replace([np.inf, -np.inf], np.nan)
    if expected_columns is not None:
        encoded = encoded.reindex(columns=list(expected_columns), fill_value=0.0)
    return encoded.astype(np.float32)


def _fit_lightgbm(
    model_name: str,
    forecast_hour: int,
    feature_variant: str,
    source_columns: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: dict[str, Any],
    objective: str | None = None,
    alpha: float | None = None,
    n_estimators: int | None = None,
    omit_station_identity: bool = False,
    use_early_stopping: bool = True,
) -> TrainedModel:
    x_train = encode_features(
        train, source_columns, omit_station_identity=omit_station_identity
    )
    x_validation = (
        encode_features(
            validation,
            source_columns,
            expected_columns=x_train.columns,
            omit_station_identity=omit_station_identity,
        )
        if len(validation)
        else pd.DataFrame(index=validation.index, columns=x_train.columns, dtype=np.float32)
    )
    params = dict(config["lightgbm"])
    params["random_state"] = int(config["random_seed"])
    if objective is not None:
        params["objective"] = objective
    if alpha is not None:
        params["alpha"] = float(alpha)
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    estimator = lgb.LGBMRegressor(**params)
    callbacks = [lgb.log_evaluation(period=0)]
    fit_kwargs: dict[str, Any] = {}
    if use_early_stopping and len(validation):
        callbacks.insert(0, lgb.early_stopping(stopping_rounds=80, verbose=False))
        fit_kwargs["eval_set"] = [(x_validation, validation.target_pm25_ug_m3)]
        fit_kwargs["eval_metric"] = "quantile" if objective == "quantile" else "l1"
    started = time.perf_counter()
    estimator.fit(
        x_train,
        train.target_pm25_ug_m3,
        callbacks=callbacks,
        **fit_kwargs,
    )
    elapsed = time.perf_counter() - started
    best_iteration = int(
        estimator.best_iteration_
        if getattr(estimator, "best_iteration_", 0)
        else params["n_estimators"]
    )
    return TrainedModel(
        model_name=model_name,
        forecast_hour=forecast_hour,
        feature_variant=feature_variant,
        feature_columns=x_train.columns.tolist(),
        estimator=estimator,
        best_iteration=best_iteration,
        training_rows=len(train),
        validation_rows=len(validation),
        training_seconds=elapsed,
    )


def _fit_xgboost(
    forecast_hour: int,
    source_columns: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: dict[str, Any],
    n_estimators: int | None = None,
    use_early_stopping: bool = True,
) -> TrainedModel:
    x_train = encode_features(train, source_columns)
    params = dict(config["xgboost"])
    params["random_state"] = int(config["random_seed"])
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    fit_kwargs: dict[str, Any] = {}
    if use_early_stopping and len(validation):
        x_validation = encode_features(
            validation, source_columns, expected_columns=x_train.columns
        )
        params["early_stopping_rounds"] = 80
        fit_kwargs["eval_set"] = [
            (x_validation, validation.target_pm25_ug_m3)
        ]
        fit_kwargs["verbose"] = False
    estimator = xgb.XGBRegressor(**params)
    started = time.perf_counter()
    estimator.fit(
        x_train,
        train.target_pm25_ug_m3,
        **fit_kwargs,
    )
    elapsed = time.perf_counter() - started
    best_iteration = int(getattr(estimator, "best_iteration", params["n_estimators"] - 1)) + 1
    return TrainedModel(
        model_name="cams_xgboost",
        forecast_hour=forecast_hour,
        feature_variant="cams",
        feature_columns=x_train.columns.tolist(),
        estimator=estimator,
        best_iteration=best_iteration,
        training_rows=len(train),
        validation_rows=len(validation),
        training_seconds=elapsed,
    )


def predict_model(model: TrainedModel, frame: pd.DataFrame, source_columns: list[str]) -> np.ndarray:
    x = encode_features(frame, source_columns, expected_columns=model.feature_columns)
    prediction = model.estimator.predict(x)
    return np.maximum(np.asarray(prediction, dtype=float), 0.0)


def _climatology_predictor(
    train: pd.DataFrame, frames: Sequence[pd.DataFrame]
) -> list[np.ndarray]:
    valid = train.dropna(subset=["target_pm25_ug_m3"])
    detailed = valid.groupby(
        ["station_code", "target_month", "target_hour_local"], observed=True
    ).target_pm25_ug_m3.median()
    station_hour = valid.groupby(
        ["station_code", "target_hour_local"], observed=True
    ).target_pm25_ug_m3.median()
    hour = valid.groupby("target_hour_local").target_pm25_ug_m3.median()
    overall = float(valid.target_pm25_ug_m3.median())
    outputs: list[np.ndarray] = []
    for frame in frames:
        detailed_keys = pd.MultiIndex.from_frame(
            frame[["station_code", "target_month", "target_hour_local"]]
        )
        prediction = detailed.reindex(detailed_keys).to_numpy(float)
        missing = ~np.isfinite(prediction)
        if missing.any():
            station_hour_keys = pd.MultiIndex.from_frame(
                frame.loc[missing, ["station_code", "target_hour_local"]]
            )
            prediction[missing] = station_hour.reindex(station_hour_keys).to_numpy(float)
        missing = ~np.isfinite(prediction)
        if missing.any():
            prediction[missing] = hour.reindex(
                frame.loc[missing, "target_hour_local"]
            ).to_numpy(float)
        prediction[~np.isfinite(prediction)] = overall
        outputs.append(np.maximum(prediction, 0.0))
    return outputs


def metric_values(observed: pd.Series, predicted: pd.Series) -> dict[str, float]:
    valid = observed.notna() & predicted.notna() & np.isfinite(predicted)
    y = observed.loc[valid].to_numpy(float)
    p = predicted.loc[valid].to_numpy(float)
    if not len(y):
        return {
            "n": 0,
            "mae_ug_m3": np.nan,
            "rmse_ug_m3": np.nan,
            "bias_ug_m3": np.nan,
            "correlation": np.nan,
            "r2": np.nan,
        }
    correlation = float(np.corrcoef(y, p)[0, 1]) if len(y) >= 2 and np.std(y) > 0 and np.std(p) > 0 else np.nan
    return {
        "n": len(y),
        "mae_ug_m3": float(mean_absolute_error(y, p)),
        "rmse_ug_m3": float(math.sqrt(mean_squared_error(y, p))),
        "bias_ug_m3": float(np.mean(p - y)),
        "correlation": correlation,
        "r2": float(r2_score(y, p)) if len(y) >= 2 and np.std(y) > 0 else np.nan,
    }


def performance_tables(
    predictions: pd.DataFrame, model_columns: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    station_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    for (split, horizon), group in predictions.groupby(["split", "forecast_hour"]):
        common = group.target_pm25_ug_m3.notna()
        for column in model_columns:
            common &= group[column].notna()
        common_group = group.loc[common]
        persistence_mae = metric_values(
            common_group.target_pm25_ug_m3, common_group.persistence
        )["mae_ug_m3"]
        for model_name in model_columns:
            values = metric_values(
                common_group.target_pm25_ug_m3, common_group[model_name]
            )
            values.update(
                {
                    "split": split,
                    "forecast_hour": int(horizon),
                    "model": model_name,
                    "scope": "pooled_common_cases",
                    "skill_vs_persistence_pct": (
                        100.0 * (1.0 - values["mae_ug_m3"] / persistence_mae)
                        if persistence_mae and np.isfinite(persistence_mae)
                        else np.nan
                    ),
                }
            )
            pooled_rows.append(values)
            for station_code, station in common_group.groupby("station_code"):
                station_values = metric_values(
                    station.target_pm25_ug_m3, station[model_name]
                )
                station_persistence = metric_values(
                    station.target_pm25_ug_m3, station.persistence
                )["mae_ug_m3"]
                station_values.update(
                    {
                        "split": split,
                        "forecast_hour": int(horizon),
                        "model": model_name,
                        "station_code": station_code,
                        "skill_vs_persistence_pct": (
                            100.0
                            * (1.0 - station_values["mae_ug_m3"] / station_persistence)
                            if station_persistence and np.isfinite(station_persistence)
                            else np.nan
                        ),
                    }
                )
                station_rows.append(station_values)
    by_station = pd.DataFrame(station_rows)
    summary = pd.DataFrame(pooled_rows)
    balanced = (
        by_station.groupby(["split", "forecast_hour", "model"], as_index=False)
        .agg(
            n_stations=("station_code", "nunique"),
            n=("n", "sum"),
            mae_ug_m3=("mae_ug_m3", "mean"),
            rmse_ug_m3=("rmse_ug_m3", "mean"),
            bias_ug_m3=("bias_ug_m3", "mean"),
            correlation=("correlation", "mean"),
            r2=("r2", "mean"),
            skill_vs_persistence_pct=("skill_vs_persistence_pct", "mean"),
        )
    )
    balanced["scope"] = "station_balanced_common_cases"
    summary = pd.concat([summary, balanced], ignore_index=True, sort=False)
    return by_station, summary


def choose_champion(validation_summary: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    candidates = ["obs_lgbm", "cams_lgbm", "cams_xgboost"]
    subset = validation_summary.loc[
        validation_summary.split.eq("validation")
        & validation_summary.scope.eq("station_balanced_common_cases")
        & validation_summary.model.isin(candidates)
    ]
    ranking = (
        subset.groupby("model", as_index=False)
        .agg(
            mean_station_balanced_mae_ug_m3=("mae_ug_m3", "mean"),
            mean_skill_vs_persistence_pct=("skill_vs_persistence_pct", "mean"),
        )
        .sort_values("mean_station_balanced_mae_ug_m3")
        .reset_index(drop=True)
    )
    if len(ranking) != len(candidates):
        raise ValueError(f"Champion ranking lacks candidates: {ranking}")
    best = float(ranking.iloc[0].mean_station_balanced_mae_ug_m3)
    within_one_percent = ranking.loc[
        ranking.mean_station_balanced_mae_ug_m3.le(best * 1.01)
    ].copy()
    priority = {"cams_lgbm": 0, "obs_lgbm": 1, "cams_xgboost": 2}
    within_one_percent["operational_priority"] = within_one_percent.model.map(priority)
    champion = str(within_one_percent.sort_values("operational_priority").iloc[0].model)
    ranking["selected_champion"] = ranking.model.eq(champion)
    return champion, ranking


def _conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    scores = scores[np.isfinite(scores)]
    if not len(scores):
        return 0.0
    probability = min(math.ceil((len(scores) + 1) * coverage) / len(scores), 1.0)
    # Use a conservative non-negative correction. Standard conformalized
    # quantile regression permits a negative value (interval shrinkage), but
    # expansion-only calibration is safer for the first operational release.
    return max(float(np.quantile(scores, probability, method="higher")), 0.0)


def _interval_period_mask(
    frame: pd.DataFrame,
    config: dict[str, Any],
    period: str,
) -> pd.Series:
    """Return the configured target-time mask for quantile tuning or calibration."""

    interval_config = config["prediction_intervals"]
    if period == "tuning":
        start_key = "quantile_tuning_target_start_utc"
        end_key = "quantile_tuning_target_end_utc"
    elif period == "calibration":
        start_key = "conformal_calibration_target_start_utc"
        end_key = "conformal_calibration_target_end_utc"
    else:
        raise ValueError(f"Unknown interval period: {period}")
    start = pd.Timestamp(interval_config[start_key])
    end = pd.Timestamp(interval_config[end_key])
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Prediction-interval boundaries must include a timezone")
    start = start.tz_convert("UTC")
    end = end.tz_convert("UTC")
    if start > end:
        raise ValueError(f"Prediction-interval {period} start is after its end")
    return frame.target_time_utc.between(start, end, inclusive="both")


def interval_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    alpha = 0.2
    for (split, horizon), group in predictions.groupby(["split", "forecast_hour"]):
        valid = group.dropna(
            subset=["target_pm25_ug_m3", "prediction_q10", "prediction_q50", "prediction_q90"]
        )
        if valid.empty:
            continue
        y = valid.target_pm25_ug_m3.to_numpy(float)
        lower = valid.prediction_q10.to_numpy(float)
        median = valid.prediction_q50.to_numpy(float)
        upper = valid.prediction_q90.to_numpy(float)
        interval_score = (upper - lower) + 2.0 / alpha * (
            (lower - y) * (y < lower) + (y - upper) * (y > upper)
        )
        rows.append(
            {
                "split": split,
                "forecast_hour": int(horizon),
                "n": len(valid),
                "nominal_coverage_pct": 80.0,
                "empirical_coverage_pct": 100.0 * np.mean((y >= lower) & (y <= upper)),
                "mean_interval_width_ug_m3": float(np.mean(upper - lower)),
                "mean_interval_score_ug_m3": float(np.mean(interval_score)),
                "median_pinball_loss_ug_m3": float(
                    mean_pinball_loss(y, median, alpha=0.5)
                ),
                "q10_pinball_loss_ug_m3": float(
                    mean_pinball_loss(y, lower, alpha=0.1)
                ),
                "q90_pinball_loss_ug_m3": float(
                    mean_pinball_loss(y, upper, alpha=0.9)
                ),
            }
        )
    return pd.DataFrame(rows)


def block_bootstrap_skill(
    predictions: pd.DataFrame,
    model_columns: Sequence[str],
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    test = predictions.loc[predictions.split.eq("test")].copy()
    calendar = test.target_time_utc.dt.isocalendar()
    test["station_week"] = (
        test.station_code.astype(str)
        + "_"
        + calendar.year.astype(str)
        + "W"
        + calendar.week.astype(str).str.zfill(2)
    )
    for horizon, group in test.groupby("forecast_hour"):
        for model_name in model_columns:
            if model_name == "persistence":
                continue
            valid = group.dropna(
                subset=["target_pm25_ug_m3", "persistence", model_name]
            ).copy()
            valid["improvement"] = (
                (valid.persistence - valid.target_pm25_ug_m3).abs()
                - (valid[model_name] - valid.target_pm25_ug_m3).abs()
            )
            cluster_values = valid.groupby("station_week").improvement.mean().to_numpy(float)
            if not len(cluster_values):
                continue
            samples = rng.choice(
                cluster_values,
                size=(replicates, len(cluster_values)),
                replace=True,
            ).mean(axis=1)
            rows.append(
                {
                    "forecast_hour": int(horizon),
                    "model": model_name,
                    "n_station_weeks": len(cluster_values),
                    "mae_improvement_ug_m3": float(cluster_values.mean()),
                    "ci95_lower_ug_m3": float(np.quantile(samples, 0.025)),
                    "ci95_upper_ug_m3": float(np.quantile(samples, 0.975)),
                    "bootstrap_replicates": replicates,
                }
            )
    return pd.DataFrame(rows)


def _feature_importance(models: Sequence[TrainedModel]) -> pd.DataFrame:
    rows = []
    for trained in models:
        importance = np.asarray(trained.estimator.feature_importances_, dtype=float)
        total = importance.sum()
        for name, value in zip(trained.feature_columns, importance, strict=True):
            rows.append(
                {
                    "model": trained.model_name,
                    "forecast_hour": trained.forecast_hour,
                    "feature": name,
                    "gain_or_split_importance": value,
                    "normalized_importance_pct": 100.0 * value / total if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _spatial_transfer_evaluation(
    modeling: pd.DataFrame,
    source_columns: list[str],
    include_cams: bool,
    best_iterations: dict[int, int],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    station_codes = np.array(sorted(modeling.station_code.unique()))
    splitter = KFold(
        n_splits=int(config["station_transfer_folds"]),
        shuffle=True,
        random_state=int(config["random_seed"]),
    )
    assignments = []
    prediction_rows = []
    runtime_rows = []
    for fold, (_, held_index) in enumerate(splitter.split(station_codes), start=1):
        held_stations = set(station_codes[held_index])
        for station_code in sorted(held_stations):
            assignments.append({"fold": fold, "station_code": station_code})
        for horizon in config["forecast_hours"]:
            horizon_data = modeling.loc[modeling.forecast_hour.eq(horizon)]
            train = horizon_data.loc[
                horizon_data.split.eq("train")
                & ~horizon_data.station_code.isin(held_stations)
                & horizon_data.target_pm25_ug_m3.notna()
            ].copy()
            test = horizon_data.loc[
                horizon_data.split.eq("test")
                & horizon_data.station_code.isin(held_stations)
                & horizon_data.target_pm25_ug_m3.notna()
            ].copy()
            if include_cams:
                train = train.loc[train.cams_pm25_ug_m3.notna()]
                test = test.loc[test.cams_pm25_ug_m3.notna()]
            if train.empty or test.empty:
                continue
            trained = _fit_lightgbm(
                model_name="station_transfer_lgbm",
                forecast_hour=int(horizon),
                feature_variant="cams" if include_cams else "observation_only",
                source_columns=source_columns,
                train=train,
                validation=test.iloc[0:0],
                config=config,
                n_estimators=best_iterations[int(horizon)],
                omit_station_identity=True,
                use_early_stopping=False,
            )
            test["prediction"] = predict_model(trained, test, source_columns)
            test["fold"] = fold
            prediction_rows.append(
                test[
                    [
                        "fold",
                        "station_code",
                        "issue_time_utc",
                        "target_time_utc",
                        "forecast_hour",
                        "target_pm25_ug_m3",
                        "prediction",
                        "pm25_lag_0h",
                    ]
                ]
            )
            runtime_rows.append(
                {
                    "stage": "station_transfer_fit",
                    "fold": fold,
                    "forecast_hour": int(horizon),
                    "seconds": trained.training_seconds,
                    "training_rows": trained.training_rows,
                    "evaluation_rows": len(test),
                }
            )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    metric_rows = []
    for (horizon, station_code), group in predictions.groupby(
        ["forecast_hour", "station_code"]
    ):
        values = metric_values(group.target_pm25_ug_m3, group.prediction)
        persistence = metric_values(group.target_pm25_ug_m3, group.pm25_lag_0h)
        values.update(
            {
                "forecast_hour": int(horizon),
                "station_code": station_code,
                "persistence_mae_ug_m3": persistence["mae_ug_m3"],
                "skill_vs_persistence_pct": (
                    100.0 * (1.0 - values["mae_ug_m3"] / persistence["mae_ug_m3"])
                    if persistence["mae_ug_m3"]
                    else np.nan
                ),
            }
        )
        metric_rows.append(values)
    assignments_frame = pd.DataFrame(assignments)
    return predictions, pd.DataFrame(metric_rows), pd.concat(
        [assignments_frame.assign(stage="station_assignment"), pd.DataFrame(runtime_rows)],
        ignore_index=True,
        sort=False,
    )


def train_and_evaluate(paths: ExperimentPaths | None = None) -> dict[str, Any]:
    paths = paths or ExperimentPaths()
    config = load_config(paths.config)
    modeling = load_modeling_table(paths)
    paths.tables.mkdir(parents=True, exist_ok=True)
    model_dir = paths.root / "models" / "research"
    model_dir.mkdir(parents=True, exist_ok=True)
    paths.provenance.mkdir(parents=True, exist_ok=True)

    obs_columns = feature_columns(modeling, include_cams=False)
    cams_columns = feature_columns(modeling, include_cams=True)
    candidate_models: dict[str, dict[int, TrainedModel]] = {
        "obs_lgbm": {},
        "cams_lgbm": {},
        "cams_xgboost": {},
    }
    predictions_by_horizon: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()

    for horizon in config["forecast_hours"]:
        horizon_data = modeling.loc[modeling.forecast_hour.eq(horizon)].copy()
        train = horizon_data.loc[
            horizon_data.split.eq("train") & horizon_data.target_pm25_ug_m3.notna()
        ].copy()
        validation = horizon_data.loc[
            horizon_data.split.eq("validation") & horizon_data.target_pm25_ug_m3.notna()
        ].copy()
        test = horizon_data.loc[
            horizon_data.split.eq("test") & horizon_data.target_pm25_ug_m3.notna()
        ].copy()
        cams_train = train.loc[train.cams_pm25_ug_m3.notna()].copy()
        cams_validation = validation.loc[validation.cams_pm25_ug_m3.notna()].copy()
        if cams_train.empty or cams_validation.empty:
            raise ValueError(f"CAMS predictor coverage is absent for horizon {horizon}")

        obs_model = _fit_lightgbm(
            "obs_lgbm",
            int(horizon),
            "observation_only",
            obs_columns,
            train,
            validation,
            config,
        )
        cams_lgbm = _fit_lightgbm(
            "cams_lgbm",
            int(horizon),
            "cams",
            cams_columns,
            cams_train,
            cams_validation,
            config,
        )
        cams_xgboost = _fit_xgboost(
            int(horizon), cams_columns, cams_train, cams_validation, config
        )
        for trained in (obs_model, cams_lgbm, cams_xgboost):
            candidate_models[trained.model_name][int(horizon)] = trained
            runtime_rows.append(
                {
                    "stage": "candidate_fit",
                    "model": trained.model_name,
                    "forecast_hour": int(horizon),
                    "seconds": trained.training_seconds,
                    "training_rows": trained.training_rows,
                    "validation_rows": trained.validation_rows,
                    "best_iteration": trained.best_iteration,
                }
            )

        climatology_validation, climatology_test = _climatology_predictor(
            train, [validation, test]
        )
        for split_name, frame, climatology in (
            ("validation", validation, climatology_validation),
            ("test", test, climatology_test),
        ):
            output = frame[
                [
                    "station_code",
                    "station_name",
                    "region",
                    "issue_time_utc",
                    "target_time_utc",
                    "forecast_hour",
                    "target_pm25_ug_m3",
                    "pm25_lag_0h",
                    "cams_pm25_ug_m3",
                    "latest_pm25_age_hours",
                    "target_month",
                    "target_hour_local",
                ]
            ].copy()
            output["split"] = split_name
            output["persistence"] = output.pm25_lag_0h
            output["climatology"] = climatology
            output["raw_cams"] = output.cams_pm25_ug_m3
            output["obs_lgbm"] = predict_model(obs_model, frame, obs_columns)
            cams_available = output.cams_pm25_ug_m3.notna().to_numpy()
            output["cams_lgbm"] = np.where(
                cams_available,
                predict_model(cams_lgbm, frame, cams_columns),
                np.nan,
            )
            output["cams_xgboost"] = np.where(
                cams_available,
                predict_model(
                cams_xgboost, frame, cams_columns
                ),
                np.nan,
            )
            predictions_by_horizon.append(output)

    predictions = pd.concat(predictions_by_horizon, ignore_index=True)
    candidate_columns = [
        "persistence",
        "climatology",
        "raw_cams",
        "obs_lgbm",
        "cams_lgbm",
        "cams_xgboost",
    ]
    candidate_station_metrics, candidate_summary = performance_tables(
        predictions, candidate_columns
    )
    champion, ranking = choose_champion(candidate_summary)
    predictions["champion"] = predictions[champion]

    include_cams = champion != "obs_lgbm"
    champion_source_columns = cams_columns if include_cams else obs_columns
    quantile_models: dict[float, dict[int, TrainedModel]] = {
        float(quantile): {} for quantile in config["quantiles"]
    }
    corrections: dict[int, float] = {}
    quantile_crossings: list[dict[str, Any]] = []
    interval_config = config["prediction_intervals"]
    nominal_coverage = float(interval_config["nominal_coverage"])
    for horizon in config["forecast_hours"]:
        horizon_data = modeling.loc[modeling.forecast_hour.eq(horizon)].copy()
        train = horizon_data.loc[
            horizon_data.split.eq("train") & horizon_data.target_pm25_ug_m3.notna()
        ].copy()
        validation = horizon_data.loc[
            horizon_data.split.eq("validation") & horizon_data.target_pm25_ug_m3.notna()
        ].copy()
        tuning = validation.loc[_interval_period_mask(validation, config, "tuning")].copy()
        calibration = validation.loc[
            _interval_period_mask(validation, config, "calibration")
        ].copy()
        if tuning.empty or calibration.empty:
            raise ValueError(
                f"Interval tuning or calibration is empty for horizon {horizon}"
            )
        if tuning.target_time_utc.max() >= calibration.target_time_utc.min():
            raise ValueError(
                "Quantile tuning must end before conformal calibration begins"
            )
        if include_cams:
            train = train.loc[train.cams_pm25_ug_m3.notna()]
            validation = validation.loc[validation.cams_pm25_ug_m3.notna()]
            tuning = tuning.loc[tuning.cams_pm25_ug_m3.notna()]
            calibration = calibration.loc[calibration.cams_pm25_ug_m3.notna()]
        quantile_predictions: dict[float, np.ndarray] = {}
        prediction_index = predictions.index[
            predictions.forecast_hour.eq(horizon)
            & predictions.split.eq("validation")
            & (
                predictions.cams_pm25_ug_m3.notna()
                if include_cams
                else pd.Series(True, index=predictions.index)
            )
        ]
        test_index = predictions.index[
            predictions.forecast_hour.eq(horizon)
            & predictions.split.eq("test")
            & (
                predictions.cams_pm25_ug_m3.notna()
                if include_cams
                else pd.Series(True, index=predictions.index)
            )
        ]
        validation_prediction_frame = modeling.loc[
            modeling.forecast_hour.eq(horizon)
            & modeling.split.eq("validation")
            & modeling.target_pm25_ug_m3.notna()
        ].copy()
        test_prediction_frame = modeling.loc[
            modeling.forecast_hour.eq(horizon)
            & modeling.split.eq("test")
            & modeling.target_pm25_ug_m3.notna()
        ].copy()
        if include_cams:
            validation_prediction_frame = validation_prediction_frame.loc[
                validation_prediction_frame.cams_pm25_ug_m3.notna()
            ]
            test_prediction_frame = test_prediction_frame.loc[
                test_prediction_frame.cams_pm25_ug_m3.notna()
            ]
        for quantile in (0.1, 0.5, 0.9):
            trained = _fit_lightgbm(
                model_name=f"quantile_{quantile:.1f}",
                forecast_hour=int(horizon),
                feature_variant="cams" if include_cams else "observation_only",
                source_columns=champion_source_columns,
                train=train,
                validation=tuning,
                config=config,
                objective="quantile",
                alpha=quantile,
            )
            quantile_models[quantile][int(horizon)] = trained
            runtime_rows.append(
                {
                    "stage": "quantile_fit",
                    "model": trained.model_name,
                    "forecast_hour": int(horizon),
                    "seconds": trained.training_seconds,
                    "training_rows": trained.training_rows,
                    "validation_rows": trained.validation_rows,
                    "best_iteration": trained.best_iteration,
                }
            )
            quantile_predictions[quantile] = predict_model(
                trained, validation_prediction_frame, champion_source_columns
            )
            predictions.loc[test_index, f"prediction_q{int(quantile * 100):02d}"] = (
                predict_model(trained, test_prediction_frame, champion_source_columns)
            )
        raw_validation = np.column_stack(
            [quantile_predictions[0.1], quantile_predictions[0.5], quantile_predictions[0.9]]
        )
        ordered_validation = np.sort(raw_validation, axis=1)
        crossing_rate = 100.0 * np.mean(
            (raw_validation[:, 0] > raw_validation[:, 1])
            | (raw_validation[:, 1] > raw_validation[:, 2])
        )
        calibration_mask = _interval_period_mask(
            validation_prediction_frame, config, "calibration"
        ).to_numpy()
        if int(calibration_mask.sum()) != len(calibration):
            raise ValueError(
                f"Interval calibration row mismatch for horizon {horizon}: "
                f"{int(calibration_mask.sum())} prediction rows versus {len(calibration)} fit rows"
            )
        y_calibration = validation_prediction_frame.loc[
            calibration_mask, "target_pm25_ug_m3"
        ].to_numpy(float)
        ordered_calibration = ordered_validation[calibration_mask]
        scores = np.maximum(
            ordered_calibration[:, 0] - y_calibration,
            y_calibration - ordered_calibration[:, 2],
        )
        correction = _conformal_quantile(scores, coverage=nominal_coverage)
        corrections[int(horizon)] = correction
        predictions.loc[prediction_index, "prediction_q10"] = np.maximum(
            ordered_validation[:, 0] - correction, 0.0
        )
        predictions.loc[prediction_index, "prediction_q50"] = ordered_validation[:, 1]
        predictions.loc[prediction_index, "prediction_q90"] = (
            ordered_validation[:, 2] + correction
        )
        raw_test = predictions.loc[
            test_index, ["prediction_q10", "prediction_q50", "prediction_q90"]
        ].to_numpy(float)
        ordered_test = np.sort(raw_test, axis=1)
        predictions.loc[test_index, "prediction_q10"] = np.maximum(
            ordered_test[:, 0] - correction, 0.0
        )
        predictions.loc[test_index, "prediction_q50"] = ordered_test[:, 1]
        predictions.loc[test_index, "prediction_q90"] = ordered_test[:, 2] + correction
        quantile_crossings.append(
            {
                "forecast_hour": int(horizon),
                "validation_raw_crossing_pct": crossing_rate,
                "conformal_correction_ug_m3": correction,
                "quantile_tuning_rows": len(tuning),
                "conformal_calibration_rows": len(calibration),
                "quantile_tuning_target_start_utc": interval_config[
                    "quantile_tuning_target_start_utc"
                ],
                "quantile_tuning_target_end_utc": interval_config[
                    "quantile_tuning_target_end_utc"
                ],
                "conformal_calibration_target_start_utc": interval_config[
                    "conformal_calibration_target_start_utc"
                ],
                "conformal_calibration_target_end_utc": interval_config[
                    "conformal_calibration_target_end_utc"
                ],
            }
        )

    final_model_columns = candidate_columns + ["champion", "prediction_q50"]
    by_station, summary = performance_tables(predictions, final_model_columns)
    intervals = interval_metrics(predictions)
    bootstrap = block_bootstrap_skill(
        predictions,
        [
            "climatology",
            "raw_cams",
            "obs_lgbm",
            "cams_lgbm",
            "cams_xgboost",
            "champion",
        ],
        int(config["bootstrap_replicates"]),
        int(config["random_seed"]),
    )

    transfer_iteration_source = (
        champion if champion in {"obs_lgbm", "cams_lgbm"} else "cams_lgbm"
    )
    best_iterations = {
        int(horizon): candidate_models[transfer_iteration_source][
            int(horizon)
        ].best_iteration
        for horizon in config["forecast_hours"]
    }
    transfer_predictions, transfer_metrics, transfer_runtime = _spatial_transfer_evaluation(
        modeling,
        champion_source_columns,
        include_cams,
        best_iterations,
        config,
    )

    trained_for_importance = (
        list(candidate_models[champion].values())
        if champion in candidate_models
        else list(candidate_models["cams_lgbm"].values())
    )
    importance = _feature_importance(trained_for_importance)
    runtime = pd.concat([pd.DataFrame(runtime_rows), transfer_runtime], ignore_index=True, sort=False)
    runtime.loc[len(runtime)] = {
        "stage": "complete_train_evaluate",
        "seconds": time.perf_counter() - started_all,
    }

    predictions.to_csv(
        paths.derived / "validation_test_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    transfer_predictions.to_csv(
        paths.derived / "station_transfer_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    by_station.to_csv(paths.tables / "metrics_by_station.csv", index=False)
    summary.to_csv(paths.tables / "metrics_summary.csv", index=False)
    candidate_station_metrics.to_csv(
        paths.tables / "candidate_validation_metrics_by_station.csv", index=False
    )
    candidate_summary.to_csv(
        paths.tables / "candidate_validation_metrics_summary.csv", index=False
    )
    ranking.to_csv(paths.tables / "model_selection_ranking.csv", index=False)
    intervals.to_csv(paths.tables / "prediction_interval_metrics.csv", index=False)
    bootstrap.to_csv(paths.tables / "block_bootstrap_skill.csv", index=False)
    pd.DataFrame(quantile_crossings).to_csv(
        paths.tables / "quantile_calibration.csv", index=False
    )
    transfer_metrics.to_csv(paths.tables / "station_transfer_metrics.csv", index=False)
    importance.to_csv(paths.tables / "feature_importance.csv", index=False)
    runtime.to_csv(paths.tables / "runtime_by_stage.csv", index=False)

    model_manifest: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": config["experiment_name"],
        "config_sha256": file_sha256(paths.config),
        "modeling_table_sha256": file_sha256(
            paths.derived / "modeling_table.csv.gz"
        ),
        "champion": champion,
        "champion_feature_variant": "cams" if include_cams else "observation_only",
        "selection_rule": (
            "lowest validation station-balanced MAE; prefer LightGBM/CAMS when within "
            "1 percent of the minimum for operational simplicity and physical forecast input"
        ),
        "feature_source_columns": champion_source_columns,
        "conformal_correction_ug_m3": corrections,
        "prediction_interval_design": {
            "nominal_coverage_pct": 100.0 * nominal_coverage,
            "method": "lead-specific split-conformal expansion pooled across stations",
            "quantile_fit_target_period": {
                "start_utc": config["splits"]["training_target_start_utc"],
                "end_utc": config["splits"]["training_target_end_utc"],
            },
            "quantile_tuning_target_period": {
                "start_utc": interval_config["quantile_tuning_target_start_utc"],
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
                str(row["forecast_hour"]): int(row["conformal_calibration_rows"])
                for row in quantile_crossings
            },
        },
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__,
        },
        "models": [],
    }
    for model_name, horizon_models in candidate_models.items():
        for horizon, trained in horizon_models.items():
            output = model_dir / f"{model_name}_{horizon:03d}h.joblib"
            joblib.dump(trained, output, compress=3)
            model_manifest["models"].append(
                {
                    "model": model_name,
                    "forecast_hour": horizon,
                    "path": str(output.relative_to(paths.root)),
                    "bytes": output.stat().st_size,
                    "sha256": file_sha256(output),
                    "best_iteration": trained.best_iteration,
                    "training_rows": trained.training_rows,
                    "validation_rows": trained.validation_rows,
                }
            )
    for quantile, horizon_models in quantile_models.items():
        for horizon, trained in horizon_models.items():
            output = model_dir / f"quantile_{int(quantile * 100):02d}_{horizon:03d}h.joblib"
            joblib.dump(trained, output, compress=3)
            model_manifest["models"].append(
                {
                    "model": f"quantile_{quantile:.1f}",
                    "forecast_hour": horizon,
                    "path": str(output.relative_to(paths.root)),
                    "bytes": output.stat().st_size,
                    "sha256": file_sha256(output),
                    "best_iteration": trained.best_iteration,
                    "training_rows": trained.training_rows,
                    "validation_rows": trained.validation_rows,
                }
            )
    write_json(paths.provenance / "research_model_manifest.json", model_manifest)
    return {
        "champion": champion,
        "ranking": ranking.to_dict(orient="records"),
        "test_metrics": summary.loc[
            summary.split.eq("test")
            & summary.scope.eq("station_balanced_common_cases")
            & summary.model.eq("champion")
        ].to_dict(orient="records"),
        "interval_metrics": intervals.loc[intervals.split.eq("test")].to_dict(
            orient="records"
        ),
        "runtime_seconds": time.perf_counter() - started_all,
    }
