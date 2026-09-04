"""Robustness and failure-mode diagnostics for the held-out PM2.5 forecasts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .data import ExperimentPaths, load_config
from .modeling import (
    _fit_lightgbm,
    _fit_xgboost,
    feature_columns,
    load_modeling_table,
    metric_values,
    performance_tables,
    predict_model,
)


def _station_balanced_metrics(
    frame: pd.DataFrame, model_name: str, prediction_column: str
) -> dict[str, Any]:
    station_rows: list[dict[str, float]] = []
    for _, station in frame.groupby("station_code", observed=True):
        values = metric_values(
            station.target_pm25_ug_m3, station[prediction_column]
        )
        persistence_mae = metric_values(
            station.target_pm25_ug_m3, station.persistence
        )["mae_ug_m3"]
        values["skill_vs_persistence_pct"] = (
            100.0 * (1.0 - values["mae_ug_m3"] / persistence_mae)
            if persistence_mae and np.isfinite(persistence_mae)
            else np.nan
        )
        station_rows.append(values)
    metrics = pd.DataFrame(station_rows)
    if metrics.empty:
        return {
            "model": model_name,
            "n": 0,
            "n_stations": 0,
            "mae_ug_m3": np.nan,
            "rmse_ug_m3": np.nan,
            "bias_ug_m3": np.nan,
            "correlation": np.nan,
            "r2": np.nan,
            "skill_vs_persistence_pct": np.nan,
        }
    return {
        "model": model_name,
        "n": int(metrics.n.sum()),
        "n_stations": len(metrics),
        "mae_ug_m3": float(metrics.mae_ug_m3.mean()),
        "rmse_ug_m3": float(metrics.rmse_ug_m3.mean()),
        "bias_ug_m3": float(metrics.bias_ug_m3.mean()),
        "correlation": float(metrics.correlation.mean()),
        "r2": float(metrics.r2.mean()),
        "skill_vs_persistence_pct": float(
            metrics.skill_vs_persistence_pct.mean()
        ),
    }


def station_high_thresholds(modeling: pd.DataFrame) -> pd.DataFrame:
    """Compute station/lead high-concentration thresholds from training only."""

    return (
        modeling.loc[
            modeling.split.eq("train") & modeling.target_pm25_ug_m3.notna()
        ]
        .groupby(["station_code", "forecast_hour"], as_index=False)
        .target_pm25_ug_m3.quantile(0.9)
        .rename(columns={"target_pm25_ug_m3": "training_station_q90_ug_m3"})
    )


def stratified_performance(
    predictions: pd.DataFrame, modeling: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate wet/dry seasons and training-defined high PM2.5 episodes."""

    thresholds = station_high_thresholds(modeling)
    test = predictions.loc[predictions.split.eq("test")].merge(
        thresholds,
        on=["station_code", "forecast_hour"],
        how="left",
        validate="many_to_one",
    )
    test["season"] = np.where(
        test.target_month.isin([11, 12, 1, 2, 3, 4]),
        "November-April",
        "May-October",
    )
    test["concentration_regime"] = np.where(
        test.target_pm25_ug_m3.ge(test.training_station_q90_ug_m3),
        "at_or_above_training_station_q90",
        "below_training_station_q90",
    )

    rows: list[dict[str, Any]] = []
    models = {
        "persistence": "persistence",
        "climatology": "climatology",
        "raw_cams": "raw_cams",
        "observation_only_ml": "obs_lgbm",
        "selected_model": "champion",
    }
    strata = [
        ("all_test", pd.Series("all", index=test.index)),
        ("season", test.season),
        ("concentration_regime", test.concentration_regime),
    ]
    for horizon, horizon_frame in test.groupby("forecast_hour", observed=True):
        for stratum_type, labels in strata:
            local_labels = labels.loc[horizon_frame.index]
            for stratum, group_index in local_labels.groupby(local_labels).groups.items():
                group = horizon_frame.loc[group_index]
                required = ["target_pm25_ug_m3", "persistence", *models.values()]
                group = group.dropna(subset=list(dict.fromkeys(required)))
                for label, prediction_column in models.items():
                    values = _station_balanced_metrics(
                        group, label, prediction_column
                    )
                    values.update(
                        {
                            "forecast_hour": int(horizon),
                            "stratum_type": stratum_type,
                            "stratum": str(stratum),
                        }
                    )
                    rows.append(values)

    event_rows: list[dict[str, Any]] = []
    for horizon, group in test.groupby("forecast_hour", observed=True):
        valid = group.dropna(
            subset=[
                "target_pm25_ug_m3",
                "champion",
                "training_station_q90_ug_m3",
            ]
        )
        observed_event = valid.target_pm25_ug_m3.ge(
            valid.training_station_q90_ug_m3
        )
        predicted_event = valid.champion.ge(valid.training_station_q90_ug_m3)
        hits = int((observed_event & predicted_event).sum())
        misses = int((observed_event & ~predicted_event).sum())
        false_alarms = int((~observed_event & predicted_event).sum())
        correct_negatives = int((~observed_event & ~predicted_event).sum())
        event_rows.append(
            {
                "forecast_hour": int(horizon),
                "threshold_definition": "station-specific training-period 90th percentile",
                "n": len(valid),
                "hits": hits,
                "misses": misses,
                "false_alarms": false_alarms,
                "correct_negatives": correct_negatives,
                "probability_of_detection_pct": (
                    100.0 * hits / (hits + misses) if hits + misses else np.nan
                ),
                "false_alarm_ratio_pct": (
                    100.0 * false_alarms / (hits + false_alarms)
                    if hits + false_alarms
                    else np.nan
                ),
                "critical_success_index_pct": (
                    100.0 * hits / (hits + misses + false_alarms)
                    if hits + misses + false_alarms
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(event_rows), thresholds


def residual_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    test = predictions.loc[predictions.split.eq("test")].dropna(
        subset=["target_pm25_ug_m3", "champion"]
    ).copy()
    test["residual_ug_m3"] = test.champion - test.target_pm25_ug_m3
    rows: list[dict[str, Any]] = []
    dimensions = {
        "target_month": "target_month",
        "target_local_hour": "target_hour_local",
        "station": "station_code",
    }
    for horizon, horizon_frame in test.groupby("forecast_hour", observed=True):
        for label, column in dimensions.items():
            for value, group in horizon_frame.groupby(column, observed=True):
                rows.append(
                    {
                        "forecast_hour": int(horizon),
                        "dimension": label,
                        "group": value,
                        "n": len(group),
                        "mean_residual_ug_m3": float(group.residual_ug_m3.mean()),
                        "mae_ug_m3": float(group.residual_ug_m3.abs().mean()),
                        "residual_sd_ug_m3": float(group.residual_ug_m3.std(ddof=1)),
                    }
                )
    return pd.DataFrame(rows)


def _population_stability_index(train: pd.Series, test: pd.Series) -> float:
    train_values = train.dropna().to_numpy(float)
    test_values = test.dropna().to_numpy(float)
    if len(train_values) < 20 or len(test_values) < 20:
        return np.nan
    edges = np.unique(np.quantile(train_values, np.linspace(0.0, 1.0, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    train_counts, _ = np.histogram(train_values, bins=edges)
    test_counts, _ = np.histogram(test_values, bins=edges)
    train_fraction = np.maximum(train_counts / train_counts.sum(), 1.0e-6)
    test_fraction = np.maximum(test_counts / test_counts.sum(), 1.0e-6)
    return float(
        np.sum((test_fraction - train_fraction) * np.log(test_fraction / train_fraction))
    )


def distribution_shift_diagnostics(
    modeling: pd.DataFrame, source_columns: Sequence[str]
) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in source_columns
        if column in modeling.columns
        and column not in {"station_code", "region", "timezone"}
        and pd.api.types.is_numeric_dtype(modeling[column])
    ]
    rows: list[dict[str, Any]] = []
    for horizon, frame in modeling.groupby("forecast_hour", observed=True):
        train = frame.loc[frame.split.eq("train")]
        test = frame.loc[frame.split.eq("test")]
        for column in numeric_columns:
            train_values = train[column]
            test_values = test[column]
            train_mean = float(train_values.mean())
            test_mean = float(test_values.mean())
            train_sd = float(train_values.std(ddof=1))
            rows.append(
                {
                    "forecast_hour": int(horizon),
                    "feature": column,
                    "train_n": int(train_values.notna().sum()),
                    "test_n": int(test_values.notna().sum()),
                    "train_missing_pct": 100.0 * train_values.isna().mean(),
                    "test_missing_pct": 100.0 * test_values.isna().mean(),
                    "train_mean": train_mean,
                    "test_mean": test_mean,
                    "standardized_mean_difference": (
                        (test_mean - train_mean) / train_sd
                        if train_sd > 0 and np.isfinite(train_sd)
                        else np.nan
                    ),
                    "population_stability_index": _population_stability_index(
                        train_values, test_values
                    ),
                }
            )
    return pd.DataFrame(rows)


def cams_incremental_skill(
    predictions: pd.DataFrame, replicates: int, random_seed: int
) -> pd.DataFrame:
    """Paired station-week bootstrap of CAMS MOS versus observation-only ML.

    Positive differences mean lower absolute error after adding forecast-valid
    CAMS PM2.5. Clustering by station-week preserves much of the temporal and
    within-station dependence that an independent-row bootstrap would ignore.
    """

    rng = np.random.default_rng(random_seed + 7919)
    rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        split_frame = predictions.loc[predictions.split.eq(split)].copy()
        calendar = split_frame.target_time_utc.dt.isocalendar()
        split_frame["station_week"] = (
            split_frame.station_code.astype(str)
            + "_"
            + calendar.year.astype(str)
            + "W"
            + calendar.week.astype(str).str.zfill(2)
        )
        horizons: list[int | str] = [
            *sorted(split_frame.forecast_hour.unique().tolist()),
            "all",
        ]
        for horizon in horizons:
            group = (
                split_frame
                if horizon == "all"
                else split_frame.loc[split_frame.forecast_hour.eq(horizon)]
            )
            valid = group.dropna(
                subset=["target_pm25_ug_m3", "obs_lgbm", "cams_lgbm"]
            ).copy()
            valid["obs_absolute_error"] = (
                valid.obs_lgbm - valid.target_pm25_ug_m3
            ).abs()
            valid["cams_absolute_error"] = (
                valid.cams_lgbm - valid.target_pm25_ug_m3
            ).abs()
            valid["improvement"] = (
                valid.obs_absolute_error - valid.cams_absolute_error
            )
            clusters = valid.groupby("station_week", observed=True).agg(
                improvement=("improvement", "mean"),
                obs_absolute_error=("obs_absolute_error", "mean"),
            )
            if clusters.empty:
                continue
            samples = rng.choice(
                clusters.improvement.to_numpy(float),
                size=(replicates, len(clusters)),
                replace=True,
            ).mean(axis=1)
            mean_improvement = float(clusters.improvement.mean())
            mean_obs_error = float(clusters.obs_absolute_error.mean())
            rows.append(
                {
                    "split": split,
                    "forecast_hour": str(horizon),
                    "n_common_rows": len(valid),
                    "n_station_weeks": len(clusters),
                    "cams_mae_improvement_over_obs_ml_ug_m3": mean_improvement,
                    "relative_improvement_over_obs_ml_pct": (
                        100.0 * mean_improvement / mean_obs_error
                        if mean_obs_error > 0
                        else np.nan
                    ),
                    "ci95_lower_ug_m3": float(np.quantile(samples, 0.025)),
                    "ci95_upper_ug_m3": float(np.quantile(samples, 0.975)),
                    "bootstrap_probability_positive_pct": float(
                        100.0 * np.mean(samples > 0)
                    ),
                    "bootstrap_replicates": replicates,
                }
            )
    return pd.DataFrame(rows)


def recent_window_sensitivity(
    modeling: pd.DataFrame,
    predictions: pd.DataFrame,
    champion: str,
    best_iterations: dict[int, int],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the frozen champion with a 2024-only training sensitivity."""

    include_cams = champion != "obs_lgbm"
    source_columns = feature_columns(modeling, include_cams=include_cams)
    outputs: list[pd.DataFrame] = []
    for horizon in config["forecast_hours"]:
        frame = modeling.loc[modeling.forecast_hour.eq(horizon)]
        train = frame.loc[
            frame.split.eq("train")
            & frame.target_time_utc.ge(pd.Timestamp("2024-01-01T00:00:00Z"))
            & frame.target_pm25_ug_m3.notna()
        ].copy()
        test = frame.loc[
            frame.split.eq("test") & frame.target_pm25_ug_m3.notna()
        ].copy()
        if include_cams:
            train = train.loc[train.cams_pm25_ug_m3.notna()]
            test = test.loc[test.cams_pm25_ug_m3.notna()]
        if champion == "cams_xgboost":
            fitted = _fit_xgboost(
                int(horizon),
                source_columns,
                train,
                test.iloc[0:0],
                config,
                n_estimators=best_iterations[int(horizon)],
                use_early_stopping=False,
            )
        else:
            fitted = _fit_lightgbm(
                "recent_window_lgbm",
                int(horizon),
                "cams" if include_cams else "observation_only",
                source_columns,
                train,
                test.iloc[0:0],
                config,
                n_estimators=best_iterations[int(horizon)],
                use_early_stopping=False,
            )
        output = test[
            [
                "station_code",
                "issue_time_utc",
                "target_time_utc",
                "forecast_hour",
                "target_pm25_ug_m3",
                "pm25_lag_0h",
            ]
        ].copy()
        output["split"] = "test"
        output["persistence"] = output.pm25_lag_0h
        output["recent_window"] = predict_model(fitted, test, source_columns)
        outputs.append(output)
    sensitivity_predictions = pd.concat(outputs, ignore_index=True)
    _, summary = performance_tables(
        sensitivity_predictions, ["persistence", "recent_window"]
    )
    frozen = predictions.loc[
        predictions.split.eq("test"),
        [
            "station_code",
            "target_time_utc",
            "forecast_hour",
            "target_pm25_ug_m3",
            "persistence",
            "champion",
        ],
    ]
    frozen["split"] = "test"
    _, frozen_summary = performance_tables(frozen, ["persistence", "champion"])
    comparison = pd.concat(
        [
            summary.loc[summary.scope.eq("station_balanced_common_cases")].assign(
                training_window="2024_only"
            ),
            frozen_summary.loc[
                frozen_summary.scope.eq("station_balanced_common_cases")
            ].assign(training_window="2023_2024_frozen_champion"),
        ],
        ignore_index=True,
    )
    return sensitivity_predictions, comparison


def run_diagnostics(paths: ExperimentPaths | None = None) -> dict[str, Any]:
    paths = paths or ExperimentPaths()
    config = load_config(paths.config)
    modeling = load_modeling_table(paths)
    predictions = pd.read_csv(
        paths.derived / "validation_test_predictions.csv.gz",
        parse_dates=["issue_time_utc", "target_time_utc"],
    )
    manifest = json.loads(
        (paths.provenance / "research_model_manifest.json").read_text(encoding="utf-8")
    )
    champion = str(manifest["champion"])
    include_cams = champion != "obs_lgbm"
    source_columns = feature_columns(modeling, include_cams=include_cams)
    best_iterations = {
        int(row["forecast_hour"]): int(row["best_iteration"])
        for row in manifest["models"]
        if row["model"] == champion
    }

    stratified, events, thresholds = stratified_performance(predictions, modeling)
    residuals = residual_diagnostics(predictions)
    shift = distribution_shift_diagnostics(modeling, source_columns)
    recent_predictions, recent = recent_window_sensitivity(
        modeling, predictions, champion, best_iterations, config
    )
    cams_increment = cams_incremental_skill(
        predictions,
        int(config["bootstrap_replicates"]),
        int(config["random_seed"]),
    )
    stratified.to_csv(paths.tables / "stratified_test_metrics.csv", index=False)
    events.to_csv(paths.tables / "high_event_detection_metrics.csv", index=False)
    thresholds.to_csv(paths.tables / "training_high_thresholds.csv", index=False)
    residuals.to_csv(paths.tables / "residual_diagnostics.csv", index=False)
    shift.to_csv(paths.tables / "feature_distribution_shift.csv", index=False)
    recent.to_csv(paths.tables / "recent_window_sensitivity.csv", index=False)
    cams_increment.to_csv(
        paths.tables / "cams_incremental_skill_vs_observation_ml.csv", index=False
    )
    recent_predictions.to_csv(
        paths.derived / "recent_window_test_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    return {
        "champion": champion,
        "stratified_rows": len(stratified),
        "event_rows": len(events),
        "shift_rows": len(shift),
        "recent_window_rows": len(recent_predictions),
        "cams_incremental_rows": len(cams_increment),
    }
