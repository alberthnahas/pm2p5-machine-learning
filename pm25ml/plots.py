"""Publication-ready figures for the PM2.5 station forecast experiment."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.dates import ConciseDateFormatter, AutoDateLocator

from .data import ExperimentPaths, file_sha256, write_json


COLORS = {
    "persistence": "#6B7280",
    "climatology": "#CC79A7",
    "raw_cams": "#56B4E9",
    "obs_lgbm": "#0072B2",
    "cams_lgbm": "#009E73",
    "cams_xgboost": "#E69F00",
    "champion": "#D55E00",
    "interval": "#56B4E9",
}
LABELS = {
    "persistence": "Persistence",
    "climatology": "Training climatology",
    "raw_cams": "Raw CAMS",
    "obs_lgbm": "Observation-only LightGBM",
    "cams_lgbm": "CAMS MOS LightGBM",
    "cams_xgboost": "CAMS MOS XGBoost",
    "champion": "Selected model",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def _canvas(title: str, highlight: str, height: float = 4.6) -> tuple[plt.Figure, Any]:
    fig = plt.figure(figsize=(7.15, height), constrained_layout=False)
    fig.suptitle(title, x=0.08, y=0.975, ha="left", va="top", fontsize=14, weight="bold")
    fig.text(
        0.08,
        0.895,
        highlight,
        ha="left",
        va="top",
        fontsize=9.2,
        color="#1F2937",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#EEF6F3", "edgecolor": "none"},
    )
    axis = fig.add_axes([0.11, 0.15, 0.84, 0.61])
    return fig, axis


def _footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.08, 0.035, text, ha="left", va="bottom", fontsize=7.2, color="#4B5563")


def _save(fig: plt.Figure, figures: Path, stem: str) -> list[dict[str, Any]]:
    figures.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in (
        (".png", {"dpi": 300}),
        (".svg", {}),
    ):
        path = figures / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        if suffix == ".svg":
            # Matplotlib emits trailing spaces in SVG path data. Normalize the
            # text so publication diffs remain reviewable without changing the
            # rendered vector content.
            normalized = "\n".join(
                line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
            )
            path.write_text(normalized + "\n", encoding="utf-8")
        outputs.append(
            {
                "path": str(path.relative_to(figures.parent)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    plt.close(fig)
    return outputs


def _plot_data_quality(paths: ExperimentPaths) -> tuple[str, list[dict[str, Any]]]:
    monthly = pd.read_csv(paths.tables / "monthly_station_coverage.csv")
    quality = pd.read_csv(paths.tables / "data_quality_by_station.csv")
    pivot = monthly.pivot(
        index="station_code", columns="month_utc", values="valid_pm25_coverage_pct"
    )
    order = quality.sort_values("valid_pm25_pct_of_expected_hours").station_code
    pivot = pivot.reindex(order)
    median_coverage = float(quality.valid_pm25_pct_of_expected_hours.median())
    fig, ax = _canvas(
        "Observation availability is heterogeneous across the 27-station network",
        f"Median valid PM₂.₅ coverage is {median_coverage:.1f}% within each station's observed span; blank cells denote no station-month coverage.",
        height=6.4,
    )
    masked = np.ma.masked_invalid(pivot.to_numpy(float))
    cmap = LinearSegmentedColormap.from_list(
        "coverage", ["#F7FBFF", "#9ECAE1", "#2171B5"]
    ).copy()
    cmap.set_bad("#E5E7EB")
    image = ax.imshow(masked, aspect="auto", vmin=0, vmax=100, cmap=cmap)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    month_labels = pivot.columns.tolist()
    tick_positions = np.arange(0, len(month_labels), 6)
    ax.set_xticks(tick_positions, [month_labels[i] for i in tick_positions], rotation=45, ha="right")
    ax.set_xlabel("Month (UTC)")
    ax.set_ylabel("Station code")
    colorbar = fig.colorbar(image, ax=ax, pad=0.015, fraction=0.03)
    colorbar.set_label("Valid PM₂.₅ coverage (%)")
    _footer(
        fig,
        "Source: BMKG hourly station observations, 1 Jan 2021–31 Aug 2026. Values <0 or ≥985 µg m⁻³ are invalid; grey indicates unavailable.",
    )
    return "figure_01_observation_coverage", _save(fig, paths.root / "figures", "figure_01_observation_coverage")


def _plot_performance(paths: ExperimentPaths) -> tuple[str, list[dict[str, Any]]]:
    metrics = pd.read_csv(paths.tables / "metrics_summary.csv")
    subset = metrics.loc[
        metrics.split.eq("test")
        & metrics.scope.eq("station_balanced_common_cases")
        & metrics.model.isin(["persistence", "climatology", "raw_cams", "obs_lgbm", "cams_lgbm", "cams_xgboost"])
    ]
    selected = pd.read_csv(paths.tables / "model_selection_ranking.csv").loc[
        lambda x: x.selected_champion
    ].iloc[0].model
    lead72 = subset.loc[subset.forecast_hour.eq(72)].set_index("model")
    selected_skill = float(lead72.loc[selected, "skill_vs_persistence_pct"])
    skill_phrase = (
        f"reduces station-balanced 72 h MAE by {selected_skill:.1f}%"
        if selected_skill >= 0
        else f"increases station-balanced 72 h MAE by {abs(selected_skill):.1f}%"
    )
    fig, ax = _canvas(
        "Forecast error increases with lead time, but learned models retain skill",
        f"The validation-selected {LABELS[selected]} {skill_phrase} relative to persistence on the untouched 2026 test period.",
    )
    for model in ["persistence", "climatology", "raw_cams", "obs_lgbm", "cams_lgbm", "cams_xgboost"]:
        group = subset.loc[subset.model.eq(model)].sort_values("forecast_hour")
        ax.plot(
            group.forecast_hour,
            group.mae_ug_m3,
            marker="o",
            linewidth=2.3 if model == selected else 1.5,
            markersize=5,
            color=COLORS[model],
            label=LABELS[model] + (" (selected)" if model == selected else ""),
            zorder=3 if model == selected else 2,
        )
    ax.set_xticks(sorted(subset.forecast_hour.unique()))
    ax.set_xlabel("Forecast lead (hours)")
    ax.set_ylabel("Station-balanced MAE (µg m⁻³)")
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.6)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    _footer(
        fig,
        "Test targets: 1 Jan–31 Aug 2026. Each station contributes equally; models are compared on common valid target cases.",
    )
    return "figure_02_test_performance", _save(fig, paths.root / "figures", "figure_02_test_performance")


def _plot_station_skill(paths: ExperimentPaths) -> tuple[str, list[dict[str, Any]]]:
    metrics = pd.read_csv(paths.tables / "metrics_by_station.csv")
    subset = metrics.loc[
        metrics.split.eq("test") & metrics.model.eq("champion")
    ]
    pivot = subset.pivot(
        index="station_code", columns="forecast_hour", values="skill_vs_persistence_pct"
    )
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    positive = 100.0 * np.mean(pivot.to_numpy(float) > 0)
    limit = max(10.0, float(np.nanpercentile(np.abs(pivot.to_numpy(float)), 95)))
    fig, ax = _canvas(
        "Forecast skill varies materially by station and lead",
        f"The selected model improves on persistence in {positive:.0f}% of station–lead combinations; red cells identify operational failure modes requiring monitoring.",
        height=6.2,
    )
    cmap = LinearSegmentedColormap.from_list(
        "skill", ["#B2182B", "#F7F7F7", "#2166AC"]
    )
    image = ax.imshow(
        pivot.to_numpy(float),
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)), [f"+{x} h" for x in pivot.columns])
    ax.set_xlabel("Forecast lead")
    ax.set_ylabel("Station code")
    colorbar = fig.colorbar(image, ax=ax, pad=0.015, fraction=0.03)
    colorbar.set_label("MAE skill vs persistence (%)")
    _footer(
        fig,
        "Blue indicates lower MAE than persistence; red indicates higher MAE. Held-out test period: 1 Jan–31 Aug 2026.",
    )
    return "figure_03_station_skill", _save(fig, paths.root / "figures", "figure_03_station_skill")


def _plot_observed_predicted(paths: ExperimentPaths) -> tuple[str, list[dict[str, Any]]]:
    predictions = pd.read_csv(paths.derived / "validation_test_predictions.csv.gz")
    subset = predictions.loc[
        predictions.split.eq("test")
        & predictions.forecast_hour.isin([24, 72])
    ].dropna(subset=["target_pm25_ug_m3", "champion"])
    upper = float(np.nanpercentile(np.r_[subset.target_pm25_ug_m3, subset.champion], 99.5))
    fig = plt.figure(figsize=(7.15, 4.7))
    fig.suptitle(
        "The selected model captures the central range but compresses some extremes",
        x=0.08,
        y=0.98,
        ha="left",
        va="top",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.08,
        0.90,
        "Hexagonal density reveals the full test distribution; the dashed identity line marks an unbiased point forecast.",
        ha="left",
        va="top",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#EEF6F3", "edgecolor": "none"},
    )
    axes = [fig.add_axes([0.10, 0.17, 0.37, 0.58]), fig.add_axes([0.57, 0.17, 0.37, 0.58])]
    hexbin = None
    for ax, horizon in zip(axes, [24, 72], strict=True):
        data = subset.loc[subset.forecast_hour.eq(horizon)]
        hexbin = ax.hexbin(
            data.target_pm25_ug_m3,
            data.champion,
            gridsize=48,
            mincnt=1,
            bins="log",
            cmap="viridis",
            extent=[0, upper, 0, upper],
        )
        ax.plot([0, upper], [0, upper], linestyle="--", color="#D55E00", linewidth=1.2)
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_title(f"+{horizon} h")
        ax.set_xlabel("Observed PM₂.₅ (µg m⁻³)")
        ax.set_ylabel("Forecast PM₂.₅ (µg m⁻³)")
    if hexbin is not None:
        cbar = fig.colorbar(hexbin, ax=axes, pad=0.02, fraction=0.025)
        cbar.set_label("Case density (log scale)")
    _footer(
        fig,
        "All valid station cases in the independent 2026 test period; axes are limited to the pooled 99.5th percentile for legibility.",
    )
    return "figure_04_observed_vs_predicted", _save(fig, paths.root / "figures", "figure_04_observed_vs_predicted")


def _plot_chronological_comparison(
    paths: ExperimentPaths,
) -> tuple[str, list[dict[str, Any]]]:
    training = pd.read_csv(
        paths.derived / "training_oof_predictions.csv.gz",
        parse_dates=["target_time_utc"],
        low_memory=False,
    )
    later = pd.read_csv(
        paths.derived / "validation_test_predictions.csv.gz",
        parse_dates=["target_time_utc"],
        low_memory=False,
    )
    frames = []
    for label, frame in (
        ("Training: expanding-window out-of-fold", training),
        ("Validation: model selection", later.loc[later.split.eq("validation")]),
        ("Independent test", later.loc[later.split.eq("test")]),
    ):
        subset = frame.loc[frame.forecast_hour.eq(24)].copy()
        subset["date_utc"] = subset.target_time_utc.dt.floor("D")
        daily = (
            subset.groupby("date_utc", as_index=False)
            .agg(
                observed=("target_pm25_ug_m3", "median"),
                model=("champion", "median"),
                persistence=("persistence", "median"),
                stations=("station_code", "nunique"),
            )
            .sort_values("date_utc")
        )
        complete_index = pd.date_range(
            daily.date_utc.min(), daily.date_utc.max(), freq="D", tz="UTC"
        )
        daily = daily.set_index("date_utc").reindex(complete_index)
        frames.append((label, daily))

    fig = plt.figure(figsize=(7.15, 7.0))
    fig.suptitle(
        "Chronological evaluation exposes timing and peak errors hidden by aggregate scores",
        x=0.08,
        y=0.985,
        ha="left",
        va="top",
        fontsize=13.2,
        weight="bold",
    )
    fig.text(
        0.08,
        0.925,
        "Daily station-median +24 h forecasts are shown without smoothing; training-period values are expanding-window out-of-fold predictions, not fitted values.",
        ha="left",
        va="top",
        fontsize=8.8,
        color="#1F2937",
        bbox={"boxstyle": "round,pad=0.42", "facecolor": "#EEF6F3", "edgecolor": "none"},
    )
    axes = [
        fig.add_axes([0.10, 0.67, 0.84, 0.16]),
        fig.add_axes([0.10, 0.405, 0.84, 0.16]),
        fig.add_axes([0.10, 0.14, 0.84, 0.16]),
    ]
    for ax, (label, daily) in zip(axes, frames, strict=True):
        ax.plot(
            daily.index,
            daily.observed,
            color="#111827",
            linewidth=1.35,
            label="Observed",
            zorder=4,
        )
        ax.plot(
            daily.index,
            daily.model,
            color=COLORS["champion"],
            linewidth=1.25,
            label="Selected model",
            zorder=3,
        )
        ax.plot(
            daily.index,
            daily.persistence,
            color=COLORS["persistence"],
            linewidth=0.9,
            linestyle="--",
            alpha=0.85,
            label="Persistence",
            zorder=2,
        )
        ax.set_title(label, loc="left", fontsize=9.3, weight="bold")
        ax.set_ylabel("PM₂.₅\n(µg m⁻³)")
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.5)
        locator = AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    axes[-1].set_xlabel("Target date (UTC)")
    _footer(
        fig,
        "Network median across available stations for the 00 UTC cycle; one target per station-day. Missing calendar days remain gaps. Model family and tree counts were selected using 2025 validation.",
    )
    return "figure_04_chronological_comparison", _save(
        fig,
        paths.root / "figures",
        "figure_04_chronological_comparison",
    )


def _plot_all_station_test_atlas(
    paths: ExperimentPaths,
) -> tuple[str, list[dict[str, Any]]]:
    predictions = pd.read_csv(
        paths.derived / "validation_test_predictions.csv.gz",
        parse_dates=["target_time_utc"],
        low_memory=False,
    )
    predictions = predictions.loc[
        predictions.split.eq("test")
        & predictions.forecast_hour.isin([6, 24, 72])
    ].copy()
    station_order = (
        predictions[["station_code", "station_name"]]
        .drop_duplicates()
        .sort_values(["station_name", "station_code"])
    )
    output = paths.root / "figures" / "supplement_all_station_test_timeseries.pdf"
    page_count = 0
    with PdfPages(output, metadata={"Title": "All-station PM2.5 test time-series atlas"}) as pdf:
        for page_start in range(0, len(station_order), 3):
            page_stations = station_order.iloc[page_start : page_start + 3]
            fig, axes = plt.subplots(
                nrows=len(page_stations),
                ncols=3,
                figsize=(11.69, 8.27),
                squeeze=False,
                sharex=True,
            )
            fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.10, hspace=0.45, wspace=0.25)
            fig.suptitle(
                "Independent-test time series show station and lead-specific behaviour",
                x=0.07,
                y=0.96,
                ha="left",
                fontsize=15,
                weight="bold",
            )
            fig.text(
                0.07,
                0.91,
                "Observed and forecast PM₂.₅ for 1 January–31 August 2026; shaded bands are calibrated 10th–90th percentile prediction intervals.",
                ha="left",
                fontsize=10,
                color="#374151",
            )
            for row_index, station in enumerate(page_stations.itertuples(index=False)):
                for column_index, horizon in enumerate((6, 24, 72)):
                    ax = axes[row_index, column_index]
                    data = predictions.loc[
                        predictions.station_code.eq(station.station_code)
                        & predictions.forecast_hour.eq(horizon)
                    ].sort_values("target_time_utc")
                    ax.fill_between(
                        data.target_time_utc,
                        data.prediction_q10,
                        data.prediction_q90,
                        color=COLORS["interval"],
                        alpha=0.18,
                        linewidth=0,
                        label="80% interval",
                    )
                    ax.plot(data.target_time_utc, data.target_pm25_ug_m3, color="#111827", linewidth=1.0, label="Observed")
                    ax.plot(data.target_time_utc, data.champion, color=COLORS["champion"], linewidth=0.95, label="Selected model")
                    ax.plot(data.target_time_utc, data.persistence, color=COLORS["persistence"], linewidth=0.7, linestyle="--", alpha=0.8, label="Persistence")
                    ax.set_title(f"{station.station_name} ({station.station_code}), +{horizon} h", fontsize=8.8, loc="left")
                    ax.grid(axis="y", color="#D1D5DB", linewidth=0.45)
                    locator = AutoDateLocator(minticks=3, maxticks=5)
                    ax.xaxis.set_major_locator(locator)
                    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
                    if column_index == 0:
                        ax.set_ylabel("PM₂.₅ (µg m⁻³)")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, frameon=False, ncol=4, loc="upper right", bbox_to_anchor=(0.98, 0.895))
            fig.text(
                0.07,
                0.035,
                "One 00 UTC forecast target per station-day. Lines are unsmoothed; missing values are not interpolated. Forecasts are from the validation-selected model on the untouched 2026 test period.",
                fontsize=8,
                color="#4B5563",
            )
            pdf.savefig(fig)
            plt.close(fig)
            page_count += 1
    return "supplement_all_station_test_timeseries", [
        {
            "path": str(output.relative_to(paths.root)),
            "bytes": output.stat().st_size,
            "sha256": file_sha256(output),
            "pages": page_count,
            "stations": len(station_order),
        }
    ]


def _plot_intervals(paths: ExperimentPaths) -> tuple[str, list[dict[str, Any]]]:
    intervals = pd.read_csv(paths.tables / "prediction_interval_metrics.csv")
    test = intervals.loc[intervals.split.eq("test")].sort_values("forecast_hour")
    mean_coverage = float(test.empirical_coverage_pct.mean())
    fig = plt.figure(figsize=(7.15, 4.7))
    fig.suptitle(
        "Prediction intervals quantify uncertainty, with coverage checked independently",
        x=0.08,
        y=0.98,
        ha="left",
        va="top",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.08,
        0.90,
        f"Nominal 80% intervals achieve {mean_coverage:.1f}% mean empirical coverage across leads in the untouched test period.",
        ha="left",
        va="top",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#EEF6F3", "edgecolor": "none"},
    )
    left = fig.add_axes([0.10, 0.17, 0.37, 0.58])
    right = fig.add_axes([0.58, 0.17, 0.36, 0.58])
    left.plot(test.forecast_hour, test.empirical_coverage_pct, marker="o", color="#0072B2", linewidth=2)
    left.axhline(80, color="#D55E00", linestyle="--", linewidth=1.2, label="Nominal 80%")
    left.set_xticks(test.forecast_hour)
    left.set_xlabel("Forecast lead (hours)")
    left.set_ylabel("Empirical coverage (%)")
    left.set_ylim(0, 100)
    left.legend(frameon=False)
    left.grid(axis="y", color="#D1D5DB", linewidth=0.6)
    right.plot(test.forecast_hour, test.mean_interval_width_ug_m3, marker="o", color="#009E73", linewidth=2)
    right.set_xticks(test.forecast_hour)
    right.set_xlabel("Forecast lead (hours)")
    right.set_ylabel("Mean interval width (µg m⁻³)")
    right.grid(axis="y", color="#D1D5DB", linewidth=0.6)
    _footer(
        fig,
        "Intervals are q10–q90 LightGBM estimates calibrated on held July–December 2025 data; coverage is evaluated on 2026 only.",
    )
    return "figure_05_prediction_intervals", _save(fig, paths.root / "figures", "figure_05_prediction_intervals")


def _plot_events(paths: ExperimentPaths) -> tuple[str, list[dict[str, Any]]]:
    events = pd.read_csv(paths.tables / "high_event_detection_metrics.csv").sort_values("forecast_hour")
    csi = float(events.critical_success_index_pct.mean())
    fig, ax = _canvas(
        "High-concentration episodes remain the hardest operational regime",
        f"Mean critical success index is {csi:.1f}% for events defined independently from training-period station 90th percentiles.",
    )
    for column, label, color in (
        ("probability_of_detection_pct", "Probability of detection", "#0072B2"),
        ("false_alarm_ratio_pct", "False-alarm ratio", "#D55E00"),
        ("critical_success_index_pct", "Critical success index", "#009E73"),
    ):
        ax.plot(events.forecast_hour, events[column], marker="o", linewidth=2, label=label, color=color)
    ax.set_xticks(events.forecast_hour)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Forecast lead (hours)")
    ax.set_ylabel("Event metric (%)")
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    _footer(
        fig,
        "Independent test period: 1 Jan–31 Aug 2026. Thresholds use each station's 2023–2024 training distribution only.",
    )
    return "figure_06_high_event_detection", _save(fig, paths.root / "figures", "figure_06_high_event_detection")


def _plot_importance(paths: ExperimentPaths) -> tuple[str, list[dict[str, Any]]]:
    importance = pd.read_csv(paths.tables / "feature_importance.csv")
    grouped = (
        importance.groupby("feature", as_index=False)
        .normalized_importance_pct.mean()
        .nlargest(12, "normalized_importance_pct")
        .sort_values("normalized_importance_pct")
    )
    top = str(grouped.iloc[-1].feature)
    fig, ax = _canvas(
        "Forecast skill is distributed across persistence, CAMS, time, and site context",
        f"{top} has the largest mean tree importance across leads; importance is predictive and must not be read causally.",
        height=5.1,
    )
    ax.barh(grouped.feature, grouped.normalized_importance_pct, color="#0072B2")
    ax.set_xlabel("Mean normalized tree importance across leads (%)")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#D1D5DB", linewidth=0.6)
    _footer(
        fig,
        "Importance is the fitted tree estimator's normalized split/gain measure for the selected model; correlated predictors can share or exchange importance.",
    )
    return "figure_07_feature_importance", _save(fig, paths.root / "figures", "figure_07_feature_importance")


def _plot_transfer(paths: ExperimentPaths) -> tuple[str, list[dict[str, Any]]]:
    transfer = pd.read_csv(paths.tables / "station_transfer_metrics.csv")
    by_lead = transfer.groupby("forecast_hour", as_index=False).agg(
        mae_ug_m3=("mae_ug_m3", "mean"),
        skill_vs_persistence_pct=("skill_vs_persistence_pct", "mean"),
        n_stations=("station_code", "nunique"),
    )
    mean_skill = float(by_lead.skill_vs_persistence_pct.mean())
    fig = plt.figure(figsize=(7.15, 4.7))
    fig.suptitle(
        "Transfer to stations excluded from fitting is weaker than same-network prediction",
        x=0.08,
        y=0.98,
        ha="left",
        va="top",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.08,
        0.90,
        f"Five-fold station holdout yields {mean_skill:+.1f}% mean MAE skill versus persistence across leads, without station identity as a feature.",
        ha="left",
        va="top",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#EEF6F3", "edgecolor": "none"},
    )
    left = fig.add_axes([0.10, 0.17, 0.37, 0.58])
    right = fig.add_axes([0.58, 0.17, 0.36, 0.58])
    left.plot(by_lead.forecast_hour, by_lead.mae_ug_m3, marker="o", color="#0072B2", linewidth=2)
    left.set_xticks(by_lead.forecast_hour)
    left.set_xlabel("Forecast lead (hours)")
    left.set_ylabel("Station-balanced MAE (µg m⁻³)")
    left.grid(axis="y", color="#D1D5DB", linewidth=0.6)
    right.plot(by_lead.forecast_hour, by_lead.skill_vs_persistence_pct, marker="o", color="#009E73", linewidth=2)
    right.axhline(0, color="#6B7280", linewidth=1)
    right.set_xticks(by_lead.forecast_hour)
    right.set_xlabel("Forecast lead (hours)")
    right.set_ylabel("MAE skill vs persistence (%)")
    right.grid(axis="y", color="#D1D5DB", linewidth=0.6)
    _footer(
        fig,
        "Spatial-transfer diagnostic: each station is predicted by a model trained on other stations; test targets are from 2026.",
    )
    return "figure_08_station_transfer", _save(fig, paths.root / "figures", "figure_08_station_transfer")


def _plot_residuals(paths: ExperimentPaths) -> tuple[str, list[dict[str, Any]]]:
    residuals = pd.read_csv(paths.tables / "residual_diagnostics.csv")
    subset = residuals.loc[residuals.dimension.eq("target_month")].copy()
    pivot = subset.pivot(index="forecast_hour", columns="group", values="mean_residual_ug_m3")
    limit = max(2.0, float(np.nanpercentile(np.abs(pivot.to_numpy(float)), 95)))
    fig, ax = _canvas(
        "Residual bias changes with month and forecast lead",
        "Positive cells indicate overprediction and negative cells underprediction; seasonal structure is a monitoring trigger, not proof of a physical cause.",
        height=4.4,
    )
    cmap = LinearSegmentedColormap.from_list("residual", ["#2166AC", "#F7F7F7", "#B2182B"])
    image = ax.imshow(
        pivot.to_numpy(float),
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    ax.set_yticks(np.arange(len(pivot.index)), [f"+{x} h" for x in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)), [str(int(x)) for x in pivot.columns])
    ax.set_xlabel("Target month (UTC-derived target date)")
    ax.set_ylabel("Forecast lead")
    colorbar = fig.colorbar(image, ax=ax, pad=0.015, fraction=0.03)
    colorbar.set_label("Mean forecast minus observation (µg m⁻³)")
    _footer(fig, "Selected-model residuals on the independent 2026 test period; pooled across stations within each month and lead.")
    return "figure_09_residual_bias", _save(fig, paths.root / "figures", "figure_09_residual_bias")


def make_all_figures(paths: ExperimentPaths | None = None) -> dict[str, Any]:
    paths = paths or ExperimentPaths()
    _style()
    makers = [
        _plot_data_quality,
        _plot_performance,
        _plot_station_skill,
        _plot_chronological_comparison,
        _plot_observed_predicted,
        _plot_intervals,
        _plot_events,
        _plot_importance,
        _plot_transfer,
        _plot_residuals,
        _plot_all_station_test_atlas,
    ]
    records = []
    for maker in makers:
        name, outputs = maker(paths)
        records.append({"figure": name, "outputs": outputs})
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "figures": records,
        "format": "SVG plus 300 dpi PNG",
        "style": "colorblind-aware restrained scientific chart standard",
    }
    write_json(paths.provenance / "figure_manifest.json", manifest)
    return {
        "figures": len(records),
        "files": sum(len(record["outputs"]) for record in records),
    }
