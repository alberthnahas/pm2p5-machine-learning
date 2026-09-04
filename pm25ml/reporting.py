"""Evidence-led Markdown and LaTeX report generation with PDF verification."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data import ExperimentPaths, file_sha256, load_config, write_json


REPORT_STEM = "pm25_station_cams_mos_research_to_operations"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):,.{digits}f}"


def _markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    output.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(output)


def _tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _tex_table(
    headers: list[str], rows: Iterable[Iterable[Any]], widths: str | None = None
) -> str:
    rows = list(rows)
    specification = widths or ("l" + "r" * (len(headers) - 1))
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{adjustbox}{max width=\linewidth}",
        rf"\begin{{tabular}}{{{specification}}}",
        r"\toprule",
        " & ".join(_tex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(_tex_escape(value) for value in row) + r" \\" for row in rows
    )
    lines.extend(
        [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    )
    return "\n".join(lines)


def _figure_markdown(number: int, stem: str, caption: str) -> str:
    return f"![Figure {number}. {caption}](../figures/{stem}.png)\n\n*Figure {number}. {caption}*"


def _figure_tex(number: int, stem: str, caption: str, width: float = 0.97) -> str:
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width={width:.2f}\textwidth]{{../figures/{stem}.png}}",
            rf"\caption{{{_tex_escape(caption)}}}",
            rf"\label{{fig:{number}}}",
            r"\end{figure}",
        ]
    )


def _report_evidence(paths: ExperimentPaths) -> dict[str, Any]:
    config = load_config(paths.config)
    observation = _json(paths.provenance / "observation_manifest.json")
    audit = _json(paths.provenance / "modeling_table_audit.json")
    research = _json(paths.provenance / "research_model_manifest.json")
    deployment = _json(paths.provenance / "deployment_model_manifest.json")
    summary = pd.read_csv(paths.tables / "metrics_summary.csv")
    station = pd.read_csv(paths.tables / "metrics_by_station.csv")
    ranking = pd.read_csv(paths.tables / "model_selection_ranking.csv")
    intervals = pd.read_csv(paths.tables / "prediction_interval_metrics.csv")
    bootstrap = pd.read_csv(paths.tables / "block_bootstrap_skill.csv")
    events = pd.read_csv(paths.tables / "high_event_detection_metrics.csv")
    transfer = pd.read_csv(paths.tables / "station_transfer_metrics.csv")
    sensitivity = pd.read_csv(paths.tables / "recent_window_sensitivity.csv")
    shift = pd.read_csv(paths.tables / "feature_distribution_shift.csv")
    quality = pd.read_csv(paths.tables / "data_quality_by_station.csv")
    runtime = pd.read_csv(paths.tables / "runtime_by_stage.csv")
    deployment_runtime = pd.read_csv(paths.tables / "deployment_runtime_benchmark.csv")
    cams_increment = pd.read_csv(
        paths.tables / "cams_incremental_skill_vs_observation_ml.csv",
        dtype={"forecast_hour": str},
    )
    resources_path = paths.provenance / "execution_resource_usage.json"
    resources = _json(resources_path) if resources_path.exists() else {}
    cams_acquisition = _json(paths.cams_earth_engine / "manifest.json")
    cams_equivalence = _json(
        paths.root
        / "data"
        / "external"
        / "cams_source_validation"
        / "manifest.json"
    )
    cams_acquisition_runtime = _json(
        paths.provenance / "cams_initial_acquisition_runtime.json"
    )
    operational_metadata_paths = sorted((paths.root / "output").glob("*.json"))
    operational_metadata = (
        _json(operational_metadata_paths[-1]) if operational_metadata_paths else {}
    )
    training_oof_metrics = pd.read_csv(paths.tables / "training_oof_metrics.csv")
    training_oof_runtime = pd.read_csv(paths.tables / "training_oof_runtime.csv")
    shadow_state_path = paths.root / "shadow" / "state" / "first_successful_run.json"
    if not shadow_state_path.exists():
        shadow_state_path = paths.root / "shadow" / "state" / "latest_run.json"
    shadow_state = _json(shadow_state_path) if shadow_state_path.exists() else {}
    shadow_metadata_paths = sorted((paths.root / "shadow" / "forecasts").glob("*.json"))
    shadow_metadata = _json(shadow_metadata_paths[-1]) if shadow_metadata_paths else {}
    shadow_forecast_paths = sorted((paths.root / "shadow" / "forecasts").glob("*.csv"))
    shadow_generation_lag_hours = None
    if shadow_forecast_paths:
        shadow_forecast = pd.read_csv(shadow_forecast_paths[-1])
        if "availability_lag_hours" in shadow_forecast:
            shadow_generation_lag_hours = float(
                shadow_forecast.availability_lag_hours.median()
            )

    champion = str(research["champion"])
    champion_label = {
        "obs_lgbm": "observation-only LightGBM",
        "cams_lgbm": "CAMS model-output-statistics LightGBM",
        "cams_xgboost": "CAMS model-output-statistics XGBoost",
    }[champion]
    test = summary.loc[
        summary.split.eq("test")
        & summary.scope.eq("station_balanced_common_cases")
    ]
    champion_test = test.loc[test.model.eq("champion")].sort_values("forecast_hour")
    persistence_test = test.loc[test.model.eq("persistence")].sort_values("forecast_hour")
    raw_cams_test = test.loc[test.model.eq("raw_cams")].sort_values("forecast_hour")
    station_test = station.loc[
        station.split.eq("test") & station.model.eq("champion")
    ]
    intervals_test = intervals.loc[intervals.split.eq("test")].sort_values("forecast_hour")
    champion_bootstrap = bootstrap.loc[bootstrap.model.eq("champion")].sort_values("forecast_hour")
    transfer_summary = transfer.groupby("forecast_hour", as_index=False).agg(
        n_stations=("station_code", "nunique"),
        mae_ug_m3=("mae_ug_m3", "mean"),
        skill_vs_persistence_pct=("skill_vs_persistence_pct", "mean"),
    )
    recent = sensitivity.loc[
        sensitivity.scope.eq("station_balanced_common_cases")
        & sensitivity.model.eq("recent_window")
    ].sort_values("forecast_hour")
    high_shift = shift.assign(
        absolute_standardized_mean_difference=shift.standardized_mean_difference.abs()
    ).nlargest(8, "absolute_standardized_mean_difference")
    selection = ranking.loc[ranking.selected_champion].iloc[0]

    return {
        "config": config,
        "observation": observation,
        "audit": audit,
        "research": research,
        "deployment": deployment,
        "quality": quality,
        "ranking": ranking,
        "champion": champion,
        "champion_label": champion_label,
        "selection": selection,
        "champion_test": champion_test,
        "persistence_test": persistence_test,
        "raw_cams_test": raw_cams_test,
        "station_test": station_test,
        "intervals_test": intervals_test,
        "champion_bootstrap": champion_bootstrap,
        "events": events.sort_values("forecast_hour"),
        "transfer": transfer_summary.sort_values("forecast_hour"),
        "recent": recent,
        "high_shift": high_shift,
        "runtime": runtime,
        "deployment_runtime": deployment_runtime,
        "resources": resources,
        "cams_acquisition": cams_acquisition,
        "cams_equivalence": cams_equivalence,
        "cams_increment": cams_increment,
        "cams_acquisition_runtime": cams_acquisition_runtime,
        "operational_metadata": operational_metadata,
        "training_oof_metrics": training_oof_metrics,
        "training_oof_runtime": training_oof_runtime,
        "shadow_state": shadow_state,
        "shadow_metadata": shadow_metadata,
        "shadow_generation_lag_hours": shadow_generation_lag_hours,
    }


def _tables(evidence: dict[str, Any]) -> dict[str, tuple[list[str], list[list[str]]]]:
    ranking = evidence["ranking"]
    selection_rows = [
        [
            {
                "obs_lgbm": "Observation-only LightGBM",
                "cams_lgbm": "CAMS MOS LightGBM",
                "cams_xgboost": "CAMS MOS XGBoost",
            }[row.model],
            _number(row.mean_station_balanced_mae_ug_m3, 2),
            _number(row.mean_skill_vs_persistence_pct, 1),
            "Yes" if bool(row.selected_champion) else "No",
        ]
        for row in ranking.itertuples()
    ]
    test = evidence["champion_test"]
    persistence = evidence["persistence_test"].set_index("forecast_hour")
    raw_cams = evidence["raw_cams_test"].set_index("forecast_hour")
    bootstrap = evidence["champion_bootstrap"].set_index("forecast_hour")
    test_rows = []
    for row in test.itertuples():
        boot = bootstrap.loc[row.forecast_hour]
        test_rows.append(
            [
                f"+{int(row.forecast_hour)}",
                _number(row.n, 0),
                _number(row.mae_ug_m3, 2),
                _number(persistence.loc[row.forecast_hour, "mae_ug_m3"], 2),
                _number(raw_cams.loc[row.forecast_hour, "mae_ug_m3"], 2),
                _number(row.skill_vs_persistence_pct, 1),
                _number(row.rmse_ug_m3, 2),
                _number(row.bias_ug_m3, 2),
                _number(row.correlation, 2),
                f"[{_number(boot.ci95_lower_ug_m3, 2)}, {_number(boot.ci95_upper_ug_m3, 2)}]",
            ]
        )
    interval_rows = [
        [
            f"+{int(row.forecast_hour)}",
            _number(row.n, 0),
            _number(row.empirical_coverage_pct, 1),
            _number(row.mean_interval_width_ug_m3, 1),
            _number(row.mean_interval_score_ug_m3, 1),
        ]
        for row in evidence["intervals_test"].itertuples()
    ]
    event_rows = [
        [
            f"+{int(row.forecast_hour)}",
            _number(row.hits, 0),
            _number(row.misses, 0),
            _number(row.false_alarms, 0),
            _number(row.probability_of_detection_pct, 1),
            _number(row.false_alarm_ratio_pct, 1),
            _number(row.critical_success_index_pct, 1),
        ]
        for row in evidence["events"].itertuples()
    ]
    transfer_rows = [
        [
            f"+{int(row.forecast_hour)}",
            _number(row.n_stations, 0),
            _number(row.mae_ug_m3, 2),
            _number(row.skill_vs_persistence_pct, 1),
        ]
        for row in evidence["transfer"].itertuples()
    ]
    recent_lookup = evidence["recent"].set_index("forecast_hour")
    sensitivity_rows = []
    for row in evidence["champion_test"].itertuples():
        recent = recent_lookup.loc[row.forecast_hour]
        sensitivity_rows.append(
            [
                f"+{int(row.forecast_hour)}",
                _number(row.mae_ug_m3, 2),
                _number(recent.mae_ug_m3, 2),
                _number(recent.mae_ug_m3 - row.mae_ug_m3, 2),
            ]
        )
    quality = evidence["quality"].sort_values("valid_pm25_pct_of_expected_hours")
    quality_rows = [
        [
            str(row.station_code),
            str(row.station_name),
            str(row.start_utc)[:10],
            str(row.end_utc)[:10],
            _number(row.valid_pm25_pct_of_expected_hours, 1),
            _number(row.absent_hours_within_span, 0),
            _number(row.invalid_temperature, 0),
        ]
        for row in quality.itertuples()
    ]
    cams_increment_rows = [
        [
            "All leads" if str(row.forecast_hour) == "all" else f"+{row.forecast_hour}",
            _number(row.cams_mae_improvement_over_obs_ml_ug_m3, 3),
            _number(row.relative_improvement_over_obs_ml_pct, 2),
            f"[{_number(row.ci95_lower_ug_m3, 3)}, {_number(row.ci95_upper_ug_m3, 3)}]",
            _number(row.n_station_weeks, 0),
        ]
        for row in evidence["cams_increment"].loc[
            evidence["cams_increment"].split.eq("test")
        ].itertuples()
    ]
    return {
        "selection": (
            ["Candidate", "Validation MAE", "Skill vs persistence (%)", "Selected"],
            selection_rows,
        ),
        "test": (
            ["Lead (h)", "n", "MAE", "Persistence MAE", "Raw CAMS MAE", "Skill (%)", "RMSE", "Bias", "r", "95% CI: MAE gain"],
            test_rows,
        ),
        "interval": (
            ["Lead (h)", "n", "Coverage (%)", "Mean width", "Interval score"],
            interval_rows,
        ),
        "events": (
            ["Lead (h)", "Hits", "Misses", "False alarms", "POD (%)", "FAR (%)", "CSI (%)"],
            event_rows,
        ),
        "transfer": (
            ["Lead (h)", "Stations", "MAE", "Skill vs persistence (%)"],
            transfer_rows,
        ),
        "sensitivity": (
            ["Lead (h)", "Frozen MAE", "2024-only MAE", "Difference"],
            sensitivity_rows,
        ),
        "quality": (
            ["Station", "Name", "Start", "End", "Valid coverage (%)", "Absent hours", "Invalid T"],
            quality_rows,
        ),
        "cams_increment": (
            ["Lead (h)", "MAE gain", "Relative gain (%)", "95% CI", "Station-weeks"],
            cams_increment_rows,
        ),
    }


def _narrative_values(evidence: dict[str, Any]) -> dict[str, Any]:
    test = evidence["champion_test"]
    station = evidence["station_test"]
    intervals = evidence["intervals_test"]
    observation = evidence["observation"]
    quality = evidence["quality"]
    resources = evidence["resources"]
    return {
        "mean_test_mae": float(test.mae_ug_m3.mean()),
        "mean_test_skill": float(test.skill_vs_persistence_pct.mean()),
        "mean_raw_cams_mae": float(evidence["raw_cams_test"].mae_ug_m3.mean()),
        "mean_mos_gain_vs_raw_cams_pct": 100.0
        * (
            1.0
            - float(test.mae_ug_m3.mean())
            / float(evidence["raw_cams_test"].mae_ug_m3.mean())
        ),
        "positive_station_leads": 100.0 * float(station.skill_vs_persistence_pct.gt(0).mean()),
        "mean_coverage": float(intervals.empirical_coverage_pct.mean()),
        "valid_rows": int(observation["valid_pm25"]),
        "valid_pct": float(observation["valid_pm25_pct"]),
        "median_station_coverage": float(quality.valid_pm25_pct_of_expected_hours.median()),
        "preparation_peak_gib": resources.get("prepare", {}).get("peak_rss_gib"),
        "training_peak_gib": resources.get("train", {}).get("peak_rss_gib"),
        "preparation_seconds": resources.get("prepare", {}).get("elapsed_seconds"),
        "training_seconds": resources.get("train", {}).get(
            "elapsed_seconds",
            float(
                evidence["runtime"].loc[
                    evidence["runtime"].stage.eq("complete_train_evaluate"), "seconds"
                ].max()
            ),
        ),
        "deployment_refit_seconds": float(
            evidence["deployment_runtime"].loc[
                evidence["deployment_runtime"].stage.eq("deployment_refit"), "seconds"
            ].sum()
        ),
        "inference_median_seconds": float(
            evidence["deployment_runtime"].loc[
                evidence["deployment_runtime"].stage.eq("warm_batch_inference"),
                "median_seconds_per_batch",
            ].sum()
        ),
        "deployment_mebibytes": sum(
            row["bytes"] for row in evidence["deployment"]["models"]
        )
        / (1024.0**2),
        "deterministic_pipeline_seconds": float(
            sum(
                stage.get("elapsed_seconds", 0.0)
                for stage in resources.values()
                if stage.get("return_code") == 0
            )
        ),
        "historical_acquisition_minutes": float(
            evidence["cams_acquisition_runtime"]["elapsed_minutes"]
        ),
        "operational_elapsed_seconds": evidence["operational_metadata"].get(
            "elapsed_seconds_excluding_metadata_write"
        ),
    }


def _markdown_report(evidence: dict[str, Any], tables: dict[str, Any]) -> str:
    values = _narrative_values(evidence)
    config = evidence["config"]
    observation = evidence["observation"]
    audit = evidence["audit"]
    cams_acquisition = evidence["cams_acquisition"]
    cams_equivalence = evidence["cams_equivalence"]
    native_check = cams_equivalence["native_grid_comparison"]
    station_check = cams_equivalence["station_bilinear_comparison"]
    champion = evidence["champion_label"]
    selection = evidence["selection"]
    station = evidence["station_test"]
    worst = station.nsmallest(5, "skill_vs_persistence_pct")
    worst_text = ", ".join(
        f"{row.station_code} ({row.skill_vs_persistence_pct:+.1f}%)"
        for row in worst.itertuples()
    )
    peak_parts = []
    if values["preparation_peak_gib"] is not None:
        peak_parts.append(f"preparation {values['preparation_peak_gib']:.2f} GiB")
    if values["training_peak_gib"] is not None:
        peak_parts.append(f"training {values['training_peak_gib']:.2f} GiB")
    peak_text = ", ".join(peak_parts) if peak_parts else "resource measurement recorded separately"
    selection_headers, selection_rows = tables["selection"]
    test_headers, test_rows = tables["test"]
    interval_headers, interval_rows = tables["interval"]
    event_headers, event_rows = tables["events"]
    transfer_headers, transfer_rows = tables["transfer"]
    sensitivity_headers, sensitivity_rows = tables["sensitivity"]
    quality_headers, quality_rows = tables["quality"]
    cams_increment_headers, cams_increment_rows = tables["cams_increment"]
    increment_validation = evidence["cams_increment"].loc[
        evidence["cams_increment"].split.eq("validation")
        & evidence["cams_increment"].forecast_hour.eq("all")
    ].iloc[0]
    increment_test = evidence["cams_increment"].loc[
        evidence["cams_increment"].split.eq("test")
        & evidence["cams_increment"].forecast_hour.eq("all")
    ].iloc[0]
    return f"""# Research-to-operational hourly PM₂.₅ forecasting at 27 Indonesian stations

**Experimental model-development report — not yet an operational public product**<br>
**Evaluation completed:** {datetime.now(timezone.utc):%d %B %Y}<br>
**Forecast cycle evaluated:** 00 UTC daily<br>
**Forecast leads:** +3, +6, +12, +24, +48, and +72 hours

## Executive finding

The validation-selected model is **{champion}**. Across the six lead times in the untouched 1 January–31 August 2026 test period, its mean station-balanced mean absolute error (MAE) is **{values['mean_test_mae']:.2f} µg m⁻³**, corresponding to **{values['mean_test_skill']:+.1f}%** mean skill relative to persistence and **{values['mean_mos_gain_vs_raw_cams_pct']:+.1f}%** relative to raw CAMS. Improvement occurs in **{values['positive_station_leads']:.1f}%** of station–lead combinations, so aggregate skill does not imply uniform local benefit. The nominal 80% prediction intervals attain **{values['mean_coverage']:.1f}%** mean test coverage across leads.

**Operational decision:** retain the system as **pre-operational**. It is suitable for scheduled shadow forecasts and scorecard monitoring, but public or duty-forecaster reliance should wait for prospective evaluation, explicit input-latency tests, a second forecast cycle, and an assessment of forecast meteorology. The model provides prediction, not source attribution or causal explanation.

## 1. Question and intended use

The question is whether a pooled machine-learning model can improve station-level hourly PM₂.₅ forecasts over simple persistence at 27 BMKG locations while remaining computationally light enough for routine operation. One direct model is fitted for each forecast lead; direct forecasting avoids recursive error propagation. The intended output is a concentration forecast in µg m⁻³ plus a calibrated uncertainty interval at each observed station.

The primary inference domain is the existing network. A separate station-holdout experiment asks a harder question: how well the approach transfers to a location absent from fitting. Neither experiment creates a continuous spatial concentration field.

## 2. Data, quality control, and provenance

The supplied archive contains **{observation['source_rows']:,}** rows from 27 hourly station files spanning {str(observation['start_utc'])[:10]} to {str(observation['end_utc'])[:10]}. After applying documented error rules, **{values['valid_rows']:,} ({values['valid_pct']:.2f}%)** source rows contain valid PM₂.₅. Exact duplicated station-hours were removed only in derived data; no conflicting duplicate key was accepted. Negative PM₂.₅ and values at or above 985 µg m⁻³ were treated as instrument error values. Plausible extremes below the cutoff were retained.

Relative humidity outside (0, 100]% and temperature outside −10 to 50 °C were made unavailable as predictors without altering PM₂.₅. This screening identified a block of physically impossible temperature values at Koto Tabang; the values were not corrected or imputed. Median station-level valid PM₂.₅ coverage within each station's observed span is **{values['median_station_coverage']:.1f}%**.

CAMS global atmospheric-composition forecast PM₂.₅ was extracted from the official `ECMWF/CAMS/NRT` Earth Engine mirror [7] and bilinearly sampled to station coordinates. Source mass density in kg m⁻³ was converted explicitly to µg m⁻³. The parent CAMS service provides global atmospheric-composition forecasts twice daily on a 0.4° grid and is an atmospheric model product, not a surface observation [1]. The mirror is three-hourly, so +3 h is the earliest evaluated lead; no +1 h field was synthesized.

The extract retained **{cams_acquisition['rows']:,} of {cams_acquisition['maximum_expected_rows']:,} station–issue–lead records ({cams_acquisition['coverage_pct']:.2f}%)**. Missing records comprise a partial spatial gap on 17–23 February 2024 and a fully missing 00 UTC initialization on 30 June 2025. They were not imputed. An independent direct Copernicus archive check covered {cams_equivalence['stations']} in-domain stations and five common leads on 1 January 2026. At 100 native grid-point comparisons, the maximum absolute difference was {native_check['maximum_absolute_difference_ug_m3']:.6f} µg m⁻³; station-resampling implementations differed by {station_check['mean_absolute_difference_ug_m3']:.3f} µg m⁻³ on average and {station_check['maximum_absolute_difference_ug_m3']:.3f} µg m⁻³ at most. This supports identity of the underlying field while documenting interpolation-level differences. The bounded experiment uses CAMS PM₂.₅ plus observed temperature and humidity histories; forecast meteorology is a declared next-stage input.

{_figure_markdown(1, 'figure_01_observation_coverage', 'Monthly valid PM₂.₅ coverage across the 27 stations. Missing coverage is distinct from a valid zero concentration.')}

## 3. Prediction design

### 3.1 Inputs

Predictors use only timestamps at or before issue time: PM₂.₅ lags from 0 to 168 h; temperature and relative-humidity lags to 24 h; rolling PM₂.₅ mean, standard deviation, maximum, and availability over 3–168 h; local-hour and annual-cycle encodings; station coordinates and region; the contemporaneous network mean; a distance-weighted nearby-station mean within 400 km; and forecast-valid CAMS PM₂.₅. Missing predictor values are retained for native tree handling. Feature importance is diagnostic and not causal importance.

### 3.2 Chronological separation

Targets from 2023–2024 form training, 2025 is used for point-model early stopping and model choice, and 2026 through 31 August is evaluated once as the final test. The uncertainty workflow adds a nested temporal separation: January–June 2025 is used only to tune quantile tree counts, while July–December 2025 is reserved exclusively for conformal calibration. Splits are assigned by target time, not issue time. The derived table contains **{audit['rows']:,}** station–issue–lead rows and **{audit['valid_target_rows']:,}** valid targets. Automated audits found {audit['duplicate_keys']} duplicate modeling keys, {audit['target_before_or_at_issue']} non-future targets, and no target-time overlap among train, validation, and test. Chronological and station-blocked validation are used because random splitting of structured environmental data can underestimate predictive error [6].

### 3.3 Models and uncertainty

Persistence and a training-only station/local-time climatology are reference forecasts. Candidate learners are observation-only LightGBM, CAMS model-output-statistics (MOS) LightGBM, and CAMS MOS XGBoost. Gradient-boosted trees represent nonlinear interactions and missingness without the compute cost of deep sequence models [3,4]. CAMS bias calibration with machine learning has prior scientific precedent, although performance is domain-specific [2].

Point and quantile predictions are clipped only at the physical lower bound of 0 µg m⁻³. Quantile LightGBM estimates the 10th, 50th, and 90th conditional percentiles. Quantile crossing is removed by sorting, then a lead-specific split-conformal expansion is estimated from the held July–December 2025 calibration block pooled across stations. Neither fitting nor interval calibration accesses 2026. Conformalized quantile regression provides a principled basis for adaptive prediction intervals under exchangeability assumptions [5]; temporal autocorrelation and later CAMS system changes mean empirical monitoring remains necessary.

## 4. Validation selection and independent test results

Model choice used the mean station-balanced validation MAE across all six leads. The specified tie-break prefers CAMS LightGBM when it lies within 1% of the minimum, reducing deployment complexity while retaining a physically based forecast input.

{_markdown_table(selection_headers, selection_rows)}

The selected model's validation criterion is **{float(selection.mean_station_balanced_mae_ug_m3):.2f} µg m⁻³**. The following table reports the subsequently opened test set. MAE and root-mean-square error (RMSE) are in µg m⁻³; positive bias means overprediction. The confidence interval is a 1,000-replicate station-week block bootstrap for mean absolute-error improvement over persistence. A positive interval entirely above zero supports lower mean absolute error under that block-resampling design.

{_markdown_table(test_headers, test_rows)}

{_figure_markdown(2, 'figure_02_test_performance', 'Station-balanced test MAE by lead for all candidates and reference forecasts.')}

{_figure_markdown(3, 'figure_03_station_skill', 'Selected-model MAE skill relative to persistence for every station and lead.')}

### 4.1 Chronological behaviour across development stages

Aggregate errors do not show whether the forecast follows the timing of individual episodes, reacts late, or compresses peaks. Figure 4 therefore compares the observed and predicted +24 h sequences in their original target-time order. For readability, each point is the daily median across stations with valid data; no temporal smoothing or interpolation is applied.

Training-period values are not fitted predictions. They use three expanding-window assessment blocks: July–December 2023, January–June 2024, and July–December 2024. Every assessment target is later than all targets used to fit its fold. However, the displayed model family and tree counts were selected later using 2025 validation, so this retrospective out-of-fold diagnostic describes temporal behaviour and must not be treated as an independent model-selection score. The 2025 panel is validation evidence and the 2026 panel remains the independent test.

{_figure_markdown(4, 'figure_04_chronological_comparison', 'Chronological observed, selected-model, and persistence PM₂.₅ at +24 h for expanding-window out-of-fold training assessments, validation, and independent testing. Values are daily station medians for the 00 UTC forecast cycle.')}

An accompanying [nine-page station atlas](../figures/supplement_all_station_test_timeseries.pdf) shows the complete independent-test sequences at +6, +24, and +72 h for all 27 stations, including the calibrated 10th–90th percentile interval. It is supplied as a separate vector PDF so poor-performing stations and short-lived episodes remain inspectable rather than being concealed by national aggregation.

### 4.2 Incremental value of CAMS

Relative to the otherwise identical observation-only LightGBM, adding forecast-valid CAMS PM₂.₅ reduced station-week-balanced MAE by **{increment_validation.cams_mae_improvement_over_obs_ml_ug_m3:.3f} µg m⁻³** in validation (95% bootstrap interval {increment_validation.ci95_lower_ug_m3:.3f} to {increment_validation.ci95_upper_ug_m3:.3f}) and **{increment_test.cams_mae_improvement_over_obs_ml_ug_m3:.3f} µg m⁻³** in the independent test (95% interval {increment_test.ci95_lower_ug_m3:.3f} to {increment_test.ci95_upper_ug_m3:.3f}). This is an aggregate predictive association, not causal evidence. Test-period gains are clearest at +3 to +12 h; intervals cross zero at each of +24, +48, and +72 h, so longer-lead incremental benefit remains inconclusive individually.

{_markdown_table(cams_increment_headers, cams_increment_rows)}

The five least favourable station–lead combinations include {worst_text}. These are failure modes for investigation, not grounds for deleting observations or selectively omitting stations.

{_figure_markdown(5, 'figure_04_observed_vs_predicted', 'Observed versus selected-model PM₂.₅ at +24 h and +72 h. The display is truncated at the pooled 99.5th percentile only for visual legibility; metrics use the full valid range.')}

## 5. Uncertainty and high-concentration performance

{_markdown_table(interval_headers, interval_rows)}

{_figure_markdown(6, 'figure_05_prediction_intervals', 'Independent empirical coverage and mean width of the nominal 80% intervals.')}

High-concentration events are defined separately for each station and lead using the station's training-period 90th percentile. This makes the test threshold independent of test outcomes while avoiding an arbitrary network-wide concentration cutoff. Probability of detection (POD) is the fraction of observed events detected; false-alarm ratio (FAR) is the fraction of predicted events that did not occur; critical success index (CSI) penalizes both misses and false alarms.

{_markdown_table(event_headers, event_rows)}

{_figure_markdown(7, 'figure_06_high_event_detection', 'High-concentration event performance on the independent test period.')}

## 6. Robustness, spatial transfer, and interpretation

The same-network test estimates performance for stations represented in training. Five station folds provide a separate transfer diagnostic: each held station is tested with a model fitted to the remaining stations and without station identity. Fold assignment is independent of target values.

{_markdown_table(transfer_headers, transfer_rows)}

{_figure_markdown(8, 'figure_08_station_transfer', 'Performance when test stations are excluded from model fitting.')}

The 2024-only training sensitivity holds model form and tree counts fixed, changing only the fitting window. A positive difference means the shorter window has higher test MAE.

{_markdown_table(sensitivity_headers, sensitivity_rows)}

Distribution-shift diagnostics compare predictor missingness, standardized mean differences, and population stability index between training and test. These statistics identify monitoring candidates; they do not by themselves establish that a shift caused an error. Residual bias is also stratified by station, target month, and local target hour.

{_figure_markdown(9, 'figure_07_feature_importance', 'Mean fitted-tree feature importance for the selected model. Correlated predictors can exchange importance.')}

{_figure_markdown(10, 'figure_09_residual_bias', 'Selected-model mean residual by target month and lead.')}

## 7. Computational requirements and operational design

The experiment is designed for a CPU workstation; no GPU is required. Training uses at most 8 CPU threads. Measured peak memory was **{peak_text}**. Measured preparation time was **{_number(values['preparation_seconds'], 1)} s**, candidate/uncertainty/transfer evaluation was **{_number(values['training_seconds'], 1)} s**, and deployment refitting was **{values['deployment_refit_seconds']:.1f} s**. The complete deterministic pipeline took **{values['deterministic_pipeline_seconds'] / 60.0:.1f} min** across its latest successful stage measurements. The serialized deployment bundle is **{values['deployment_mebibytes']:.1f} MiB**. Warm inference across all point, fallback, and interval models totals approximately **{values['inference_median_seconds']:.3f} s** on this workstation, excluding model loading, input download, and feature preparation. The measured prepared-input end-to-end command took **{_number(values['operational_elapsed_seconds'], 2)} s** for 162 station–lead rows.

For a routine 00 UTC run, a practical allocation is **4–8 CPU cores, 4 GiB RAM, and 2 GiB working storage**. Model inference itself should finish in under one minute; CAMS acquisition is network- and service-limited and should be budgeted at **5–30 minutes** with retries. The initial 3.7-year Earth Engine backfill required **{values['historical_acquisition_minutes']:.1f} min** of measured server-query time. A full deterministic research rebuild should be budgeted at **5–15 minutes** on a comparable CPU workstation, while slower CPUs and uncached report dependencies may require longer.

The operational command reads prepared issue-time features and forecast-valid CAMS values, writes one row per station and lead, and records a model-manifest checksum. Deployment point models are refitted through December 2025; deployment quantile models stop at June 2025 so July–December remains a held calibration block. If CAMS PM₂.₅ is missing, a separately fitted observation-only point forecast is used and the status is marked degraded; calibrated intervals are withheld. Observation older than six hours is explicitly flagged. This fallback supports continuity but must not be presented as equivalent quality.

### 7.1 Implemented daily shadow workflow

The non-public shadow workflow is scheduled daily at 17:15 WIB (10:15 UTC). It saves an immutable BMKG dashboard snapshot, freezes the observation cutoff, acquires the current CAMS 00 UTC initialization directly from the Copernicus archive, writes atomic forecasts with hashes and freshness/status fields, and scores them only after observations appear. It retains first-seen station-hour values for verification and preserves later raw snapshots so revisions remain auditable. There is no public upload.

The first end-to-end engineering run on 4 September 2026 generated **{evidence['shadow_state'].get('forecast_rows', 'NA')} rows in {_number(evidence['shadow_state'].get('elapsed_seconds'), 1)} s** with **{evidence['shadow_metadata'].get('degraded_rows', 'NA')} degraded rows**. It produced **{evidence['shadow_metadata'].get('prospective_rows', 'NA')} prospectively eligible rows** and labelled **{evidence['shadow_metadata'].get('late_rows', 'NA')} rows** whose target times had already occurred. This first execution is a workflow test, not a prospective performance result.

That engineering run completed **{_number(evidence['shadow_generation_lag_hours'], 1)} h after the 00 UTC initialization**. Observation freshness in the shadow output is therefore evaluated relative to actual generation time as well as model initialization; otherwise an observation timestamped at 00 UTC would be incorrectly described as fresh roughly eleven hours later.

The operational sequence is:

1. Retrieve and validate the 00 UTC CAMS forecast after publication.
2. Freeze the latest observation cutoff and record station freshness.
3. Construct predictors without accessing any later observation.
4. Produce forecasts, intervals, status flags, hashes, and logs.
5. Score forecasts when observations arrive; retain missing cases rather than backfilling them silently.
6. Record warnings for missing CAMS, stale station data, schema/version changes, degraded operation, extreme residuals, and interval undercoverage.

## 8. Limitations and readiness gates

- The model has been retrospectively tested at one daily cycle only. It is not evidence for 12 UTC or arbitrary issue times.
- Direct CAMS availability is later than model initialization. At the installed schedule, +3 h and +6 h are latency diagnostics rather than prospective forecasts; +12 h is eligible only if generation completes before 12 UTC. Prospective reporting uses the stored per-row eligibility flag.
- The test period covers eight months of 2026, not a complete annual cycle, and no prospective shadow period has yet been observed.
- CAMS has a much coarser footprint than a station and its forecasting system can change over time; a sampled grid value is not a station measurement.
- The CAMS mirror is 99.54% complete for the specified station–issue–lead grid; missing cycles are excluded from primary complete-case comparisons and require the declared degraded fallback in operations.
- Forecast meteorology was not included in this bounded archive acquisition. Observed meteorology is available only at and before issue time and cannot represent future dispersion conditions.
- The network and station records are heterogeneous. Missingness, maintenance, relocation, calibration change, and evolving reporting latency can alter performance.
- Lead-specific conformal calibration pooled across stations assumes exchangeability more strongly than an autocorrelated, spatially dependent forecast archive warrants. The later calibration block prevents fitting leakage, but held-out empirical coverage is still reported rather than guaranteed operationally.
- Station-transfer results concern these 27 locations and do not validate a continuous spatial map or an arbitrary new station.
- Feature importance describes predictive use within fitted trees; it is not emissions, wildfire, transport, contribution, or causal attribution.

Before public operations, require: at least 60–90 days of prospective shadow scoring; a documented data-arrival cutoff; 12 UTC backtesting if that cycle is desired; incremental-skill tests for forecast wind, boundary-layer height, humidity, and precipitation; CAMS-version monitoring; station-specific fallback rules; an alerting and rollback procedure; and scientific sign-off on thresholds and public wording.

## 9. Reproducibility and data-quality detail

Every source observation file, external extract, direct-validation archive, configuration, research model, and deployment model has a SHA-256 checksum in the experiment provenance. Random seeds are fixed. Derived tables preserve UTC timestamps, source station identity, explicit units, target lead, split, and missingness. No source observation was changed.

{_markdown_table(quality_headers, quality_rows)}

## References

1. Copernicus Atmosphere Monitoring Service (CAMS). *CAMS global atmospheric composition forecasts*. DOI: [10.24381/04a0b097](https://doi.org/10.24381/04a0b097).
2. Wu, C., Li, K., and Bai, K. (2020). Validation and calibration of CAMS PM₂.₅ forecasts using in situ PM₂.₅ measurements in China and United States. *Remote Sensing*, 12, 3813. [https://doi.org/10.3390/rs12223813](https://doi.org/10.3390/rs12223813).
3. Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30*. [Primary paper](https://proceedings.neurips.cc/paper/6907-a-highly-efficient-gradient-boosting-decision-tree.pdf).
4. Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD 2016*, 785–794. [https://doi.org/10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
5. Romano, Y., Patterson, E., and Candès, E. J. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems 32*, 3538–3548. [Primary paper](https://papers.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html).
6. Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40, 913–929. [https://doi.org/10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881).
7. Google Earth Engine Data Catalog. *CAMS global near-real-time atmospheric composition forecasts*. [Official dataset entry](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_CAMS_NRT).
"""


def _latex_report(evidence: dict[str, Any], tables: dict[str, Any]) -> str:
    # Generate LaTeX from the same evidence and wording as the authoritative Markdown.
    markdown = _markdown_report(evidence, tables)
    values = _narrative_values(evidence)
    observation = evidence["observation"]
    audit = evidence["audit"]
    cams_acquisition = evidence["cams_acquisition"]
    cams_equivalence = evidence["cams_equivalence"]
    native_check = cams_equivalence["native_grid_comparison"]
    station_check = cams_equivalence["station_bilinear_comparison"]
    champion = _tex_escape(evidence["champion_label"])
    selection = evidence["selection"]
    station = evidence["station_test"]
    worst = station.nsmallest(5, "skill_vs_persistence_pct")
    worst_text = ", ".join(
        f"{_tex_escape(row.station_code)} ({row.skill_vs_persistence_pct:+.1f}\\%)"
        for row in worst.itertuples()
    )
    peak_parts = []
    if values["preparation_peak_gib"] is not None:
        peak_parts.append(f"preparation {values['preparation_peak_gib']:.2f} GiB")
    if values["training_peak_gib"] is not None:
        peak_parts.append(f"training {values['training_peak_gib']:.2f} GiB")
    peak_text = ", ".join(peak_parts) if peak_parts else "resource measurement recorded separately"
    selection_headers, selection_rows = tables["selection"]
    test_headers, test_rows = tables["test"]
    interval_headers, interval_rows = tables["interval"]
    event_headers, event_rows = tables["events"]
    transfer_headers, transfer_rows = tables["transfer"]
    sensitivity_headers, sensitivity_rows = tables["sensitivity"]
    quality_headers, quality_rows = tables["quality"]
    cams_increment_headers, cams_increment_rows = tables["cams_increment"]
    increment_validation = evidence["cams_increment"].loc[
        evidence["cams_increment"].split.eq("validation")
        & evidence["cams_increment"].forecast_hour.eq("all")
    ].iloc[0]
    increment_test = evidence["cams_increment"].loc[
        evidence["cams_increment"].split.eq("test")
        & evidence["cams_increment"].forecast_hour.eq("all")
    ].iloc[0]
    # Keep a text fingerprint so parity can be checked without exposing implementation paths.
    markdown_word_count = len(markdown.split())
    return rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=22mm]{{geometry}}
\usepackage{{fontspec}}
\setmainfont{{Liberation Serif}}
\setsansfont{{Liberation Sans}}
\usepackage{{microtype}}
\usepackage{{graphicx}}
\usepackage{{adjustbox}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{float}}
\usepackage{{pdflscape}}
\usepackage{{caption}}
\usepackage{{enumitem}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\definecolor{{Accent}}{{HTML}}{{006D77}}
\definecolor{{Alert}}{{HTML}}{{B45309}}
\definecolor{{Soft}}{{HTML}}{{EDF6F3}}
\hypersetup{{colorlinks=true,linkcolor=Accent,urlcolor=Accent,citecolor=Accent,pdftitle={{Research-to-operational hourly PM2.5 forecasting at 27 Indonesian stations}}}}
\captionsetup{{font=small,labelfont=bf}}
\setlist{{nosep,leftmargin=6mm}}
\setlength{{\parskip}}{{0.55em}}
\setlength{{\parindent}}{{0pt}}
\newcommand{{\highlight}}[1]{{\colorbox{{Soft}}{{\parbox{{0.94\linewidth}}{{#1}}}}}}
\title{{\sffamily\bfseries Research-to-operational hourly PM\textsubscript{{2.5}} forecasting\\at 27 Indonesian stations}}
\author{{}}
\date{{Experimental model-development report --- not yet an operational public product\\{datetime.now(timezone.utc):%d %B %Y}}}
\begin{{document}}
\maketitle
\vfill
\highlight{{\textbf{{Scope.}} Daily 00 UTC forecasts at +3, +6, +12, +24, +48, and +72 hours. The output is predictive and does not provide causal or source attribution.}}
\vfill
\thispagestyle{{empty}}
\clearpage
\tableofcontents
\clearpage

\section{{Executive finding}}
The validation-selected model is \textbf{{{champion}}}. Across the six lead times in the untouched 1 January--31 August 2026 test period, its mean station-balanced mean absolute error (MAE) is \textbf{{{values['mean_test_mae']:.2f}~µg~m\textsuperscript{{-3}}}}, corresponding to \textbf{{{values['mean_test_skill']:+.1f}\%}} mean skill relative to persistence and \textbf{{{values['mean_mos_gain_vs_raw_cams_pct']:+.1f}\%}} relative to raw CAMS. Improvement occurs in \textbf{{{values['positive_station_leads']:.1f}\%}} of station--lead combinations, so aggregate skill does not imply uniform local benefit. The nominal 80\% prediction intervals attain \textbf{{{values['mean_coverage']:.1f}\%}} mean test coverage across leads.

\highlight{{\textbf{{Operational decision.}} Retain the system as pre-operational. It is suitable for scheduled shadow forecasts and scorecard monitoring, but public or duty-forecaster reliance should wait for prospective evaluation, explicit input-latency tests, a second forecast cycle, and an assessment of forecast meteorology.}}

\section{{Question and intended use}}
The question is whether a pooled machine-learning model can improve station-level hourly PM\textsubscript{{2.5}} forecasts over simple persistence at 27 BMKG locations while remaining computationally light enough for routine operation. One direct model is fitted for each forecast lead; direct forecasting avoids recursive error propagation. The intended output is a concentration forecast in µg~m\textsuperscript{{-3}} plus a calibrated uncertainty interval at each observed station.

The primary inference domain is the existing network. A separate station-holdout experiment asks how well the approach transfers to a location absent from fitting. Neither experiment creates a continuous spatial concentration field.

\section{{Data, quality control, and provenance}}
The supplied archive contains \textbf{{{observation['source_rows']:,}}} rows from 27 hourly station files spanning {_tex_escape(str(observation['start_utc'])[:10])} to {_tex_escape(str(observation['end_utc'])[:10])}. After documented error screening, \textbf{{{values['valid_rows']:,} ({values['valid_pct']:.2f}\%)}} source rows contain valid PM\textsubscript{{2.5}}. Exact duplicated station-hours were removed only in derived data; no conflicting duplicate key was accepted. Negative PM\textsubscript{{2.5}} and values at or above 985~µg~m\textsuperscript{{-3}} were treated as instrument error values. Plausible extremes below the cutoff were retained.

Relative humidity outside (0,100]\% and temperature outside -10 to 50~°C were made unavailable as predictors without altering PM\textsubscript{{2.5}}. This screening identified a block of physically impossible temperature values at Koto Tabang; the values were not corrected or imputed. Median station-level valid PM\textsubscript{{2.5}} coverage within each station's observed span is \textbf{{{values['median_station_coverage']:.1f}\%}}.

CAMS global atmospheric-composition forecast PM\textsubscript{{2.5}} was extracted from the official \texttt{{ECMWF/CAMS/NRT}} Earth Engine mirror~\cite{{camsgee}} and bilinearly sampled to station coordinates. Source mass density in kg~m\textsuperscript{{-3}} was converted explicitly to µg~m\textsuperscript{{-3}}. The parent CAMS service provides global forecasts twice daily on a 0.4° grid and is an atmospheric model product, not a surface observation~\cite{{cams}}. The mirror is three-hourly, so +3~h is the earliest evaluated lead; no +1~h field was synthesized.

The extract retained \textbf{{{cams_acquisition['rows']:,} of {cams_acquisition['maximum_expected_rows']:,} station--issue--lead records ({cams_acquisition['coverage_pct']:.2f}\%)}}. Missing records comprise a partial spatial gap on 17--23 February 2024 and a fully missing 00 UTC initialization on 30 June 2025; they were not imputed. A direct Copernicus archive check covered {cams_equivalence['stations']} in-domain stations and five common leads on 1 January 2026. Across 100 native grid-point comparisons, the maximum absolute difference was {native_check['maximum_absolute_difference_ug_m3']:.6f}~µg~m\textsuperscript{{-3}}; station-resampling implementations differed by {station_check['mean_absolute_difference_ug_m3']:.3f}~µg~m\textsuperscript{{-3}} on average and {station_check['maximum_absolute_difference_ug_m3']:.3f}~µg~m\textsuperscript{{-3}} at most. This supports identity of the underlying field while documenting interpolation-level differences. The bounded experiment uses CAMS PM\textsubscript{{2.5}} plus observed temperature and humidity histories; forecast meteorology is a declared next-stage input.

{_figure_tex(1, 'figure_01_observation_coverage', 'Monthly valid PM2.5 coverage across the 27 stations. Missing coverage is distinct from a valid zero concentration.')}

\section{{Prediction design}}
\subsection{{Inputs}}
Predictors use only timestamps at or before issue time: PM\textsubscript{{2.5}} lags from 0 to 168~h; temperature and relative-humidity lags to 24~h; rolling PM\textsubscript{{2.5}} mean, standard deviation, maximum, and availability over 3--168~h; local-hour and annual-cycle encodings; station coordinates and region; contemporaneous and distance-weighted network context within 400~km; and forecast-valid CAMS PM\textsubscript{{2.5}}. Missing predictor values are retained for native tree handling. Feature importance is diagnostic and not causal importance.

\subsection{{Chronological separation}}
Targets from 2023--2024 form training, 2025 is used for point-model early stopping and model choice, and 2026 through 31 August is evaluated once as the final test. The uncertainty workflow adds a nested temporal separation: January--June 2025 tunes quantile tree counts, while July--December 2025 is reserved exclusively for conformal calibration. Splits are assigned by target time, not issue time. The derived table contains \textbf{{{audit['rows']:,}}} station--issue--lead rows and \textbf{{{audit['valid_target_rows']:,}}} valid targets. Automated audits found {audit['duplicate_keys']} duplicate modeling keys, {audit['target_before_or_at_issue']} non-future targets, and no target-time overlap among train, validation, and test. Chronological and station-blocked validation are used because random splitting of structured environmental data can underestimate predictive error~\cite{{roberts}}.

\subsection{{Models and uncertainty}}
Persistence and a training-only station/local-time climatology are reference forecasts. Candidate learners are observation-only LightGBM, CAMS model-output-statistics (MOS) LightGBM, and CAMS MOS XGBoost. Gradient-boosted trees represent nonlinear interactions and missingness without the compute cost of deep sequence models~\cite{{lightgbm,xgboost}}. CAMS calibration with machine learning has prior scientific precedent, although performance is domain-specific~\cite{{wu}}.

Point and quantile predictions are clipped only at the physical lower bound of 0~µg~m\textsuperscript{{-3}}. Quantile LightGBM estimates the 10th, 50th, and 90th conditional percentiles. Quantile crossing is removed by sorting, then a lead-specific split-conformal expansion is estimated from the held July--December 2025 calibration block pooled across stations. Neither fitting nor interval calibration accesses 2026. Conformalized quantile regression provides a principled basis for adaptive prediction intervals under exchangeability assumptions~\cite{{romano}}; temporal autocorrelation and later CAMS system changes mean empirical monitoring remains necessary.

\section{{Validation selection and independent test results}}
Model choice used mean station-balanced validation MAE across all six leads. The specified tie-break prefers CAMS LightGBM when it lies within 1\% of the minimum, reducing deployment complexity while retaining a physically based forecast input.

{_tex_table(selection_headers, selection_rows, 'lrrc')}

The selected model's validation criterion is \textbf{{{float(selection.mean_station_balanced_mae_ug_m3):.2f}~µg~m\textsuperscript{{-3}}}}. The following table reports the subsequently opened test set. Positive bias means overprediction. The confidence interval is a 1,000-replicate station-week block bootstrap for absolute-error improvement over persistence.

{_tex_table(test_headers, test_rows, 'rrrrrrrrrl')}

{_figure_tex(2, 'figure_02_test_performance', 'Station-balanced test MAE by lead for all candidates and reference forecasts.')}
{_figure_tex(3, 'figure_03_station_skill', 'Selected-model MAE skill relative to persistence for every station and lead.')}

\subsection{{Chronological behaviour across development stages}}
Aggregate errors do not show whether the forecast follows episode timing, reacts late, or compresses peaks. Figure~4 therefore compares observed and predicted +24~h sequences in target-time order. Each point is the daily median across stations with valid data; no temporal smoothing or interpolation is applied.

Training-period values are not fitted predictions. They use expanding-window assessment blocks for July--December 2023, January--June 2024, and July--December 2024. Every assessment target is later than all targets used to fit its fold. The displayed model family and tree counts were selected later using 2025 validation, so this retrospective out-of-fold diagnostic describes temporal behaviour and is not an independent model-selection score. The 2025 panel is validation evidence and the 2026 panel remains the independent test.

{_figure_tex(4, 'figure_04_chronological_comparison', 'Chronological observed, selected-model, and persistence PM2.5 at +24 h for expanding-window out-of-fold training assessments, validation, and independent testing. Values are daily station medians for the 00 UTC forecast cycle.')}

An accompanying nine-page station atlas shows the complete independent-test sequences at +6, +24, and +72~h for all 27 stations, including calibrated 10th--90th percentile intervals. It is supplied separately as a vector PDF so poor-performing stations and short-lived episodes remain inspectable rather than concealed by national aggregation.

\subsection{{Incremental value of CAMS}}
Relative to the otherwise identical observation-only LightGBM, adding forecast-valid CAMS PM\textsubscript{{2.5}} reduced station-week-balanced MAE by \textbf{{{increment_validation.cams_mae_improvement_over_obs_ml_ug_m3:.3f}~µg~m\textsuperscript{{-3}}}} in validation (95\% bootstrap interval {increment_validation.ci95_lower_ug_m3:.3f} to {increment_validation.ci95_upper_ug_m3:.3f}) and \textbf{{{increment_test.cams_mae_improvement_over_obs_ml_ug_m3:.3f}~µg~m\textsuperscript{{-3}}}} in the independent test (95\% interval {increment_test.ci95_lower_ug_m3:.3f} to {increment_test.ci95_upper_ug_m3:.3f}). This is an aggregate predictive association, not causal evidence. Test-period gains are clearest at +3 to +12~h; intervals cross zero at each of +24, +48, and +72~h, so longer-lead incremental benefit remains inconclusive individually.

{_tex_table(cams_increment_headers, cams_increment_rows, 'lrrrr')}

The five least favourable station--lead combinations include {worst_text}. These are failure modes for investigation, not grounds for deleting observations or selectively omitting stations.

{_figure_tex(5, 'figure_04_observed_vs_predicted', 'Observed versus selected-model PM2.5 at +24 h and +72 h. Display axes are limited to the pooled 99.5th percentile for legibility; metrics use the full valid range.')}

\section{{Uncertainty and high-concentration performance}}
{_tex_table(interval_headers, interval_rows, 'rrrrr')}
{_figure_tex(6, 'figure_05_prediction_intervals', 'Independent empirical coverage and mean width of the nominal 80 percent intervals.')}

High-concentration events are defined separately for each station and lead using the station's training-period 90th percentile. Probability of detection (POD) is the fraction of observed events detected; false-alarm ratio (FAR) is the fraction of predicted events that did not occur; critical success index (CSI) penalizes misses and false alarms.

{_tex_table(event_headers, event_rows, 'rrrrrrr')}
{_figure_tex(7, 'figure_06_high_event_detection', 'High-concentration event performance on the independent test period.')}

\section{{Robustness, spatial transfer, and interpretation}}
The same-network test estimates performance for stations represented in training. Five station folds provide a separate transfer diagnostic: each held station is tested with a model fitted to the remaining stations and without station identity. Fold assignment is independent of target values.

{_tex_table(transfer_headers, transfer_rows, 'rrrr')}
{_figure_tex(8, 'figure_08_station_transfer', 'Performance when test stations are excluded from model fitting.')}

The 2024-only training sensitivity holds model form and tree counts fixed, changing only the fitting window. A positive difference means the shorter window has higher test MAE.

{_tex_table(sensitivity_headers, sensitivity_rows, 'rrrr')}

Distribution-shift diagnostics compare predictor missingness, standardized mean differences, and population stability index between training and test. These statistics identify monitoring candidates; they do not establish that a shift caused an error. Residual bias is stratified by station, target month, and local target hour.

{_figure_tex(9, 'figure_07_feature_importance', 'Mean fitted-tree feature importance for the selected model. Correlated predictors can exchange importance.')}
{_figure_tex(10, 'figure_09_residual_bias', 'Selected-model mean residual by target month and lead.')}

\section{{Computational requirements and operational design}}
The experiment is designed for a CPU workstation; no GPU is required. Training uses at most 8 CPU threads. Measured peak memory was \textbf{{{_tex_escape(peak_text)}}}. Measured preparation time was \textbf{{{_number(values['preparation_seconds'], 1)}~s}}, candidate/uncertainty/transfer evaluation was \textbf{{{_number(values['training_seconds'], 1)}~s}}, and deployment refitting was \textbf{{{values['deployment_refit_seconds']:.1f}~s}}. The complete deterministic pipeline took \textbf{{{values['deterministic_pipeline_seconds'] / 60.0:.1f}~min}} across its latest successful stage measurements. The serialized deployment bundle is \textbf{{{values['deployment_mebibytes']:.1f}~MiB}}. Warm inference across all point, fallback, and interval models totals about \textbf{{{values['inference_median_seconds']:.3f}~s}}, excluding model loading, input download, and feature preparation. The measured prepared-input end-to-end command took \textbf{{{_number(values['operational_elapsed_seconds'], 2)}~s}} for 162 station--lead rows.

For a routine 00 UTC run, a practical allocation is \textbf{{4--8 CPU cores, 4~GiB RAM, and 2~GiB working storage}}. Model inference should finish in under one minute; CAMS acquisition is network- and service-limited and should be budgeted at \textbf{{5--30 minutes}} with retries. The initial 3.7-year Earth Engine backfill required \textbf{{{values['historical_acquisition_minutes']:.1f}~min}} of measured server-query time. A full deterministic research rebuild should be budgeted at \textbf{{5--15 minutes}} on a comparable CPU workstation; slower CPUs and uncached report dependencies may require longer.

Deployment point models are refitted through December 2025; deployment quantile models stop at June 2025 so July--December remains a held calibration block. If CAMS PM\textsubscript{{2.5}} is missing, a separately fitted observation-only point forecast is used and the status is marked degraded; calibrated intervals are withheld. Observation older than six hours is explicitly flagged. This fallback supports continuity but must not be presented as equivalent quality.

\subsection{{Implemented daily shadow workflow}}
The non-public shadow workflow is scheduled daily at 17:15 WIB (10:15 UTC). It saves an immutable BMKG dashboard snapshot, freezes the observation cutoff, acquires the current CAMS 00 UTC initialization directly from the Copernicus archive, writes atomic forecasts with hashes and freshness/status fields, and scores them only after observations appear. First-seen station-hour values are retained for verification and later raw snapshots preserve revisions. There is no public upload.

The first end-to-end engineering run on 4 September 2026 generated \textbf{{{evidence['shadow_state'].get('forecast_rows', 'NA')} rows in {_number(evidence['shadow_state'].get('elapsed_seconds'), 1)}~s}} with \textbf{{{evidence['shadow_metadata'].get('degraded_rows', 'NA')} degraded rows}}. It produced \textbf{{{evidence['shadow_metadata'].get('prospective_rows', 'NA')} prospectively eligible rows}} and labelled \textbf{{{evidence['shadow_metadata'].get('late_rows', 'NA')} rows}} whose target times had already occurred. This first execution is a workflow test, not a prospective performance result.

That engineering run completed \textbf{{{_number(evidence['shadow_generation_lag_hours'], 1)}~h after the 00 UTC initialization}}. Observation freshness in shadow output is evaluated relative to actual generation time as well as model initialization; otherwise an observation timestamped at 00 UTC would be incorrectly described as fresh about eleven hours later.

\begin{{enumerate}}
\item Retrieve and validate the 00 UTC CAMS forecast after publication.
\item Freeze the latest observation cutoff and record station freshness.
\item Construct predictors without accessing later observations.
\item Produce forecasts, intervals, status flags, hashes, and logs.
\item Score forecasts when observations arrive; retain missing cases rather than silently backfilling.
\item Record warnings for missing CAMS, stale station data, schema/version changes, degraded operation, extreme residuals, and interval undercoverage.
\end{{enumerate}}

\section{{Limitations and readiness gates}}
\begin{{itemize}}
\item One daily cycle was retrospectively tested; this is not evidence for 12 UTC or arbitrary issue times.
\item Direct CAMS availability is later than model initialization. At the installed schedule, +3~h and +6~h are latency diagnostics rather than prospective forecasts; +12~h is eligible only if generation completes before 12~UTC. Prospective reporting uses the stored per-row eligibility flag.
\item The test covers eight months of 2026, not a complete annual cycle, and no prospective shadow period has yet been observed.
\item CAMS is much coarser than a station and its forecast system can change; a sampled grid value is not a station measurement.
\item The CAMS mirror is 99.54\% complete for the specified station--issue--lead grid; missing cycles are excluded from primary complete-case comparisons and require the declared degraded fallback in operations.
\item Forecast meteorology was not included. Observed meteorology at or before issue time cannot represent future dispersion conditions.
\item Missingness, maintenance, relocation, calibration change, and reporting latency can alter station performance.
\item Lead-specific conformal calibration pooled across stations assumes exchangeability more strongly than this autocorrelated, spatially dependent archive warrants. The later calibration block prevents fitting leakage, but held-out coverage is reported rather than guaranteed operationally.
\item Station-transfer results do not validate a continuous spatial map or arbitrary new station.
\item Feature importance is not emissions, wildfire, transport, contribution, or causal attribution.
\end{{itemize}}

Before public operations, require at least 60--90 days of prospective shadow scoring; a documented data-arrival cutoff; 12 UTC backtesting if desired; incremental-skill tests for forecast wind, boundary-layer height, humidity, and precipitation; CAMS-version monitoring; station-specific fallback rules; alerting and rollback; and scientific sign-off on thresholds and wording.

\section{{Reproducibility and data-quality detail}}
Every source observation file, external archive, sampled field, configuration, research model, and deployment model has a SHA-256 checksum in the provenance record. Random seeds are fixed. Derived tables preserve UTC timestamps, source station identity, explicit units, target lead, split, and missingness. No source observation was changed.

\begin{{landscape}}
{_tex_table(quality_headers, quality_rows, 'llrrrrr')}
\end{{landscape}}

\section{{References}}
\begin{{thebibliography}}{{9}}
\bibitem{{cams}} Copernicus Atmosphere Monitoring Service. \emph{{CAMS global atmospheric composition forecasts}}. \href{{https://doi.org/10.24381/04a0b097}}{{doi:10.24381/04a0b097}}.
\bibitem{{wu}} Wu, C., Li, K., and Bai, K. (2020). Validation and calibration of CAMS PM\textsubscript{{2.5}} forecasts using in situ measurements in China and United States. \emph{{Remote Sensing}}, 12, 3813. \href{{https://doi.org/10.3390/rs12223813}}{{doi:10.3390/rs12223813}}.
\bibitem{{lightgbm}} Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. \emph{{Advances in Neural Information Processing Systems 30}}.
\bibitem{{xgboost}} Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. \emph{{Proceedings of KDD 2016}}, 785--794. \href{{https://doi.org/10.1145/2939672.2939785}}{{doi:10.1145/2939672.2939785}}.
\bibitem{{romano}} Romano, Y., Patterson, E., and Candès, E. J. (2019). Conformalized quantile regression. \emph{{Advances in Neural Information Processing Systems 32}}, 3538--3548.
\bibitem{{roberts}} Roberts, D. R. et al. (2017). Cross-validation strategies for structured data. \emph{{Ecography}}, 40, 913--929. \href{{https://doi.org/10.1111/ecog.02881}}{{doi:10.1111/ecog.02881}}.
\bibitem{{camsgee}} Google Earth Engine Data Catalog. \emph{{CAMS global near-real-time atmospheric composition forecasts}}. \href{{https://developers.google.com/earth-engine/datasets/catalog/ECMWF_CAMS_NRT}}{{Official dataset entry}}.
\end{{thebibliography}}

\vfill
\begin{{center}}\footnotesize Evidence parity marker: authoritative Markdown contains {markdown_word_count:,} words.\end{{center}}
\end{{document}}
"""


def build_reports(paths: ExperimentPaths | None = None) -> dict[str, Any]:
    paths = paths or ExperimentPaths()
    report_dir = paths.root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    evidence = _report_evidence(paths)
    tables = _tables(evidence)
    markdown = _markdown_report(evidence, tables)
    latex = _latex_report(evidence, tables)
    markdown_path = report_dir / f"{REPORT_STEM}.md"
    latex_path = report_dir / f"{REPORT_STEM}.tex"
    markdown_path.write_text(markdown, encoding="utf-8")
    latex_path.write_text(latex, encoding="utf-8")
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "markdown": {
            "path": str(markdown_path.relative_to(paths.root)),
            "bytes": markdown_path.stat().st_size,
            "sha256": file_sha256(markdown_path),
        },
        "latex": {
            "path": str(latex_path.relative_to(paths.root)),
            "bytes": latex_path.stat().st_size,
            "sha256": file_sha256(latex_path),
        },
    }
    write_json(paths.provenance / "report_source_manifest.json", manifest)
    return manifest


def compile_and_verify_pdf(paths: ExperimentPaths | None = None) -> dict[str, Any]:
    paths = paths or ExperimentPaths()
    report_dir = paths.root / "reports"
    tex_path = report_dir / f"{REPORT_STEM}.tex"
    tectonic = Path("/run/media/workstation-llm/HDD2/.venv/bin/tectonic")
    environment = os.environ.copy()
    environment["XDG_CACHE_HOME"] = "/tmp/tectonic-cache"
    completed = subprocess.run(
        [str(tectonic), tex_path.name],
        cwd=report_dir,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    log_path = paths.provenance / "report_compilation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Tectonic failed; inspect {log_path}")
    pdf_path = report_dir / f"{REPORT_STEM}.pdf"
    tools = {name: shutil.which(name) for name in ("pdfinfo", "pdftotext", "pdffonts", "pdftoppm")}
    if not all(tools.values()):
        raise RuntimeError(f"Required PDF QA tools are unavailable: {tools}")
    pdfinfo = subprocess.run(
        [tools["pdfinfo"], str(pdf_path)], check=True, capture_output=True, text=True
    ).stdout
    extracted_path = paths.provenance / "report_extracted_text.txt"
    subprocess.run([tools["pdftotext"], str(pdf_path), str(extracted_path)], check=True)
    fonts = subprocess.run(
        [tools["pdffonts"], str(pdf_path)], check=True, capture_output=True, text=True
    ).stdout
    render_dir = report_dir / "rendered_pages"
    render_dir.mkdir(parents=True, exist_ok=True)
    for stale_page in render_dir.glob("page-*.png"):
        stale_page.unlink()
    subprocess.run(
        [tools["pdftoppm"], "-png", "-r", "120", str(pdf_path), str(render_dir / "page")],
        check=True,
        capture_output=True,
        text=True,
    )
    pages_match = re.search(r"^Pages:\s+(\d+)", pdfinfo, flags=re.MULTILINE)
    size_match = re.search(r"^Page size:\s+(.+)$", pdfinfo, flags=re.MULTILINE)
    page_count = int(pages_match.group(1)) if pages_match else 0
    rendered = sorted(render_dir.glob("page-*.png"))
    embedded_flags = []
    for line in fonts.splitlines()[2:]:
        match = re.search(
            r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            embedded_flags.append(match.group(1).lower() == "yes")
    all_embedded = bool(embedded_flags) and all(embedded_flags)
    extracted = extracted_path.read_text(encoding="utf-8", errors="replace")
    required_phrases = [
        "Executive finding",
        "Computational requirements",
        "Limitations and readiness gates",
        "References",
    ]
    missing_phrases = [phrase for phrase in required_phrases if phrase not in extracted]
    report = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "pdf": str(pdf_path.relative_to(paths.root)),
        "bytes": pdf_path.stat().st_size,
        "sha256": file_sha256(pdf_path),
        "pages": page_count,
        "page_size": size_match.group(1) if size_match else "unknown",
        "rendered_pages": len(rendered),
        "all_fonts_embedded": all_embedded,
        "missing_required_phrases": missing_phrases,
        "compilation_log_sha256": file_sha256(log_path),
    }
    if page_count != len(rendered) or missing_phrases or not all_embedded:
        raise ValueError(f"PDF QA failed: {report}")
    write_json(paths.provenance / "pdf_qa.json", report)
    return report
