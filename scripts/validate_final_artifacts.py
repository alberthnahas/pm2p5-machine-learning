#!/usr/bin/env python3
"""Independently validate the frozen experiment and reader-facing artifacts."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
AQ_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from pm25ml.data import file_sha256, load_config, write_json  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    config = load_config(ROOT / "config.json")
    observation = read_json(ROOT / "provenance" / "observation_manifest.json")
    audit = read_json(ROOT / "provenance" / "modeling_table_audit.json")
    research = read_json(ROOT / "provenance" / "research_model_manifest.json")
    deployment = read_json(ROOT / "provenance" / "deployment_model_manifest.json")
    cams = read_json(ROOT / "data" / "external" / "cams_earth_engine" / "manifest.json")
    equivalence = read_json(
        ROOT / "data" / "external" / "cams_source_validation" / "manifest.json"
    )

    source_mismatches = []
    for source in observation["source_manifest"]:
        source_path = AQ_ROOT / source["path"]
        if (
            not source_path.exists()
            or source_path.stat().st_size != source["bytes"]
            or file_sha256(source_path) != source["sha256"]
        ):
            source_mismatches.append(source["path"])
    check(
        "immutable observation source checksums",
        not source_mismatches,
        {"checked": len(observation["source_manifest"]), "mismatches": source_mismatches},
    )
    check(
        "station and source reconciliation",
        observation["source_files"] == observation["station_codes"] == 27,
        {
            "source_files": observation["source_files"],
            "station_codes": observation["station_codes"],
        },
    )
    check(
        "observation key integrity",
        observation["conflicting_duplicate_keys"] == 0,
        {
            "conflicting_duplicate_keys": observation["conflicting_duplicate_keys"],
            "exact_duplicate_rows_removed": observation["duplicate_rows_removed"],
        },
    )

    leakage_keys = [
        "duplicate_keys",
        "negative_lag_feature_definitions",
        "target_before_or_at_issue",
        "train_validation_target_time_overlap",
        "train_test_target_time_overlap",
        "validation_test_target_time_overlap",
    ]
    check(
        "leakage and split audit",
        all(audit[key] == 0 for key in leakage_keys),
        {key: audit[key] for key in leakage_keys},
    )
    check(
        "configured lead coverage",
        audit["forecast_hours"] == config["forecast_hours"] == [3, 6, 12, 24, 48, 72],
        audit["forecast_hours"],
    )
    check(
        "CAMS extraction coverage and gap inventory",
        cams["coverage_pct"] >= 98.0
        and cams["missing_station_issue_leads"]
        == len(
            pd.read_csv(
                ROOT
                / "data"
                / "external"
                / "cams_earth_engine"
                / cams["missing_inventory"]["path"]
            )
        ),
        {
            "coverage_pct": cams["coverage_pct"],
            "missing_station_issue_leads": cams["missing_station_issue_leads"],
        },
    )
    check(
        "direct ADS and Earth Engine source comparison",
        equivalence["passed"],
        {
            "native_grid_max_abs_difference_ug_m3": equivalence[
                "native_grid_comparison"
            ]["maximum_absolute_difference_ug_m3"],
            "station_sampling_mean_abs_difference_ug_m3": equivalence[
                "station_bilinear_comparison"
            ]["mean_absolute_difference_ug_m3"],
        },
    )

    config_hash = file_sha256(ROOT / "config.json")
    modeling_hash = file_sha256(ROOT / "data" / "derived" / "modeling_table.csv.gz")
    check(
        "research manifest input hashes",
        research["config_sha256"] == config_hash
        and research["modeling_table_sha256"] == modeling_hash,
        {"config_sha256": config_hash, "modeling_table_sha256": modeling_hash},
    )
    interval_config = config["prediction_intervals"]
    tuning_end = pd.Timestamp(interval_config["quantile_tuning_target_end_utc"])
    calibration_start = pd.Timestamp(
        interval_config["conformal_calibration_target_start_utc"]
    )
    calibration_end = pd.Timestamp(
        interval_config["conformal_calibration_target_end_utc"]
    )
    test_start = pd.Timestamp(config["splits"]["test_target_start_utc"])
    research_interval = research["prediction_interval_design"]
    deployment_interval = deployment["interval_calibration"]
    expected_horizon_keys = {str(value) for value in config["forecast_hours"]}
    check(
        "strict interval tuning calibration and test separation",
        tuning_end < calibration_start
        and calibration_start <= calibration_end < test_start
        and research_interval["quantile_tuning_target_period"]["end_utc"]
        == interval_config["quantile_tuning_target_end_utc"]
        and research_interval["conformal_calibration_target_period"]["start_utc"]
        == interval_config["conformal_calibration_target_start_utc"]
        and deployment_interval["quantile_fit_target_period"]["end_utc"]
        == interval_config["quantile_tuning_target_end_utc"]
        and deployment_interval["conformal_calibration_target_period"]["start_utc"]
        == interval_config["conformal_calibration_target_start_utc"]
        and set(research_interval["calibration_rows_by_forecast_hour"])
        == expected_horizon_keys
        and set(deployment_interval["calibration_rows_by_forecast_hour"])
        == expected_horizon_keys
        and all(
            int(value) > 0
            for value in research_interval[
                "calibration_rows_by_forecast_hour"
            ].values()
        )
        and all(
            int(value) > 0
            for value in deployment_interval[
                "calibration_rows_by_forecast_hour"
            ].values()
        ),
        {
            "quantile_tuning_end_utc": tuning_end.isoformat(),
            "conformal_calibration_start_utc": calibration_start.isoformat(),
            "conformal_calibration_end_utc": calibration_end.isoformat(),
            "test_start_utc": test_start.isoformat(),
            "research_calibration_rows": research_interval[
                "calibration_rows_by_forecast_hour"
            ],
            "deployment_calibration_rows": deployment_interval[
                "calibration_rows_by_forecast_hour"
            ],
        },
    )
    check(
        "deployment excludes independent test targets",
        deployment["deployment_training_target_period"]["end_utc"]
        == config["splits"]["validation_target_end_utc"]
        and deployment["deployment_training_target_period"]["excluded_test_period"]
        == [
            config["splits"]["test_target_start_utc"],
            config["splits"]["test_target_end_utc"],
        ],
        deployment["deployment_training_target_period"],
    )
    model_mismatches = []
    for manifest in (research, deployment):
        for model in manifest["models"]:
            model_path = ROOT / model["path"]
            if (
                not model_path.exists()
                or model_path.stat().st_size != model["bytes"]
                or file_sha256(model_path) != model["sha256"]
            ):
                model_mismatches.append(model["path"])
    check(
        "serialized model checksums",
        not model_mismatches,
        {
            "checked": len(research["models"]) + len(deployment["models"]),
            "mismatches": model_mismatches,
        },
    )

    ranking = pd.read_csv(ROOT / "tables" / "model_selection_ranking.csv")
    check(
        "single validation-selected champion",
        int(ranking.selected_champion.sum()) == 1
        and research["champion"]
        == ranking.loc[ranking.selected_champion, "model"].iloc[0],
        {
            "champion": research["champion"],
            "selected_rows": int(ranking.selected_champion.sum()),
        },
    )
    predictions = pd.read_csv(
        ROOT / "data" / "derived" / "validation_test_predictions.csv.gz",
        parse_dates=["issue_time_utc", "target_time_utc"],
        low_memory=False,
    )
    selected_column = research["champion"]
    comparable = predictions[["champion", selected_column]].dropna()
    check(
        "champion predictions map to frozen selection",
        comparable.champion.equals(comparable[selected_column]),
        {"compared_rows": len(comparable), "selected_column": selected_column},
    )
    intervals = predictions.dropna(
        subset=["prediction_q10", "prediction_q50", "prediction_q90"]
    )
    ordered = intervals.prediction_q10.le(intervals.prediction_q50) & intervals.prediction_q50.le(
        intervals.prediction_q90
    )
    check(
        "research prediction interval ordering",
        bool(ordered.all()),
        {"checked_rows": len(intervals), "violations": int((~ordered).sum())},
    )
    incremental = pd.read_csv(
        ROOT / "tables" / "cams_incremental_skill_vs_observation_ml.csv",
        dtype={"forecast_hour": str},
    )
    incremental_test = incremental.loc[
        incremental.split.eq("test") & incremental.forecast_hour.eq("all")
    ].iloc[0]
    check(
        "CAMS ablation bootstrap completed",
        incremental_test.bootstrap_replicates == config["bootstrap_replicates"]
        and incremental_test.ci95_lower_ug_m3
        <= incremental_test.cams_mae_improvement_over_obs_ml_ug_m3
        <= incremental_test.ci95_upper_ug_m3,
        incremental_test.to_dict(),
    )

    operational_metadata_paths = sorted((ROOT / "output").glob("*.json"))
    operational_metadata = read_json(operational_metadata_paths[-1])
    operational_output = ROOT / operational_metadata["output"]
    operational = pd.read_csv(operational_output)
    op_key_duplicates = operational.duplicated(["station_code", "forecast_hour"])
    op_intervals = operational.dropna(
        subset=[
            "prediction_q10_ug_m3",
            "prediction_q50_ug_m3",
            "prediction_q90_ug_m3",
        ]
    )
    op_ordered = op_intervals.prediction_q10_ug_m3.le(
        op_intervals.prediction_q50_ug_m3
    ) & op_intervals.prediction_q50_ug_m3.le(op_intervals.prediction_q90_ug_m3)
    check(
        "experimental operational output",
        len(operational) == 27 * 6
        and operational.station_code.nunique() == 27
        and set(operational.forecast_hour) == set(config["forecast_hours"])
        and not op_key_duplicates.any()
        and operational.forecast_pm25_ug_m3.ge(0).all()
        and op_ordered.all()
        and file_sha256(operational_output) == operational_metadata["output_sha256"]
        and operational_metadata["deployment_manifest_sha256"]
        == file_sha256(ROOT / "provenance" / "deployment_model_manifest.json"),
        {
            "rows": len(operational),
            "stations": int(operational.station_code.nunique()),
            "duplicate_keys": int(op_key_duplicates.sum()),
            "interval_violations": int((~op_ordered).sum()),
            "statuses": operational.forecast_status.value_counts().to_dict(),
        },
    )

    figure_manifest = read_json(ROOT / "provenance" / "figure_manifest.json")
    figure_mismatches = []
    png_dimensions = {}
    for figure in figure_manifest["figures"]:
        for output in figure["outputs"]:
            output_path = ROOT / output["path"]
            if (
                not output_path.exists()
                or output_path.stat().st_size != output["bytes"]
                or file_sha256(output_path) != output["sha256"]
            ):
                figure_mismatches.append(output["path"])
            if output_path.suffix == ".png":
                with Image.open(output_path) as image:
                    png_dimensions[figure["figure"]] = list(image.size)
    check(
        "figure inventory and checksums",
        len(figure_manifest["figures"]) == 9 and not figure_mismatches,
        {
            "figures": len(figure_manifest["figures"]),
            "mismatches": figure_mismatches,
            "png_dimensions": png_dimensions,
        },
    )
    pdf = read_json(ROOT / "provenance" / "pdf_qa.json")
    visual_qa = read_json(ROOT / "provenance" / "visual_qa.json")
    check(
        "current figure and PDF visual inspection",
        visual_qa["status"] == "pass"
        and visual_qa["figure_manifest_sha256"]
        == file_sha256(ROOT / "provenance" / "figure_manifest.json")
        and visual_qa["pdf_sha256"]
        == file_sha256(
            ROOT / "reports" / "pm25_station_cams_mos_research_to_operations.pdf"
        )
        and visual_qa["pdf_pages_inspected"] == pdf["pages"]
        and all(visual_qa["checks"].values()),
        visual_qa,
    )
    notebook = read_json(ROOT / "provenance" / "notebook_execution.json")
    check(
        "top-to-bottom notebook execution",
        notebook["errored_cells"] == 0
        and notebook["executed_code_cells"] == notebook["code_cells"],
        notebook,
    )
    check(
        "PDF structural QA",
        pdf["pages"] == pdf["rendered_pages"]
        and pdf["all_fonts_embedded"]
        and not pdf["missing_required_phrases"]
        and pdf["page_size"].endswith("(A4)"),
        pdf,
    )
    extracted = (ROOT / "provenance" / "report_extracted_text.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    internal_markers = [
        marker
        for marker in ("/run/media/", "experiments/machine-learning", "AQ_ADS_KEY")
        if marker in extracted
    ]
    check(
        "reader-facing report excludes internal paths and credentials",
        not internal_markers,
        {"found_markers": internal_markers},
    )

    failed = [row["name"] for row in checks if not row["passed"]]
    result = {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed_checks": sum(row["passed"] for row in checks),
        "failed_checks": len(failed),
        "failed_check_names": failed,
        "status": "pass" if not failed else "fail",
    }
    write_json(ROOT / "provenance" / "final_validation.json", result)
    print(json.dumps(result, indent=2, default=str))
    if failed:
        raise SystemExit(f"Final validation failed: {failed}")


if __name__ == "__main__":
    main()
