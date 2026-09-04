# PM₂.₅ station forecast experiment

This directory contains a reproducible, research-to-operational experiment for
direct hourly PM₂.₅ forecasts at 27 Indonesian stations. One pooled model is
fitted for each lead (+3, +6, +12, +24, +48, and +72 hours) at the 00 UTC
cycle. The primary approach is machine-learning model-output statistics (MOS):
station history and network context are combined with forecast-valid CAMS
PM₂.₅. An observation-only model provides both an ablation and a degraded
fallback.

The system is **pre-operational**. It has a retrospective chronological test,
an expanding-window training diagnostic, a station-transfer diagnostic, and a
scheduled non-public shadow workflow. It is not connected to a public product.

## Scientific design

- Training targets: 1 January 2023–31 December 2024.
- Point-model validation targets: 1 January–31 December 2025.
- Independent test targets: 1 January–31 August 2026.
- Splits are assigned by target time. Predictors use only values at or before
  issue time.
- Baselines: persistence and training-only station/local-time climatology.
- Candidates: observation-only LightGBM, CAMS MOS LightGBM, and CAMS MOS
  XGBoost.
- Uncertainty: LightGBM q10/q50/q90; January–June 2025 is used for quantile
  tuning and the strictly later July–December 2025 block is reserved for
  lead-specific split-conformal expansion.
- Robustness: station-week block bootstrap, five grouped station folds,
  high-concentration events, seasonal strata, distribution shift, residuals,
  and a shorter-training-window sensitivity.

PM₂.₅ below 0 or at/above 985 µg m⁻³ is invalid. Plausible extremes below the
cutoff are retained. Source observations and downloaded archives are immutable;
all processing outputs remain inside this experiment.

## Environment

The verified local run used:

```bash
/run/media/workstation-llm/HDD2/.venv/bin/python --version
```

The exact verified package versions are in `requirements-lock.txt`. The lock is
an execution record. A standalone clone may use another environment containing
those versions, but numerical and runtime differences should then be expected
and documented.

The observation files are not published in this repository. The default
configuration expects the original AQ layout at `../../obs/all/*.csv`. For a
standalone clone, provide an absolute input glob without editing the committed
configuration:

```bash
export PM25_OBSERVATION_GLOB='/path/to/obs/all/*.csv'
```

## Reproduce

From the standalone repository root, set the verified interpreter and run:

```bash
export PM25_PYTHON=/run/media/workstation-llm/HDD2/.venv/bin/python
"$PM25_PYTHON" scripts/acquire_external_data.py --source all
"$PM25_PYTHON" scripts/validate_cams_source_equivalence.py
"$PM25_PYTHON" scripts/run_pipeline.py
"$PM25_PYTHON" -m pytest tests -q
```

External acquisition is restartable: existing validated archives are reused.
Credentials are read from the established local AQ environment and are never
copied into experiment outputs. The default CAMS path uses the official
`ECMWF/CAMS/NRT` Earth Engine mirror and records each source image identifier.
The first lead is +3 h because this mirror is three-hourly; no unavailable +1 h
field is interpolated. The direct Copernicus archive client remains available
as an explicit fallback and source-validation route.

The deterministic stages can be resumed, for example:

```bash
"$PM25_PYTHON" scripts/run_pipeline.py --from-stage diagnostics
```

## Experimental operational run

After the deployment bundle and prepared inputs exist:

```bash
"$PM25_PYTHON" scripts/run_operational_forecast.py --issue-time 2026-08-31T00:00:00Z
```

Omitting `--issue-time` selects the latest issue time common to prepared station
and CAMS inputs. Output rows include point forecasts, calibrated q10–q90
intervals, input freshness, and a status. If CAMS is absent, the point forecast
uses the observation-only fallback and intervals are withheld. A fallback row
must not be presented as equivalent to the primary forecast.

## Daily shadow evaluation

The daily workflow runs the validated 00 UTC initialization in shadow mode,
retains the actual generation time, and excludes rows generated after their
target time from prospective claims. It snapshots the official BMKG dashboard,
acquires current CAMS directly from the Copernicus Atmosphere Data Store,
writes immutable local forecasts, and scores them only after an observation
appears. See `shadow/README.md`.

The deployed model is deliberately frozen during the first 60–90 days. Shadow
observations and errors are evidence for a separately versioned retraining
decision; the cron job does not adapt the model in place because that would
invalidate prospective evaluation.

Validate the complete frozen result, including source checksums, leakage
audits, model bundles, figures, notebook, report, and operational output:

```bash
"$PM25_PYTHON" scripts/validate_final_artifacts.py
```

## Main outputs

- `tables/`: quality, performance, uncertainty, sensitivity, transfer, and
  runtime tables, including the paired CAMS-versus-observation-only ablation.
- `figures/`: matching SVG and 300 dpi PNG scientific figures, plus nine
  high-resolution atlas plates embedded in the report.
- `models/research/`: models frozen before independent test interpretation.
- `models/deployment/`: point-model refits use training plus validation;
  interval-model refits stop before the reserved calibration block. Neither
  uses test targets.
- `notebooks/`: source and executed audit notebook.
- `reports/`: paired Markdown, LaTeX, and one verified A4 PDF report. The PDF
  includes the full formulas, worked calculations, figure derivations, and the
  nine-page all-station test atlas; the atlas is not a separate document.
- `provenance/`: checksums, environments, resource measurements, compile logs,
  and QA manifests.
- `output/`: experimental operational forecast files.

Large row-level derived data, external-source snapshots, serialized model
bundles, rendered-page intermediates, and local logs are generated locally and
excluded from Git. Summary tables, figures, the executed notebook, reports,
provenance manifests, and an example forecast are versioned.

## Scope boundary

This is a station forecast system, not a continuous spatial PM₂.₅ analysis and
not an attribution model. Station holdout tests transfer to the represented
network only. Feature importance must not be interpreted as emissions,
wildfire influence, transport contribution, or causality. See
`OPERATIONAL_READINESS.md` for the remaining gates before live use.
