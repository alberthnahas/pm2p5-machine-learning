# PM2.5 station forecast experiment plan

## Objective

Develop and independently test a pooled, direct multi-horizon PM2.5 forecast
model for the 27 BMKG locations in `obs/all`. The research target is hourly
concentration at +3, +6, +12, +24, +48, and +72 hours from the 00 UTC
forecast cycle. The historical experiment uses one daily cycle to stay within
the archive's request-cost limit; the operational target is a restartable command that
uses only information available at its issue time.

## Scientific contract

- Target: quality-controlled hourly PM2.5 concentration in micrograms per cubic
  metre at the observing station.
- Domain: 27 station locations in Indonesia.
- Observation period: 1 January 2021 through 31 August 2026, subject to
  station-specific availability.
- Internal time: UTC. Local-hour features use the documented WIB, WITA, or WIT
  offset for each station.
- Primary inference: predictive skill at observed stations. A separate grouped
  station holdout assesses transfer to locations absent from model fitting.
- The model is predictive, not causal. Feature importance must not be
  interpreted as source attribution.

## Data and provenance

1. Treat every CSV in `obs/all` as immutable input.
2. Apply the repository PM2.5 rules: values below 0 or at/above 985 micrograms
   per cubic metre are invalid; retain physically plausible extremes below the
   cutoff.
3. Remove exact duplicate station-hour rows only in derived data. Fail on
   conflicting duplicates.
4. Treat temperature outside -10 to 50 degrees Celsius and relative humidity
   outside (0, 100] percent as unavailable predictors, without modifying PM2.5.
5. Acquire a timestamped BMKG station-marker snapshot for missing coordinate
   metadata and retain its URL, SHA-256 checksum, and retrieval time.
6. Acquire issue-time CAMS global PM2.5 from the official `ECMWF/CAMS/NRT`
   Earth Engine mirror, bilinearly sampled at the 27 station coordinates.
   Retain source image identifiers, retrieval manifests, checksums, source
   units, and CAMS cycle/lead coordinates. The mirror is three-hourly, so +3 h
   is the first supported lead; no synthetic +1 h interpolation is introduced.
   Direct Copernicus Atmosphere Data Store samples provide an independent
   source-equivalence check. Station RH and temperature histories provide the
   meteorological context in this bounded implementation; forecast meteorology
   remains an explicit pre-operational extension.

## Leakage-safe design

- Training targets: 1 January 2023 through 31 December 2024. The earlier
  observations remain part of the quality assessment but are outside the
  primary model comparison because the bounded CAMS acquisition begins in
  2023.
- Point-model validation targets: 1 January through 31 December 2025. Use for
  model selection and early stopping. For uncertainty, use January–June 2025
  for quantile tuning and reserve July–December 2025 exclusively for
  split-conformal calibration.
- Final test targets: 1 January through 31 August 2026. Inspect once after the
  modelling choices are frozen.
- Assign splits by target time. Every observation lag and rolling statistic is
  calculated from timestamps at or before forecast issue time.
- Compare all learned models on identical complete target cases.
- Report both pooled metrics and station-balanced summaries.

## Models and benchmarks

1. Persistence: latest valid issue-time PM2.5.
2. Station/local-time climatology computed from training data only.
3. Observation-only pooled LightGBM using autoregressive, meteorological,
   calendar, station, and network-context predictors.
4. CAMS model-output-statistics LightGBM using the observation-only features
   plus forecast-valid CAMS fields.
5. CAMS model-output-statistics XGBoost challenger using the same evidence.
6. Quantile LightGBM models for nominal 80% prediction intervals, tuned on the
   first half of 2025 and calibrated on the strictly later half without looking
   at 2026.

## Evaluation and robustness

- MAE, RMSE, bias, correlation, R-squared, and skill relative to persistence.
- Pinball loss, interval coverage, mean interval width, and interval score.
- Station, lead-time, season, and high-concentration-regime performance.
- Cluster bootstrap by station-week for uncertainty in MAE differences.
- Five grouped station folds using a transfer model without station identity.
- Recent-training-window sensitivity and CAMS ablation.
- Residual, calibration, missingness, and distribution-shift checks.

## Deliverables

- Reusable acquisition, preparation, training, evaluation, plotting, report,
  and operational-inference code.
- Derived QA, feature, prediction, metric, uncertainty, importance, and runtime
  tables.
- Vector and high-resolution raster scientific figures.
- Executed, top-to-bottom notebook companion.
- Authoritative Markdown report plus matching LaTeX and verified A4 PDF.
- Serialized research and deployment model bundles with manifests.
- Focused unit/integration tests and an operational readiness checklist.

## Acceptance criteria

- All 27 source files and station codes are reconciled.
- No target leakage or cross-split target overlap is detected.
- The final test is untouched until model/hyperparameter selection is frozen.
- Learned models are compared with both persistence and climatology.
- Claims are supported by saved tables and independently spot-checked.
- The notebook executes successfully.
- Every figure and every PDF page is visually inspected.
- Runtime, peak memory, model size, and inference latency are measured on the
  local workstation.
- No source observation is modified and no scheduler, upload, or live service
  is changed.

## Progress

- [x] Inspect repository instructions, dirty worktree, source inventory, and
  available local forecast archives.
- [x] Complete initial 27-file schema and quality profile.
- [x] Verify a minimal authenticated CAMS archive request.
- [x] Acquire and validate the bounded BMKG and CAMS external inputs. CAMS is
  extracted only at station locations from the official Earth Engine mirror,
  avoiding an unused Indonesia-wide raster archive.
- [x] Build deterministic derived data and leakage tests.
- [x] Fit, select, calibrate, and independently test models.
- [x] Run station-transfer, CAMS-ablation, and training-window sensitivity
  analyses.
- [x] Generate and inspect figures and tables.
- [x] Execute the notebook top-to-bottom without cell errors.
- [x] Build and validate Markdown, LaTeX, and 14-page A4 PDF reports.
- [x] Benchmark operational inference and complete final artifact validation.
