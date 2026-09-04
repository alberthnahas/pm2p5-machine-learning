# Operational readiness checklist

Status: **pre-operational / shadow-mode candidate**

No scheduler, upload, or public product is configured by this experiment.

## Completed research controls

- [x] Reconcile all 27 source files with an explicit station registry.
- [x] Preserve raw observations and apply documented PM₂.₅ error limits only in
  derived data.
- [x] Check timestamps, duplicates, conflicts, gaps, missingness, ranges, units,
  coordinates, and time-zone offsets.
- [x] Record official-source coordinate provenance for missing station metadata.
- [x] Use direct lead-specific targets and a chronological train/validation/test
  design.
- [x] Audit target-time separation and issue-time feature construction.
- [x] Compare learned models with persistence and training-only climatology.
- [x] Include an observation-only ablation and operational fallback.
- [x] Quantify the incremental CAMS contribution with a paired station-week
  bootstrap rather than inferring it from feature importance.
- [x] Verify native CAMS grid values against a direct Copernicus archive sample
  and retain station-resampling differences.
- [x] Evaluate station-balanced and pooled metrics, local failure modes,
  high-concentration episodes, and uncertainty intervals.
- [x] Keep the July–December 2025 conformal-calibration block chronologically
  separate from quantile fitting and from the independent 2026 test.
- [x] Assess transfer using grouped station holdouts without station identity.
- [x] Serialize research and deployment models with checksums and fixed seeds.
- [x] Provide a restartable batch inference command with status/freshness flags.
- [x] Produce an executed audit notebook, vector/raster figures, and paired
  Markdown/PDF reporting.

## Required before shadow scheduling

- [ ] Define the exact observation-arrival cutoff and simulate realistic data
  latency, including late or revised station records.
- [ ] Implement a current-cycle CAMS downloader with bounded retries, timeout,
  schema/version checks, and an explicit no-data exit status.
- [ ] Add idempotent run identifiers, structured logs, atomic publication, and
  retention rules in the intended host environment.
- [ ] Connect forecast verification to observations only after their arrival;
  never use revised future records during issue-time feature construction.
- [ ] Define alert owners and thresholds for input freshness, station coverage,
  range/schema changes, inference failure, and degraded fallback frequency.
- [ ] Exercise restart, partial-input, and rollback procedures on the intended
  operational machine.

## Required before duty-forecaster or public use

- [ ] Complete at least 60–90 days of prospective shadow verification, including
  station-, lead-, season-, and high-concentration scorecards.
- [ ] Backtest the 12 UTC cycle if it will be issued; current evidence applies to
  00 UTC only.
- [ ] Acquire forecast wind, boundary-layer height, humidity, temperature, and
  precipitation, then retain them only if they add held-out incremental skill.
- [ ] Monitor CAMS model upgrades and retrain/revalidate across version changes.
- [ ] Review stations with negative skill and define station-specific fallback
  or suppression criteria without selectively hiding poor validation results.
- [ ] Confirm prediction-interval coverage prospectively and recalibrate if the
  predeclared coverage tolerance is missed.
- [ ] Obtain scientific and operational sign-off on public wording, units,
  thresholds, uncertainty display, and degraded-product labelling.
- [ ] Approve scheduling, destination, access controls, and rollback separately.

## Proposed run-time service levels

- Scheduled cycle: 00 UTC only until another cycle is independently tested.
- Compute allocation: 4–8 CPU cores, 4 GiB RAM, 2 GiB working storage; no GPU.
- Model inference target: under 1 minute for 27 stations × 6 leads.
- Measured warm model-prediction total: approximately 0.13 seconds; measured
  prepared-input end-to-end command: approximately 2.13 seconds for 162 rows.
- External-data budget: 5–30 minutes under normal archive/network conditions,
  with a hard timeout and retry policy.
- Measured historical Earth Engine backfill: 12.75 minutes of server-query time
  for 3.7 years, 27 stations, and 6 leads; cached reruns are substantially
  faster.
- Measured deterministic research rebuild: approximately 4.7 minutes on this
  workstation; allow 5–15 minutes on a comparable CPU host.
- Stale observation flag: latest valid PM₂.₅ older than 6 hours.
- CAMS outage behavior: emit an explicitly degraded observation-only point
  forecast; withhold primary-model intervals.
- Publication behavior: no publication when schema checks or model-manifest
  checksums fail.

## Rollback principle

Keep the last verified model bundle and configuration immutable. If a new
bundle fails prospective acceptance criteria, restore the previous bundle by
versioned configuration change, not by altering historical output or source
observations. Persistence remains the transparent reference product during
model outage.
