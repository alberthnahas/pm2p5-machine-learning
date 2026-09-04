# Research-to-operational hourly PM₂.₅ forecasting at 27 Indonesian stations

**Experimental model-development report — not yet an operational public product**<br>
**Evaluation completed:** 04 September 2026<br>
**Forecast cycle evaluated:** 00 UTC daily<br>
**Forecast leads:** +3, +6, +12, +24, +48, and +72 hours

## Executive finding

The validation-selected model is **CAMS model-output-statistics LightGBM**. Across the six lead times in the untouched 1 January–31 August 2026 test period, its mean station-balanced mean absolute error (MAE) is **9.82 µg m⁻³**, corresponding to **+28.5%** mean skill relative to persistence and **+34.2%** relative to raw CAMS. Improvement occurs in **94.4%** of station–lead combinations, so aggregate skill does not imply uniform local benefit. The nominal 80% prediction intervals attain **78.8%** mean test coverage across leads.

**Operational decision:** retain the system as **pre-operational**. It is suitable for scheduled shadow forecasts and scorecard monitoring, but public or duty-forecaster reliance should wait for prospective evaluation, explicit input-latency tests, a second forecast cycle, and an assessment of forecast meteorology. The model provides prediction, not source attribution or causal explanation.

## 1. Question and intended use

The question is whether a pooled machine-learning model can improve station-level hourly PM₂.₅ forecasts over simple persistence at 27 BMKG locations while remaining computationally light enough for routine operation. One direct model is fitted for each forecast lead; direct forecasting avoids recursive error propagation. The intended output is a concentration forecast in µg m⁻³ plus a calibrated uncertainty interval at each observed station.

The primary inference domain is the existing network. A separate station-holdout experiment asks a harder question: how well the approach transfers to a location absent from fitting. Neither experiment creates a continuous spatial concentration field.

## 2. Data, quality control, and provenance

The supplied archive contains **1,158,532** rows from 27 hourly station files spanning 2021-01-01 to 2026-08-31. After applying documented error rules, **1,116,030 (96.33%)** source rows contain valid PM₂.₅. Exact duplicated station-hours were removed only in derived data; no conflicting duplicate key was accepted. Negative PM₂.₅ and values at or above 985 µg m⁻³ were treated as instrument error values. Plausible extremes below the cutoff were retained.

Relative humidity outside (0, 100]% and temperature outside −10 to 50 °C were made unavailable as predictors without altering PM₂.₅. This screening identified a block of physically impossible temperature values at Koto Tabang; the values were not corrected or imputed. Median station-level valid PM₂.₅ coverage within each station's observed span is **94.3%**.

CAMS global atmospheric-composition forecast PM₂.₅ was extracted from the official `ECMWF/CAMS/NRT` Earth Engine mirror [7] and bilinearly sampled to station coordinates. Source mass density in kg m⁻³ was converted explicitly to µg m⁻³. The parent CAMS service provides global atmospheric-composition forecasts twice daily on a 0.4° grid and is an atmospheric model product, not a surface observation [1]. The mirror is three-hourly, so +3 h is the earliest evaluated lead; no +1 h field was synthesized.

The extract retained **215,916 of 216,918 station–issue–lead records (99.54%)**. Missing records comprise a partial spatial gap on 17–23 February 2024 and a fully missing 00 UTC initialization on 30 June 2025. They were not imputed. An independent direct Copernicus archive check covered 20 in-domain stations and five common leads on 1 January 2026. At 100 native grid-point comparisons, the maximum absolute difference was 0.000017 µg m⁻³; station-resampling implementations differed by 0.107 µg m⁻³ on average and 1.217 µg m⁻³ at most. This supports identity of the underlying field while documenting interpolation-level differences. The bounded experiment uses CAMS PM₂.₅ plus observed temperature and humidity histories; forecast meteorology is a declared next-stage input.

![Figure 1. Monthly valid PM₂.₅ coverage across the 27 stations. Missing coverage is distinct from a valid zero concentration.](../figures/figure_01_observation_coverage.png)

*Figure 1. Monthly valid PM₂.₅ coverage across the 27 stations. Missing coverage is distinct from a valid zero concentration.*

## 3. Prediction design

### 3.1 Inputs

Predictors use only timestamps at or before issue time: PM₂.₅ lags from 0 to 168 h; temperature and relative-humidity lags to 24 h; rolling PM₂.₅ mean, standard deviation, maximum, and availability over 3–168 h; local-hour and annual-cycle encodings; station coordinates and region; the contemporaneous network mean; a distance-weighted nearby-station mean within 400 km; and forecast-valid CAMS PM₂.₅. Missing predictor values are retained for native tree handling. Feature importance is diagnostic and not causal importance.

### 3.2 Chronological separation

Targets from 2023–2024 form training, 2025 is used for point-model early stopping and model choice, and 2026 through 31 August is evaluated once as the final test. The uncertainty workflow adds a nested temporal separation: January–June 2025 is used only to tune quantile tree counts, while July–December 2025 is reserved exclusively for conformal calibration. Splits are assigned by target time, not issue time. The derived table contains **216,918** station–issue–lead rows and **200,724** valid targets. Automated audits found 0 duplicate modeling keys, 0 non-future targets, and no target-time overlap among train, validation, and test. Chronological and station-blocked validation are used because random splitting of structured environmental data can underestimate predictive error [6].

### 3.3 Models and uncertainty

Persistence and a training-only station/local-time climatology are reference forecasts. Candidate learners are observation-only LightGBM, CAMS model-output-statistics (MOS) LightGBM, and CAMS MOS XGBoost. Gradient-boosted trees represent nonlinear interactions and missingness without the compute cost of deep sequence models [3,4]. CAMS bias calibration with machine learning has prior scientific precedent, although performance is domain-specific [2].

Point and quantile predictions are clipped only at the physical lower bound of 0 µg m⁻³. Quantile LightGBM estimates the 10th, 50th, and 90th conditional percentiles. Quantile crossing is removed by sorting, then a lead-specific split-conformal expansion is estimated from the held July–December 2025 calibration block pooled across stations. Neither fitting nor interval calibration accesses 2026. Conformalized quantile regression provides a principled basis for adaptive prediction intervals under exchangeability assumptions [5]; temporal autocorrelation and later CAMS system changes mean empirical monitoring remains necessary.

## 4. Validation selection and independent test results

Model choice used the mean station-balanced validation MAE across all six leads. The specified tie-break prefers CAMS LightGBM when it lies within 1% of the minimum, reducing deployment complexity while retaining a physically based forecast input.

| Candidate | Validation MAE | Skill vs persistence (%) | Selected |
| --- | --- | --- | --- |
| CAMS MOS LightGBM | 7.12 | 28.7 | Yes |
| CAMS MOS XGBoost | 7.16 | 28.0 | No |
| Observation-only LightGBM | 7.19 | 28.3 | No |

The selected model's validation criterion is **7.12 µg m⁻³**. The following table reports the subsequently opened test set. MAE and root-mean-square error (RMSE) are in µg m⁻³; positive bias means overprediction. The confidence interval is a 1,000-replicate station-week block bootstrap for mean absolute-error improvement over persistence. A positive interval entirely above zero supports lower mean absolute error under that block-resampling design.

| Lead (h) | n | MAE | Persistence MAE | Raw CAMS MAE | Skill (%) | RMSE | Bias | r | 95% CI: MAE gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| +3 | 5,918 | 7.62 | 12.64 | 9.95 | 36.5 | 16.48 | -2.36 | 0.73 | [3.79, 6.29] |
| +6 | 5,753 | 6.29 | 14.18 | 7.91 | 48.0 | 13.44 | -1.58 | 0.72 | [7.26, 9.59] |
| +12 | 5,897 | 10.50 | 16.69 | 17.61 | 32.9 | 16.62 | -2.23 | 0.58 | [5.77, 8.14] |
| +24 | 5,859 | 10.84 | 13.11 | 17.51 | 17.6 | 19.55 | -3.43 | 0.66 | [1.02, 2.45] |
| +48 | 5,839 | 11.66 | 13.90 | 18.10 | 16.9 | 21.39 | -4.41 | 0.59 | [0.99, 3.10] |
| +72 | 5,832 | 12.03 | 14.68 | 18.55 | 19.2 | 21.83 | -4.45 | 0.57 | [1.16, 3.50] |

### 4.1 Incremental value of CAMS

Relative to the otherwise identical observation-only LightGBM, adding forecast-valid CAMS PM₂.₅ reduced station-week-balanced MAE by **0.065 µg m⁻³** in validation (95% bootstrap interval 0.034 to 0.098) and **0.178 µg m⁻³** in the independent test (95% interval 0.119 to 0.242). This is an aggregate predictive association, not causal evidence. Test-period gains are clearest at +3 to +12 h; intervals cross zero at each of +24, +48, and +72 h, so longer-lead incremental benefit remains inconclusive individually.

| Lead (h) | MAE gain | Relative gain (%) | 95% CI | Station-weeks |
| --- | --- | --- | --- | --- |
| +3 | 0.295 | 3.34 | [0.194, 0.394] | 945 |
| +6 | 0.390 | 5.43 | [0.286, 0.489] | 944 |
| +12 | 0.240 | 2.17 | [0.165, 0.315] | 944 |
| +24 | 0.012 | 0.11 | [-0.179, 0.184] | 945 |
| +48 | 0.045 | 0.36 | [-0.057, 0.143] | 945 |
| +72 | 0.089 | 0.68 | [-0.033, 0.222] | 945 |
| All leads | 0.178 | 1.65 | [0.119, 0.242] | 945 |

![Figure 2. Station-balanced test MAE by lead for all candidates and reference forecasts.](../figures/figure_02_test_performance.png)

*Figure 2. Station-balanced test MAE by lead for all candidates and reference forecasts.*

![Figure 3. Selected-model MAE skill relative to persistence for every station and lead.](../figures/figure_03_station_skill.png)

*Figure 3. Selected-model MAE skill relative to persistence for every station and lead.*

The five least favourable station–lead combinations include SINTANG (-15.2%), SUPADIO (-13.1%), SUPADIO (-12.9%), SINTANG (-10.2%), SUPADIO (-9.5%). These are failure modes for investigation, not grounds for deleting observations or selectively omitting stations.

![Figure 4. Observed versus selected-model PM₂.₅ at +24 h and +72 h. The display is truncated at the pooled 99.5th percentile only for visual legibility; metrics use the full valid range.](../figures/figure_04_observed_vs_predicted.png)

*Figure 4. Observed versus selected-model PM₂.₅ at +24 h and +72 h. The display is truncated at the pooled 99.5th percentile only for visual legibility; metrics use the full valid range.*

## 5. Uncertainty and high-concentration performance

| Lead (h) | n | Coverage (%) | Mean width | Interval score |
| --- | --- | --- | --- | --- |
| +3 | 6,050 | 79.4 | 19.8 | 42.3 |
| +6 | 5,905 | 79.3 | 17.7 | 33.0 |
| +12 | 6,108 | 81.1 | 34.0 | 53.0 |
| +24 | 6,084 | 78.0 | 31.7 | 53.4 |
| +48 | 6,084 | 77.5 | 32.3 | 57.6 |
| +72 | 6,084 | 77.5 | 32.5 | 60.6 |

![Figure 5. Independent empirical coverage and mean width of the nominal 80% intervals.](../figures/figure_05_prediction_intervals.png)

*Figure 5. Independent empirical coverage and mean width of the nominal 80% intervals.*

High-concentration events are defined separately for each station and lead using the station's training-period 90th percentile. This makes the test threshold independent of test outcomes while avoiding an arbitrary network-wide concentration cutoff. Probability of detection (POD) is the fraction of observed events detected; false-alarm ratio (FAR) is the fraction of predicted events that did not occur; critical success index (CSI) penalizes both misses and false alarms.

| Lead (h) | Hits | Misses | False alarms | POD (%) | FAR (%) | CSI (%) |
| --- | --- | --- | --- | --- | --- | --- |
| +3 | 392 | 322 | 148 | 54.9 | 27.4 | 45.5 |
| +6 | 403 | 314 | 125 | 56.2 | 23.7 | 47.9 |
| +12 | 225 | 475 | 85 | 32.1 | 27.4 | 28.7 |
| +24 | 357 | 423 | 148 | 45.8 | 29.3 | 38.5 |
| +48 | 280 | 500 | 153 | 35.9 | 35.3 | 30.0 |
| +72 | 252 | 528 | 154 | 32.3 | 37.9 | 27.0 |

![Figure 6. High-concentration event performance on the independent test period.](../figures/figure_06_high_event_detection.png)

*Figure 6. High-concentration event performance on the independent test period.*

## 6. Robustness, spatial transfer, and interpretation

The same-network test estimates performance for stations represented in training. Five station folds provide a separate transfer diagnostic: each held station is tested with a model fitted to the remaining stations and without station identity. Fold assignment is independent of target values.

| Lead (h) | Stations | MAE | Skill vs persistence (%) |
| --- | --- | --- | --- |
| +3 | 27 | 7.78 | 35.0 |
| +6 | 27 | 6.50 | 46.4 |
| +12 | 27 | 11.29 | 25.6 |
| +24 | 27 | 11.09 | 14.1 |
| +48 | 27 | 11.98 | 13.0 |
| +72 | 27 | 12.42 | 14.4 |

![Figure 8. Performance when test stations are excluded from model fitting.](../figures/figure_08_station_transfer.png)

*Figure 8. Performance when test stations are excluded from model fitting.*

The 2024-only training sensitivity holds model form and tree counts fixed, changing only the fitting window. A positive difference means the shorter window has higher test MAE.

| Lead (h) | Frozen MAE | 2024-only MAE | Difference |
| --- | --- | --- | --- |
| +3 | 7.62 | 7.85 | 0.24 |
| +6 | 6.29 | 6.63 | 0.34 |
| +12 | 10.50 | 10.99 | 0.49 |
| +24 | 10.84 | 11.28 | 0.44 |
| +48 | 11.66 | 12.01 | 0.35 |
| +72 | 12.03 | 12.39 | 0.36 |

Distribution-shift diagnostics compare predictor missingness, standardized mean differences, and population stability index between training and test. These statistics identify monitoring candidates; they do not by themselves establish that a shift caused an error. Residual bias is also stratified by station, target month, and local target hour.

![Figure 7. Mean fitted-tree feature importance for the selected model. Correlated predictors can exchange importance.](../figures/figure_07_feature_importance.png)

*Figure 7. Mean fitted-tree feature importance for the selected model. Correlated predictors can exchange importance.*

![Figure 9. Selected-model mean residual by target month and lead.](../figures/figure_09_residual_bias.png)

*Figure 9. Selected-model mean residual by target month and lead.*

## 7. Computational requirements and operational design

The experiment is designed for a CPU workstation; no GPU is required. Training uses at most 8 CPU threads. Measured peak memory was **preparation 1.43 GiB, training 0.81 GiB**. Measured preparation time was **92.9 s**, candidate/uncertainty/transfer evaluation was **103.5 s**, and deployment refitting was **24.4 s**. The complete deterministic pipeline took **4.7 min** across its latest successful stage measurements. The serialized deployment bundle is **9.8 MiB**. Warm inference across all point, fallback, and interval models totals approximately **0.136 s** on this workstation, excluding model loading, input download, and feature preparation. The measured prepared-input end-to-end command took **2.13 s** for 162 station–lead rows.

For a routine 00 UTC run, a practical allocation is **4–8 CPU cores, 4 GiB RAM, and 2 GiB working storage**. Model inference itself should finish in under one minute; CAMS acquisition is network- and service-limited and should be budgeted at **5–30 minutes** with retries. The initial 3.7-year Earth Engine backfill required **12.7 min** of measured server-query time. A full deterministic research rebuild should be budgeted at **5–15 minutes** on a comparable CPU workstation, while slower CPUs and uncached report dependencies may require longer.

The operational command reads prepared issue-time features and forecast-valid CAMS values, writes one row per station and lead, and records a model-manifest checksum. Deployment point models are refitted through December 2025; deployment quantile models stop at June 2025 so July–December remains a held calibration block. If CAMS PM₂.₅ is missing, a separately fitted observation-only point forecast is used and the status is marked degraded; calibrated intervals are withheld. Observation older than six hours is explicitly flagged. This fallback supports continuity but must not be presented as equivalent quality.

Recommended shadow workflow:

1. Retrieve and validate the 00 UTC CAMS forecast after publication.
2. Freeze the latest observation cutoff and record station freshness.
3. Construct predictors without accessing any later observation.
4. Produce forecasts, intervals, status flags, hashes, and logs.
5. Score forecasts when observations arrive; retain missing cases rather than backfilling them silently.
6. Alert on missing CAMS, stale station data, schema/version changes, extreme residuals, and interval undercoverage.

## 8. Limitations and readiness gates

- The model has been retrospectively tested at one daily cycle only. It is not evidence for 12 UTC or arbitrary issue times.
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

| Station | Name | Start | End | Valid coverage (%) | Absent hours | Invalid T |
| --- | --- | --- | --- | --- | --- | --- |
| PALU | Lore Lindu Bariri | 2021-01-23 | 2026-08-31 | 42.6 | 25,203 | 0 |
| PALANGKARAYA2 | Palangka Raya | 2021-01-01 | 2026-08-31 | 84.6 | 4,564 | 0 |
| KOTOTABANG2 | Koto Tabang | 2021-09-24 | 2026-08-31 | 85.7 | 1,805 | 3,662 |
| TANJUNGHARAPAN2 | Tanjung Harapan | 2021-01-01 | 2026-08-31 | 88.1 | 4,513 | 0 |
| SORONG | Sorong | 2021-08-19 | 2026-08-31 | 88.9 | 2,250 | 0 |
| SINTANG | Sintang | 2021-09-19 | 2026-08-31 | 91.0 | 496 | 0 |
| PANGKALANBUN2 | Pangkalan Bun | 2021-09-24 | 2026-08-31 | 91.2 | 1,353 | 0 |
| MAROS | Maros | 2021-10-13 | 2026-08-31 | 91.3 | 2,070 | 0 |
| SAMARINDA2 | Samarinda | 2021-01-01 | 2026-08-31 | 91.8 | 1,181 | 0 |
| JAMBI3 | Kota Jambi | 2021-01-01 | 2026-08-31 | 92.8 | 2,479 | 0 |
| BENGKULU | Bengkulu | 2021-08-19 | 2026-08-31 | 93.4 | 576 | 0 |
| INDRAPURI2 | Indrapuri | 2021-10-04 | 2026-08-31 | 93.6 | 757 | 0 |
| PEKANBARU2 | Pekanbaru | 2021-01-01 | 2026-08-31 | 93.6 | 2,547 | 0 |
| JAMBI4 | Muaro Jambi | 2021-08-18 | 2026-08-31 | 94.3 | 1,742 | 0 |
| KOTABARU | Kotabaru | 2021-10-09 | 2026-08-31 | 94.7 | 52 | 0 |
| MALANG | Malang | 2021-09-23 | 2026-08-31 | 94.8 | 1,085 | 0 |
| PALEMBANG4 | Musi 2 Palembang | 2021-08-21 | 2026-08-31 | 95.6 | 1,191 | 0 |
| PESAWARAN | Pesawaran | 2021-08-05 | 2026-08-31 | 95.8 | 890 | 0 |
| SEMARANG | Semarang | 2021-09-02 | 2026-08-31 | 95.8 | 1,609 | 0 |
| PONTIANAK2 | Mempawah | 2021-09-09 | 2026-08-31 | 96.3 | 629 | 0 |
| PALEMBANG3 | Talang Betutu Palembang | 2021-01-04 | 2026-08-31 | 96.4 | 928 | 0 |
| KEMAYORAN3 | Kemayoran | 2021-08-25 | 2026-08-31 | 96.6 | 785 | 0 |
| MEDAN2 | Medan | 2021-09-01 | 2026-08-31 | 97.1 | 946 | 0 |
| MLATI | Sleman | 2021-09-09 | 2026-08-31 | 97.4 | 286 | 0 |
| SUPADIO | Kubu Raya | 2021-09-09 | 2026-08-31 | 97.9 | 456 | 0 |
| BATAM2 | Batam | 2021-09-14 | 2026-08-31 | 97.9 | 292 | 0 |
| BANJARBARU2 | Banjarbaru | 2021-09-08 | 2026-08-31 | 98.2 | 234 | 0 |

## References

1. Copernicus Atmosphere Monitoring Service (CAMS). *CAMS global atmospheric composition forecasts*. DOI: [10.24381/04a0b097](https://doi.org/10.24381/04a0b097).
2. Wu, C., Li, K., and Bai, K. (2020). Validation and calibration of CAMS PM₂.₅ forecasts using in situ PM₂.₅ measurements in China and United States. *Remote Sensing*, 12, 3813. [https://doi.org/10.3390/rs12223813](https://doi.org/10.3390/rs12223813).
3. Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30*. [Primary paper](https://proceedings.neurips.cc/paper/6907-a-highly-efficient-gradient-boosting-decision-tree.pdf).
4. Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD 2016*, 785–794. [https://doi.org/10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
5. Romano, Y., Patterson, E., and Candès, E. J. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems 32*, 3538–3548. [Primary paper](https://papers.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html).
6. Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40, 913–929. [https://doi.org/10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881).
7. Google Earth Engine Data Catalog. *CAMS global near-real-time atmospheric composition forecasts*. [Official dataset entry](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_CAMS_NRT).
