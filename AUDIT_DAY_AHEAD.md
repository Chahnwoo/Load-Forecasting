# Day-ahead evaluation audit

## Scope and governing definition

This audit treats the required product as a **single-origin, strict day-ahead forecast**: for each CAISO operating day `D`, one run at a configurable cutoff on `D-1` must freeze the information set and issue every interval of `D`. A feature is admissible only if its value (or the particular forecast vintage supplying it) was published no later than that cutoff. Merely having a source timestamp earlier than the target timestamp is not sufficient.

The repository does not currently represent a forecast origin, cutoff, publication timestamp, weather-forecast vintage, or CAISO operating date in its model-ready data. Consequently, the existing monthly experiments cannot demonstrate this information constraint.

## Executive conclusion

The reported monthly metrics are **not valid strict day-ahead metrics**. The tabular models are retrospective fits evaluated with realized weather for the target hour. The sequence models are rolling one-hour-ahead-style evaluations whose input window advances through the validation month. GAM additionally constructs target-load lags and rolling means from validation-month actual load. The next-day path uses forecast weather, but it does not persist a forecast vintage/cutoff, does not enforce load-feature availability at that cutoff, and constructs sequence predictions using feature rows from within the day being forecast.

No scaler leakage across the explicit monthly train/validation boundary was found in the main sklearn and Torch paths: the input preprocessor and Torch target scaler are fit on training rows only. That fact does not cure the more fundamental horizon and oracle-feature leakage.

## Feature lineage and availability

The main trainers do not use an explicit feature allowlist. They drop `load_mw`, `time_utc`, and `time_key` (and `load_pred_mw` in the next-day runner), then use **every remaining column**. This table covers the expected columns and features created by repository scripts; unrecognized columns in an input CSV are also silently admitted.

| Model input | Source and source timestamp | Known at the `D-1` cutoff? | Leakage finding |
|---|---|---|---|
| `region` | CAISO historical workbook/GridStatus TAC label; static for row timestamp `t`. | Yes. | Safe as identity, subject to target-definition consistency discussed below. One-hot categories are learned on training only. |
| `year`, `month`, `day`, `day_of_week`, `hour`, `day_of_year`, `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos`; optional `hour_00`…`hour_23` | Deterministic transforms of `time_utc=t`. | Yes. | No future-data leakage. However UTC calendar fields are not CAISO operating-day/local-clock fields and mishandle the intended market-day semantics and DST. |
| `is_weekend`, `US_federal_holidays`, `state_holidays` | Deterministic calendars evaluated at the UTC date containing `t`. | Yes. | Known, but the UTC date can differ from the Pacific operating date. Holiday definitions are generic federal/custom California calendars rather than a documented CAISO holiday schedule. |
| `temperature_2m`, `apparent_temperature`, `relative_humidity_2m`, `precipitation`, `cloud_cover`, `wind_speed_10m`, `shortwave_radiation` | Historical datasets call Open-Meteo's **Archive API** for each station at observation/valid time `t`, then population-weight values by region. | No for `t` after cutoff. | **Oracle-weather leakage** in monthly validation (and retrospective training semantics): these are realized/reanalysis observations, not archived `D-1` forecast vintages. For sequence validation, rows through `t-1` also disclose evolving realized weather after the origin. |
| `cdd_65f`, `hdd_65f` | Algebraic transforms of same-row `temperature_2m(t)`. | Only if computed from a weather forecast frozen by cutoff. | In monthly validation these inherit oracle-weather leakage. |
| `load_previous_week` (the repository's effective 168-hour lag) | Exact same-region `load_mw(t-7 days)`, created by shifting timestamps forward seven days and joining. | Normally yes, because its source precedes `D-1` by almost a week. | Timestamp-wise legal for any ordinary day-ahead cutoff. Publication latency/revisions are not recorded, so strict as-of reproducibility is still unproven. This feature is sometimes described as `load_lag_168`; no column of that exact name exists. |
| GAM `load_lag_24h` | Same-region target `load_mw(t-24 hours)` created after concatenating train and validation rows. | Not necessarily. For a target late on `D`, `t-24h` is the same late hour on `D-1`, which may be after the cutoff. | Critical leakage for every hour whose lag source is later than cutoff. It also directly reads validation actuals when forecasting later validation rows. This is the implemented equivalent of the suspected `load_lag_24`. |
| GAM `load_24h_avg` | Mean of same-region actual `load_mw` from `t-24h` through `t-1h`. | No whenever any contributing observation is after cutoff. | Critical rolling actual-load leakage. Nearly the entire window for late hours of `D` can be post-cutoff, and later validation predictions use earlier validation targets. |
| CAISO `load_pred_mw` in next-day files | OASIS `SLD_FCST`, `market_run_id=DAM`, value for interval `t`. | A DAM forecast may be available by a suitable cutoff, but the collector does not request/store a publication vintage or prove it was available at the configured origin. | It is removed from model inputs, so it does not directly leak into model predictions. The evaluator incorrectly treats it as truth. |
| Any extra CSV columns | Whatever upstream supplied at row `t`; automatically included because feature selection is drop-based. | Unknown. | Moderate schema/leakage risk: columns such as post-event flags, revised data, alternate targets, or forecasts can enter without review. |

The live next-day collectors use Open-Meteo's current Forecast API rather than the Archive API, which is directionally correct. They nevertheless fetch “now,” accept a target UTC date, and retain neither request time nor model initialization/vintage. Thus a historical rerun cannot recreate what was known at a specified `D-1` cutoff, and running after part of `D` has elapsed can obtain a newer forecast.

## Findings

### 1. Sequence models implement a rolling horizon, not strict day-ahead forecasting

**Severity: critical**  
**Files/functions:** `src/modeling/train_forecaster.py` — `TorchSequenceRegressor._build_sequences`, `fit_and_predict`; `src/modeling/run_next_day_pipeline.py` — `main` sequence branch.

For target row index `i` (target hour `t`), `_build_sequences` constructs `X[idxs[i-lookback:i]]`: the last row is `t-1`, while the target is `y(t)`. During monthly evaluation `model.predict` is called on the validation month alone. It therefore drops the first `lookback` hours of each region and, thereafter, advances the feature window through validation rows. A prediction at hour `t` can see input features stamped through `t-1`, including realized weather after the notional `D-1` cutoff. This is a rolling/online one-step setup, not 24 forecasts frozen before `D`.

The next-day runner concatenates historical rows with all next-day rows before sequence construction. The first forecast uses history, but forecast `h` later in the day consumes the feature rows for earlier hours of the forecast day. Those rows contain forecast covariates rather than loads, yet this is still not the same trained input geometry or frozen-origin contract as the monthly evaluation, and it permits whatever same-day fields are present to enter. There is also no check for hourly continuity: sequence position is based on row order, so missing hours can turn a nominal 24-hour lookback into a longer and irregular temporal window.

**Effect:** monthly sequence scores measure a changing-origin, approximately one-hour-ahead feature regime and exclude the first 24 hours of every validation region. They are not comparable to an all-hours day-ahead score or to tabular models scoring all rows.

**Recommended correction:** create samples by `(region, operating_day, forecast_origin)`. Freeze an availability-filtered snapshot at the configured cutoff and emit all 24 (or the market-defined 23/25 on DST transition days) target intervals together. Either use a direct multi-output 24-hour model, one direct model per lead, or encode only history ending at cutoff plus a 24-row matrix of cutoff-vintage known-future covariates. Assert continuity and report lead-specific metrics. Never advance the origin within `D`.

**Existing metrics:** invalid as strict day-ahead metrics. They may be relabeled as rolling sequence experiments only after documenting their realized-weather information set and unequal scored rows.

### 2. GAM leaks validation actual load through 24-hour lag and rolling average

**Severity: critical**  
**File/function:** `src/modeling/train_forecaster.py` — `build_gam_features` / `prepare_gam_data`.

The function concatenates training and validation frames, sorts them, and computes `groupby("region")["load_mw"].shift(24)` and `shift(1).rolling(24)`. Thus validation actual targets directly become features for later validation predictions. Even on the first validation day, `load_lag_24h(t)` is unavailable whenever `t-24h` is after the configured cutoff. `load_24h_avg` is still more severe because it includes actual load through `t-1h`.

This is not conventional train/validation fitting leakage—the GAM itself still fits on training rows—but it is **evaluation-time target leakage** and horizon mismatch.

**Recommended correction:** compute load features from an as-of load table filtered to `observation_available_at <= forecast_cutoff`. For a single-origin day-ahead sample, use only fixed lags guaranteed to precede cutoff (often same-hour seven-day lags, or last complete day available at cutoff) and define them relative to Pacific operating intervals. Do not concatenate validation targets into feature construction.

**Existing metrics:** GAM monthly metrics are invalid, including as ordinary out-of-sample metrics, because validation labels are used as predictors.

### 3. Monthly validation uses realized weather (oracle weather)

**Severity: critical**  
**Files/functions:** `src/data_collection/collect_caiso_dataset.py` and `src/data_collection/collect_caiso_dataset_gridstatus_dotenv.py` — `fetch_weather_for_station`, `collect_weighted_region_weather`, `add_degree_days_f`; `src/modeling/train_forecaster.py` — `build_feature_matrices`, `ProphetForecaster.predict`; `src/modeling/train_linear_regression.py` — `main`.

Historical feature collection explicitly queries `archive-api.open-meteo.com/v1/archive` by valid time. The same-row weather is passed directly to all tabular models, GAM and Prophet; sequence models consume preceding validation weather. There is no archived forecast issue time. This gives models the weather that occurred, rather than the weather forecast known on `D-1`.

**Recommended correction:** build a versioned weather-forecast archive keyed by at least `(issue_time_utc, valid_time_utc, station/model)`. For each forecast origin select the latest permitted issue no later than cutoff, preserve missing vintages rather than silently substituting realized weather, and train/evaluate using comparable forecast leads. If archived forecasts cannot be obtained, report an explicitly labeled “perfect/oracle weather” upper-bound experiment, never a strict day-ahead result.

**Existing metrics:** invalid for operational day-ahead claims. They remain interpretable only as retrospective oracle-weather experiments, with the additional model-specific defects in this audit.

### 4. There is no configurable or enforced forecast cutoff / vintage

**Severity: critical**  
**Files/functions:** `src/modeling/run_next_day_pipeline.py` — `parse_args`, `main`; `test-scripts/collect_next_day_predictions.py` — forecast collectors; `test-scripts/next_day_predictions.py` — forecast collection/finalization; monthly training entry points.

No CLI or dataset field captures cutoff time. The pipeline merely receives historical and next-day CSVs. It never rejects historical observations published after cutoff, checks that all covariate issue times precede cutoff, or confirms that a next-day weather/DAM forecast was fetched before the origin. The historical file can also extend into the forecast day: sequence history is selected with `.tail(lookback)` without requiring `time_utc <= cutoff`.

**Recommended correction:** make `--operating-day`, `--forecast-cutoff` (timezone-aware), and a forecast-vintage policy mandatory. Materialize immutable as-of feature snapshots, log source issue/publication times, enforce `available_at <= cutoff` for every value, and fail closed on violations.

**Existing metrics:** monthly metrics have no forecast-origin interpretation; next-day output is not auditable as an operational vintage.

### 5. UTC “day” is not a CAISO operating day and DST is ignored

**Severity: moderate**  
**Files/functions:** `src/modeling/train_forecaster.py` — `build_features`, `parse_predict_month`; next-day collectors — UTC target-date slicing and `utc_hourly_index`; `src/preprocessing/revise_dataset.py` — calendar features.

The repository defines a day as 00:00–23:00 UTC and derives weekday/holiday/hour from UTC. CAISO operates on Pacific prevailing time. A Pacific operating day crosses two UTC dates, and DST transition days contain 23 or 25 local hourly intervals (with repeated-hour identifiers requiring care). Month splits at 00:00 UTC likewise do not coincide with local operating-month boundaries.

**Recommended correction:** retain timezone-aware UTC interval starts as the canonical join key, but add explicit Pacific `operating_date`, market hour/interval, UTC offset and fold. Define the 24-hour requirement precisely for normal days and a documented 23/25-interval DST policy. Split and group by operating date, not UTC date.

**Existing metrics:** timestamps within a single model file may be internally joinable, but “day-ahead,” by-hour, holiday, daily, and monthly interpretations are misaligned with CAISO operations.

### 6. Preprocessing is train-only in the main monthly paths, but validation policy is incomplete

**Severity: moderate**  
**Files/functions:** `src/modeling/train_forecaster.py` — `prepare_base_data`, `build_feature_matrices`, `TorchSequenceRegressor.fit`, `prepare_gam_data`; `src/modeling/train_linear_regression.py` — `main`; `src/preprocessing/merge_collected_data.py` — `join_datasets`.

Positive findings:

* The monthly boundary is chronological: rows before the requested month train the model and rows within it validate it.
* numeric imputer medians, scaling statistics, categorical imputation and one-hot categories are fit on `X_train` and only transformed on validation;
* the Torch `y_scaler_` is fit on training targets only; and
* GAM missing-value medians and region codes come from training only.

No validation-month fitting of these objects was found. However, the repository provides no tuning/feature-selection protocol or untouched test set; repeated comparison across the same monthly output files can turn those months into a de facto model-selection set. Prophet and model hyperparameters are fixed by CLI, but any human selection based on these results must be separated from final testing. Also, raw-file merging keeps the last duplicate without source vintage/provenance, so later reruns can revise historical inputs silently.

**Recommended correction:** retain train-only fitting, add explicit train/tuning/test periods or rolling-origin backtests with nested tuning, lock preprocessing/model configuration before final test, and version source snapshots. Add tests proving validation values cannot affect learned transforms.

**Existing metrics:** not invalidated by scaler leakage (none found), but invalid for the other critical reasons. They also should not be presented as untouched generalization estimates if used to select models or hyperparameters.

### 7. Feature admission is uncontrolled

**Severity: moderate**  
**Files/functions:** `src/modeling/train_forecaster.py` — `build_feature_matrices`; `src/modeling/train_linear_regression.py` — `main`; `src/modeling/run_next_day_pipeline.py` — `main`.

All columns other than a short drop list become features. `REQUIRED_COLUMNS` checks presence but does not restrict extras. The next-day runner fills missing training columns with zero, potentially converting a missing unavailable feature into a semantically meaningful value rather than failing. This prevents a complete static availability guarantee and can introduce targets, revised values, or identifiers from new preprocessing outputs.

**Recommended correction:** define a model-versioned feature manifest containing source, valid time, availability time and missing-value policy. Select exactly that allowlist and reject unexpected/missing columns. Avoid zero substitution unless zero is a documented physical value and was the trained missingness treatment.

**Existing metrics:** currently reproducible only relative to the exact unseen input schema; validity cannot be generalized from the scripts alone.

### 8. `actual_load_mw` is not proven equivalent to OASIS `SYS_FCST_ACT_MW`

**Severity: critical for CAISO benchmark claims; moderate for within-dataset model comparisons**  
**Files/functions:** `src/data_collection/collect_caiso_dataset.py` — historical workbook parsing and `build_region_frame`; `src/data_collection/collect_caiso_dataset_gridstatus_dotenv.py` — `_standardize_gridstatus_load_frame`, load-source fallback; `src/modeling/train_forecaster.py` and `src/modeling/train_linear_regression.py` — output renaming.

`actual_load_mw` is not independently collected. It is simply the model dataset's `load_mw` renamed when predictions are saved. That `load_mw` comes from CAISO's “historical EMS hourly load” workbooks or GridStatus `caiso_load_hourly`. Workbook headers are normalized through broad aliases such as `total`, `system`, `PGE`, etc.; the code does not validate a CAISO OASIS data-item identifier. For GridStatus, a numeric value column is inferred by name, and if no CAISO total exists the code synthesizes `caiso` by summing available TAC columns.

Therefore the repository does **not establish** that this target has the same telemetry, aggregation, losses, revisions, interval convention, or CA ISO-TAC scope as OASIS `SYS_FCST_ACT_MW`. Regional default training (`pge,sce,sdge`) further means model outputs are TAC targets, not the single CAISO-system target. A sum of modeled regional predictions is not automatically comparable to `CA ISO-TAC` either, especially when only three regions are selected.

**Recommended correction:** collect the benchmark actual explicitly from the same OASIS query family as the day-ahead forecast, retaining `DATA_ITEM`, `TAC_AREA_NAME`, interval start/end, OPR date/hour/interval, publication/revision and run identifiers. Choose one target contract (for example `CA ISO-TAC` system actual), train every model on exactly that target, and reconcile the historical EMS/GridStatus series against it over an overlap period before use. Never synthesize a benchmark system actual without a documented reconciliation.

**Existing metrics:** model-to-model metrics within one prediction file use a common dataset target, but cannot be claimed to measure performance against OASIS `SYS_FCST_ACT_MW` or the CA ISO-TAC system definition. Cross-source/month comparisons may also mix definitions because fallback provenance is not retained per value.

### 9. CAISO forecast comparison does not compare both forecasts to actual load

**Severity: critical**  
**Files/functions:** `src/evaluation/evaluate_next_day.py` — `main`; `test-scripts/collect_next_day_predictions.py` — `collect_caiso_day_ahead_load_forecast`; monthly comparison scripts.

`evaluate_next_day.py` assigns CAISO's `load_pred_mw` to `y_true` and computes model error against it. This measures disagreement with CAISO's forecast, not forecast accuracy. The generated next-day prediction output contains no actual-load column. Conversely, monthly prediction outputs contain model predictions and renamed dataset actuals but no CAISO day-ahead forecast. There is therefore no repository path that verifies model and CAISO forecasts against the same actual rows.

Although the OASIS collector uses interval start time and maps TAC names, it pivots with `aggfunc="mean"`, does not validate the requested `DATA_ITEM`/run returned in each row, does not preserve OPR fields or publication vintage, and the downstream comparison merely drops missing rows rather than asserting an exact key set. The model defaults also exclude `caiso`, so region coverage can differ from a CA ISO-TAC system comparison.

**Recommended correction:** build one evaluation table keyed by exact timezone-aware `interval_start_utc` plus the same TAC/target identifier. Inner-join (and separately report rejected/missing keys for) cutoff-vintage model forecast, cutoff-eligible `SYS_FCST_DA_MW`, and the exact corresponding `SYS_FCST_ACT_MW`. Assert uniqueness, identical units, interval convention, region, and complete expected operating-day coverage before scoring. Compute each forecast's errors independently against the common actual; optionally compute paired loss differences on the identical rows.

**Existing metrics:** next-day “evaluation” metrics are not accuracy metrics and are invalid for performance claims. Monthly model comparisons are not CAISO comparisons.

### 10. Monthly outputs are not a fair common-row comparison for sequence models

**Severity: moderate**  
**Files/functions:** `src/modeling/train_forecaster.py` — `fit_and_predict`, `save_outputs`; `src/evaluation/compare_monthly_model_runs.py` and `compare_model_csvs.py`.

Tabular models score every usable validation row; sequence models score only indices at least `lookback` within each region because they build sequences from validation alone rather than seeding each region with eligible pre-boundary history. The saved metric CSV is computed before any cross-model alignment. Comparison utilities compute per-file metrics over each file's own rows; their overlap logic is based primarily on time ranges rather than asserting the same `(region, time_utc, actual target)` key set and identical actual values.

**Recommended correction:** construct evaluation samples once, with a common frozen-origin eligibility mask, and hand the identical keys/targets to every model. Assert key uniqueness, region coverage and equality of actuals. Report missing predictions as failures rather than silently changing denominators.

**Existing metrics:** not directly comparable across model families; sequence metrics omit boundary hours and use a different information set.

### 11. Duplicate handling can cause retrospective revision leakage

**Severity: minor (potentially moderate if raw vintages overlap)**  
**File/function:** `src/preprocessing/merge_collected_data.py` — `join_datasets`.

Overlapping raw files are concatenated and the last duplicate `(region, time_utc)` is retained according to filename ordering, with no collection time or provenance. A later collection may contain revised actuals or forecast-derived fields and silently replace the earlier snapshot. This is acceptable for a final-revised actual target only if documented, but not for reconstructing features as known at historical cutoffs.

**Recommended correction:** preserve source file, collection time, issue time and revision; define separate “latest final actual” and “as-of feature” tables. Deduplicate by an explicit revision policy.

**Existing metrics:** exact historical reproducibility and as-of validity are not guaranteed.

## Required redesign and acceptance checks

1. Define the target explicitly: TAC/system, units, interval start/end and final/revision policy. Prefer direct OASIS `SYS_FCST_ACT_MW` ingestion for a CAISO system benchmark.
2. Represent `operating_day`, timezone-aware `forecast_cutoff`, feature `valid_time`, `issue_time`, `available_at`, source and vintage.
3. Create a frozen sample for each origin; assert every feature has `available_at <= cutoff` and every load-derived source timestamp is no later than cutoff.
4. Use archived weather forecasts issued no later than cutoff. Keep oracle-weather experiments separate and labeled.
5. Generate the full operating-day forecast jointly/directly without consuming data arriving during `D`; implement explicit DST policy.
6. Fit imputers, encoders, scalers, feature selection and models on training origins only. Tune on separate origins, then evaluate once on untouched origins.
7. Score every model and CAISO DAM forecast against one exact actual table and identical `(target, interval_start_utc)` keys. Fail on duplicates, target mismatch, missing intervals or unequal actual values.
8. Report aggregate and lead-hour metrics, the number of origins/intervals, missingness, vintage ages, and paired common-row comparisons.

Until those checks pass, label existing results as retrospective experiments with realized weather (and, for sequence/GAM, rolling-horizon or leaked-load features), not strict day-ahead forecasts.
