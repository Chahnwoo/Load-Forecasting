# Per-Hour GAM with CAISO-Specific Adaptations — Design Spec

## Goal

Add a Generalized Additive Model (GAM) to the existing `train_forecaster.py` pipeline as `--model gam`. The GAM provides full interpretability (every feature's effect is a plottable curve) while capturing nonlinear relationships that linear/ridge models miss. It differentiates from the Fan & Hyndman 2010 paper by using shortwave radiation as a CAISO-specific solar/duck-curve feature, modeling all 5 regions in a single model with region as a factor term, and using Python's PyGAM library.

## Constraints

- 2-day implementation timeline
- Must be easy for a college student to fully understand and present
- Must not modify any existing model code paths
- Must plug into the existing pipeline (same train/val split, metrics, output format)
- All 5 regions: caiso, pge, sce, sdge, vea

## Feature Engineering

### New lag features (computed on-the-fly in `train_forecaster.py`)

Computed per-region using grouped shift/rolling operations:

| Feature | Computation | Purpose |
|---|---|---|
| `load_lag_24h` | `load_mw.shift(24)` per region | Same hour yesterday |
| `load_24h_avg` | `load_mw.rolling(24).mean()` per region | Smoothed recent demand |

### Existing features used

| Feature | Source | GAM Term Type |
|---|---|---|
| `temperature_2m` | weather data | spline `s()` |
| `shortwave_radiation` | weather data | spline `s()` |
| `load_previous_week` | revise_dataset.py | spline `s()` |
| `load_lag_24h` | new, computed | spline `s()` |
| `hour` | build_features() | spline `s()` (cyclic) |
| `day_of_year` | build_features() | spline `s()` (cyclic) |
| `load_24h_avg` | new, computed | linear `l()` |
| `relative_humidity_2m` | weather data | linear `l()` |
| `cloud_cover` | weather data | linear `l()` |
| `wind_speed_10m` | weather data | linear `l()` |
| `precipitation` | weather data | linear `l()` |
| `cdd_65f` | weather data | linear `l()` |
| `hdd_65f` | weather data | linear `l()` |
| `is_weekend` | revise_dataset.py | linear `l()` |
| `US_federal_holidays` | revise_dataset.py | linear `l()` |
| `state_holidays` | revise_dataset.py | linear `l()` |
| `region` | raw data, integer-encoded | factor `f()` |
| `day_of_week` | build_features() | factor `f()` |

### Features NOT used by GAM

- `apparent_temperature` — too correlated with `temperature_2m`; would cause concurvity issues in GAM
- `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos` — replaced by cyclic spline terms on raw `hour` and `day_of_year`
- One-hot `hour_00`..`hour_23` columns — replaced by the `hour` spline term

## GAM Structure

```
load_mw = s(temperature_2m, n_splines=25)
        + s(shortwave_radiation, n_splines=25)
        + s(load_previous_week, n_splines=25)
        + s(load_lag_24h, n_splines=25)
        + s(hour, n_splines=24, basis='cp')        # cyclic
        + s(day_of_year, n_splines=20, basis='cp')  # cyclic
        + l(load_24h_avg)
        + l(relative_humidity_2m) + l(cloud_cover) + l(wind_speed_10m) + l(precipitation)
        + l(cdd_65f) + l(hdd_65f)
        + l(is_weekend) + l(US_federal_holidays) + l(state_holidays)
        + f(region) + f(day_of_week)
```

PyGAM's `LinearGAM` is used (identity link, Gaussian distribution). Smoothing penalty `lam` is set via CLI arg `--gam_lam` (default 0.6). Number of splines via `--gam_n_splines` (default 25).

## Integration Points in `train_forecaster.py`

All changes are additive — no existing code is modified.

### 1. Import (after torch import block)

```python
try:
    from pygam import LinearGAM, s, l, f
    PYGAM_AVAILABLE = True
except ImportError:
    PYGAM_AVAILABLE = False
```

### 2. `build_gam_features(df)` — new function

- Groups by region, computes `load_lag_24h` (shift 24) and `load_24h_avg` (rolling 24 mean)
- Returns augmented DataFrame

### 3. `prepare_gam_data(train_df, valid_df)` — new function

- Calls `build_gam_features()` on both splits
- Drops rows with NaN lags
- Selects GAM feature columns
- Integer-encodes `region` and ensures proper dtypes
- Imputes remaining NaN with column medians from training set
- Returns X_train, X_valid, y_train, y_valid as numpy arrays, plus metadata

### 4. `build_gam_terms(feature_names)` — new function

- Maps feature names to PyGAM term types (s/l/f)
- Constructs and returns the `TermList`

### 5. `build_model()` — new branch

```python
if args.model == "gam":
    # GAM is built during fit_and_predict since it needs feature structure
    return None
```

### 6. `fit_and_predict()` — new branch

```python
elif args.model == "gam":
    # 1. Compute GAM-specific features
    # 2. Build feature matrix (no sklearn preprocessor)
    # 3. Construct term structure
    # 4. Fit LinearGAM
    # 5. Predict on validation set
```

### 7. `parse_args()` — additions

- Add `"gam"` to `--model` choices
- Add `--gam_n_splines` (int, default 25)
- Add `--gam_lam` (float, default 0.6)

## Training Script Update

Add to `train_forecasters.sh` inside the month loop:

```bash
run_model gam "${month}" \
  --gam_n_splines 25 \
  --gam_lam 0.6
```

## Output

Same format as all other models:
- `outputs/model_runs/gam_YYYY-MM_predictions.csv`
- `outputs/model_runs/gam_YYYY-MM_metrics.csv`

Picked up automatically by `compare_monthly_model_runs.py`.

## Dependencies

- `pygam` — install via `pip install pygam`
- All other dependencies already in the project

## What This Does NOT Change

- No changes to any existing model code paths (linear, ridge, hinge_regression, xgboost, lstm, transformer)
- No changes to preprocessing scripts (merge, revise, filter)
- No changes to evaluation scripts
- No changes to data files
- No changes to the train/val split logic or metrics computation
