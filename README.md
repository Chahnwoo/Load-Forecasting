# Load-Forecasting

Forecasting pipeline for **CAISO electricity load** using hourly weather and calendar features, with monthly model training/evaluation workflows.

## What this project does

This repository builds and evaluates CAISO-area load forecasts by:

- Collecting hourly data for CAISO-related regions (`caiso`, `pge`, `sce`, `sdge`, `vea`, `mwd`).
- Building features such as weather variables, weekend/holiday indicators, and prior-week lag load.
- Training multiple forecasting models month-by-month.
- Comparing model runs across months with summary tables and plots.

## Repository layout

```text
src/
  data_collection/      # CAISO/Open-Meteo/GridStatus data collection + audits
  preprocessing/        # merge/revise/filter/feature scripts
  modeling/             # model training scripts
  evaluation/           # comparison and validation scripts
scripts/                # helper shell scripts for common workflows
data/
  raw/                  # raw collected CSVs (e.g., caiso_dataset_YYYYMMDD_to_YYYYMMDD.csv)
  processed/            # merged/revised/filtered datasets
  cache/                # API/cache artifacts
outputs/
  model_runs/           # per-model monthly predictions/metrics CSVs
logs/                   # run logs
```

## Setup

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

> Note: `requirements.txt` is in a **pip-freeze-style table format** (`Package Version` columns), not a minimal hand-curated requirements list.

### 3) Configure `.env` for GridStatus fallback

Create a local `.env` file in the repo root:

```bash
cat > .env <<'ENV'
GRIDSTATUS_API_KEY=your_gridstatus_api_key_here
ENV
```

`collect_caiso_dataset_gridstatus_dotenv.py` can load this file via `--env-file` (default `.env`).

## Data collection

Primary collection script:

```bash
python src/data_collection/collect_caiso_dataset_gridstatus_dotenv.py \
  --start 2026-01-01 \
  --end 2026-01-31 \
  --load-source caiso_then_gridstatus \
  --env-file .env
```

Key points:

- Raw outputs are saved under `data/raw/` as:
  - `caiso_dataset_YYYYMMDD_to_YYYYMMDD.csv`
  - `station_weights_YYYYMMDD_to_YYYYMMDD.csv`
- Default load strategy is `caiso_then_gridstatus`.
- GridStatus fallback uses `GRIDSTATUS_API_KEY` from `.env` (or `--gridstatus-api-key`).

## Preprocessing workflow

Run these steps in order:

### 1) Merge raw collection files

```bash
python src/preprocessing/merge_collected_data.py \
  --raw-dir data/raw \
  --processed-dir data/processed
```

### 2) Add calendar + lag features (`load_previous_week`)

```bash
python src/preprocessing/revise_dataset.py \
  --input data/processed/caiso_dataset_20200101_to_20260421.csv \
  --output data/processed/revised_caiso_dataset_20200101_to_20260421.csv
```

### 3) Filter to rows with required target + lag

```bash
python src/preprocessing/filter_dataset.py \
  data/processed/revised_caiso_dataset_20200101_to_20260421.csv
```

This writes `data/processed/filtered.csv`.

### 4) (Optional) Add hour indicator columns

```bash
python src/preprocessing/add_hour_indicators.py \
  data/processed/revised_caiso_dataset_20200101_to_20260421.csv \
  --output_csv data/processed/revised_caiso_dataset_20200101_to_20260421_hours.csv
```

## Training

### Option A: Run the monthly batch script

```bash
bash scripts/train_forecasters.sh
```

This script:

- Uses `data/processed/filtered.csv` as input.
- Trains multiple models (`linear`, `ridge`, `hinge_regression`, `lstm`, `transformer`, `xgboost`) for each month `2025-01` through `2025-12`.
- Writes outputs to `outputs/model_runs/`.

### Option B: Train one model for one month directly

```bash
python src/modeling/train_forecaster.py data/processed/filtered.csv \
  --model ridge \
  --predict_month 2025-10 \
  --ridge_alpha 10.0 \
  --output_predictions outputs/model_runs/ridge_2025-10_predictions.csv \
  --output_metrics outputs/model_runs/ridge_2025-10_metrics.csv
```

## Evaluation

Compare monthly runs and produce summary CSVs + plots:

```bash
python src/evaluation/compare_monthly_model_runs.py outputs/model_runs
```

Outputs go to:

- `outputs/model_runs/comparison_outputs/monthly_metrics_summary.csv`
- `outputs/model_runs/comparison_outputs/monthly_metrics_recomputed_from_predictions.csv`
- `outputs/model_runs/comparison_outputs/*.png` (metric trend/rank plots)

## Automation (daily update script)

Script:

```bash
bash scripts/run_next_day_feature_update.sh
```

What it does:

- Collects yesterday's data via `collect_caiso_dataset_gridstatus_dotenv.py`.
- Merges newly collected raw files via `merge_collected_data.py`.

Before using it:

- **Edit `PROJECT_DIR` inside `scripts/run_next_day_feature_update.sh`** to your local repo path.
- Ensure the virtual environment and `.env` exist at the configured locations.

Cron example (daily at 06:10):

```cron
10 6 * * * /bin/bash /absolute/path/to/Load-Forecasting/scripts/run_next_day_feature_update.sh >> /absolute/path/to/Load-Forecasting/logs/daily_update.log 2>&1
```

## Data/privacy/git hygiene

- Do **not** commit `.env` (contains secrets like `GRIDSTATUS_API_KEY`).
- Generated data in `data/raw/`, `data/processed/`, and model outputs in `outputs/model_runs/` should generally not be committed unless intentionally versioning artifacts.
- Logs and cache files (`logs/`, `data/cache/`) should generally not be committed.

## Troubleshooting

### `GRIDSTATUS_API_KEY` missing

- If `--load-source` uses GridStatus fallback (`caiso_then_gridstatus`, `gridstatus`, etc.), define the key in `.env` or pass `--gridstatus-api-key`.

### `load_previous_week` missing

- Run `src/preprocessing/revise_dataset.py` first, then rerun `filter_dataset.py` or training.

### Open-Meteo / CAISO / GridStatus request failures

- Retry later (upstream outages/timeouts happen).
- Reduce date range and run in smaller batches.
- Switch `--load-source` strategy if needed.

### Pandas time parsing warnings/errors

- Confirm `time_utc` is valid and consistently formatted.
- Re-run merge/revise steps to normalize timestamp columns before training/evaluation.

### Git issues with generated tracked files

If branch switching fails due to modified generated artifacts, restore or stash them first:

```bash
git restore data/raw data/processed outputs/model_runs logs
# or: git stash push -u
```

## Quick end-to-end commands

```bash
# 1) Collect
python src/data_collection/collect_caiso_dataset_gridstatus_dotenv.py --start 2026-01-01 --end 2026-01-31 --load-source caiso_then_gridstatus --env-file .env

# 2) Merge
python src/preprocessing/merge_collected_data.py --raw-dir data/raw --processed-dir data/processed

# 3) Revise
python src/preprocessing/revise_dataset.py --input data/processed/caiso_dataset_20200101_to_20260421.csv --output data/processed/revised_caiso_dataset_20200101_to_20260421.csv

# 4) Filter
python src/preprocessing/filter_dataset.py data/processed/revised_caiso_dataset_20200101_to_20260421.csv

# 5) Train monthly batch
bash scripts/train_forecasters.sh

# 6) Evaluate
python src/evaluation/compare_monthly_model_runs.py outputs/model_runs
```
