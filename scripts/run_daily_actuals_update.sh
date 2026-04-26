#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
YESTERDAY="$(date -v-1d +%Y-%m-%d)"

echo "============================================================"
echo "Daily actuals update started"
echo "Repository root: ${REPO_ROOT}"
echo "Target date (yesterday, macOS): ${YESTERDAY}"
echo "============================================================"

echo "[1/5] Collecting CAISO/GridStatus daily data into data/raw..."
"$PYTHON_BIN" src/data_collection/collect_caiso_dataset_gridstatus_dotenv.py \
  --start "$YESTERDAY" \
  --end "$YESTERDAY" \
  --load-source caiso_then_gridstatus \
  --env-file .env \
  --out-dir data/raw

echo "[2/5] Merging collected raw datasets into data/processed..."
"$PYTHON_BIN" src/preprocessing/merge_collected_data.py

echo "[3/5] Detecting latest merged dataset in data/processed..."
LATEST_MERGED_DATASET="$(find data/processed -maxdepth 1 -type f -name 'caiso_dataset_*.csv' | sort | tail -n 1)"

if [[ -z "${LATEST_MERGED_DATASET}" ]]; then
  echo "ERROR: No merged dataset found in data/processed."
  exit 1
fi

LATEST_MERGED_BASENAME="$(basename "$LATEST_MERGED_DATASET")"
REVISED_OUTPUT="data/processed/revised_${LATEST_MERGED_BASENAME}"

echo "Latest merged dataset: ${LATEST_MERGED_DATASET}"
echo "Revised output path:   ${REVISED_OUTPUT}"

echo "[4/5] Revising merged dataset with dynamic input/output..."
"$PYTHON_BIN" src/preprocessing/revise_dataset.py \
  --input "$LATEST_MERGED_DATASET" \
  --output "$REVISED_OUTPUT"

echo "[5/5] Filtering revised dataset..."
"$PYTHON_BIN" src/preprocessing/filter_dataset.py "$REVISED_OUTPUT"

echo "============================================================"
echo "Daily actuals update complete."
echo "Filtered output: data/processed/filtered.csv"
echo "============================================================"
