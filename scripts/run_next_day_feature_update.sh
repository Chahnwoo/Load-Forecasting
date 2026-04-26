#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="/path/to/Load Forecasting"
PYTHON="$PROJECT_DIR/.venv/bin/python"

cd "$PROJECT_DIR"

YESTERDAY="$(date -v-1d +%Y-%m-%d)"

echo "============================================================"
echo "Running daily actuals update"
echo "Date: ${YESTERDAY}"
echo "============================================================"

"$PYTHON" src/data_collection/collect_caiso_dataset_gridstatus_dotenv.py \
  --start "$YESTERDAY" \
  --end "$YESTERDAY" \
  --load-source caiso_then_gridstatus \
  --env-file "$PROJECT_DIR/.env" \
  --out-dir "$PROJECT_DIR/data/raw"

"$PYTHON" src/preprocessing/merge_collected_data.py \
  --raw-dir "$PROJECT_DIR/data/raw" \
  --processed-dir "$PROJECT_DIR/data/processed"
