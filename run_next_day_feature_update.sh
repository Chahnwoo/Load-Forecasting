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

"$PYTHON" collect_caiso_dataset_gridstatus_dotenv.py \
  --start "$YESTERDAY" \
  --end "$YESTERDAY" \
  --load-source caiso_then_gridstatus \
  --env-file "$PROJECT_DIR/.env" \
  --out-dir "$PROJECT_DIR/data/raw"

"$PYTHON" join_caiso_datasets.py \
  --raw-dir "$PROJECT_DIR/data/raw" \
  --processed-dir "$PROJECT_DIR/data/processed"