#!/usr/bin/env bash

set -euo pipefail

CSV_PATH="data/processed/filtered.csv"
SCRIPT_PATH="src/modeling/train_forecaster.py"
OUTPUT_DIR="outputs/model_runs"

mkdir -p "${OUTPUT_DIR}"

run_model() {
  local model_name="$1"
  local predict_month="$2"
  shift 2

  echo
  echo "------------------------------------------------------------"
  echo "Running ${model_name} for ${predict_month}"
  echo "------------------------------------------------------------"

  python "${SCRIPT_PATH}" "${CSV_PATH}" \
    --model "${model_name}" \
    --predict_month "${predict_month}" \
    --output_predictions "${OUTPUT_DIR}/${model_name}_${predict_month}_predictions.csv" \
    --output_metrics "${OUTPUT_DIR}/${model_name}_${predict_month}_metrics.csv" \
    "$@"
}

echo "============================================================"
echo "Running all forecasting models for every month in 2025"
echo "CSV_PATH    : ${CSV_PATH}"
echo "SCRIPT_PATH : ${SCRIPT_PATH}"
echo "OUTPUT_DIR  : ${OUTPUT_DIR}"
echo "MONTHS      : 2025-01 through 2025-12"
echo "============================================================"

for month in \
  2025-01 2025-02 2025-03 2025-04 2025-05 2025-06 \
  2025-07 2025-08 2025-09 2025-10 2025-11 2025-12
do
  echo
  echo "============================================================"
  echo "Starting runs for ${month}"
  echo "============================================================"

  run_model linear "${month}"

  run_model ridge "${month}" \
    --ridge_alpha 10.0

  run_model hinge_regression "${month}" \
    --svr_c 1.0 \
    --svr_epsilon 0.1 \
    --svr_max_iter 5000

  run_model lstm "${month}" \
    --lookback 24 \
    --epochs 10 \
    --batch_size 128 \
    --lr 0.001 \
    --hidden_dim 64 \
    --num_layers 2 \
    --dropout 0.1

  run_model transformer "${month}" \
    --lookback 24 \
    --epochs 10 \
    --batch_size 128 \
    --lr 0.001 \
    --d_model 64 \
    --nhead 4 \
    --num_layers 2 \
    --dim_feedforward 128 \
    --dropout 0.1

  run_model xgboost "${month}"

  run_model random_forest "${month}" \
    --rf_n_estimators 300

  run_model lightgbm "${month}" \
    --lgbm_n_estimators 300 \
    --lgbm_learning_rate 0.05 \
    --lgbm_num_leaves 31

  run_model bilstm "${month}" \
    --lookback 24 \
    --epochs 10 \
    --batch_size 128 \
    --lr 0.001 \
    --hidden_dim 64 \
    --num_layers 2 \
    --dropout 0.1

  run_model stcalnet "${month}" \
    --lookback 24 \
    --epochs 10 \
    --batch_size 128 \
    --lr 0.001 \
    --hidden_dim 64 \
    --num_layers 2 \
    --cnn_channels 64 \
    --dropout 0.1

  echo
  echo "Completed all models for ${month}"
done

echo
echo "============================================================"
echo "All model runs completed."
echo "Saved outputs in: ${OUTPUT_DIR}"
echo "============================================================"
