#!/usr/bin/env bash

set -euo pipefail

CSV_PATH="./data/revised_caiso_dataset_20200101_to_20260421.csv"
SCRIPT_PATH="train_forecaster.py"
OUTPUT_DIR="model_runs"
PREDICT_MONTH="2025-12"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Running all forecasting models"
echo "CSV_PATH      : ${CSV_PATH}"
echo "SCRIPT_PATH   : ${SCRIPT_PATH}"
echo "OUTPUT_DIR    : ${OUTPUT_DIR}"
echo "PREDICT_MONTH : ${PREDICT_MONTH}"
echo "============================================================"

echo
echo "------------------------------------------------------------"
echo "1) Linear Regression"
echo "------------------------------------------------------------"
python "${SCRIPT_PATH}" "${CSV_PATH}" \
  --model linear \
  --predict_month "${PREDICT_MONTH}" \
  --output_predictions "${OUTPUT_DIR}/linear_${PREDICT_MONTH}_predictions.csv" \
  --output_metrics "${OUTPUT_DIR}/linear_${PREDICT_MONTH}_metrics.csv"

echo
echo "------------------------------------------------------------"
echo "2) Ridge Regression"
echo "------------------------------------------------------------"
python "${SCRIPT_PATH}" "${CSV_PATH}" \
  --model ridge \
  --predict_month "${PREDICT_MONTH}" \
  --ridge_alpha 10.0 \
  --output_predictions "${OUTPUT_DIR}/ridge_${PREDICT_MONTH}_predictions.csv" \
  --output_metrics "${OUTPUT_DIR}/ridge_${PREDICT_MONTH}_metrics.csv"

echo
echo "------------------------------------------------------------"
echo "3) Hinge Regression (LinearSVR)"
echo "------------------------------------------------------------"
python "${SCRIPT_PATH}" "${CSV_PATH}" \
  --model hinge_regression \
  --predict_month "${PREDICT_MONTH}" \
  --svr_c 1.0 \
  --svr_epsilon 0.1 \
  --svr_max_iter 5000 \
  --output_predictions "${OUTPUT_DIR}/hinge_regression_${PREDICT_MONTH}_predictions.csv" \
  --output_metrics "${OUTPUT_DIR}/hinge_regression_${PREDICT_MONTH}_metrics.csv"

echo
echo "------------------------------------------------------------"
echo "4) LSTM"
echo "------------------------------------------------------------"
python "${SCRIPT_PATH}" "${CSV_PATH}" \
  --model lstm \
  --predict_month "${PREDICT_MONTH}" \
  --lookback 24 \
  --epochs 10 \
  --batch_size 128 \
  --lr 0.001 \
  --hidden_dim 64 \
  --num_layers 2 \
  --dropout 0.1 \
  --output_predictions "${OUTPUT_DIR}/lstm_${PREDICT_MONTH}_predictions.csv" \
  --output_metrics "${OUTPUT_DIR}/lstm_${PREDICT_MONTH}_metrics.csv"

echo
echo "------------------------------------------------------------"
echo "5) Transformer"
echo "------------------------------------------------------------"
python "${SCRIPT_PATH}" "${CSV_PATH}" \
  --model transformer \
  --predict_month "${PREDICT_MONTH}" \
  --lookback 24 \
  --epochs 10 \
  --batch_size 128 \
  --lr 0.001 \
  --d_model 64 \
  --nhead 4 \
  --num_layers 2 \
  --dim_feedforward 128 \
  --dropout 0.1 \
  --output_predictions "${OUTPUT_DIR}/transformer_${PREDICT_MONTH}_predictions.csv" \
  --output_metrics "${OUTPUT_DIR}/transformer_${PREDICT_MONTH}_metrics.csv"

echo
echo "============================================================"
echo "All model runs completed."
echo "Saved outputs in: ${OUTPUT_DIR}"
echo "Validated month : ${PREDICT_MONTH}"
echo "============================================================"