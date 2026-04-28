#!/usr/bin/env python3
"""
evaluate_next_day.py

Evaluates the predictions of the next-day forecasting pipeline.
Since true actuals do not exist for the future, this compares our model's
predictions (`predicted_load_mw`) against CAISO's day-ahead forecast (`load_pred_mw`).
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.modeling.train_forecaster import compute_metrics, pretty_print_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate next-day predictions against CAISO forecasts.")
    parser.add_argument("predictions_csv", type=str, help="Path to output predictions CSV (e.g. from run_next_day_pipeline.py).")
    parser.add_argument("--output_metrics", type=str, default=None, help="Optional path to save metrics CSV.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.predictions_csv):
        raise FileNotFoundError(f"Predictions CSV not found: {args.predictions_csv}")
        
    df = pd.read_csv(args.predictions_csv)
    
    # We need CAISO's prediction (load_pred_mw) and our prediction (predicted_load_mw)
    if "load_pred_mw" not in df.columns or "predicted_load_mw" not in df.columns:
        raise ValueError("CSV must contain 'load_pred_mw' and 'predicted_load_mw' columns.")
        
    # Drop rows where either is missing (some hours might not have CAISO day-ahead forecast)
    eval_df = df.dropna(subset=["load_pred_mw", "predicted_load_mw"]).copy()
    
    if eval_df.empty:
        raise ValueError("No rows remain after dropping NaNs in load_pred_mw or predicted_load_mw.")
        
    y_true = eval_df["load_pred_mw"].values
    y_pred = eval_df["predicted_load_mw"].values
    
    metrics = compute_metrics(y_true, y_pred)
    
    print("\n[NEXT-DAY EVALUATION vs CAISO DAY-AHEAD FORECAST]")
    pretty_print_metrics(metrics)
    
    if args.output_metrics:
        Path(args.output_metrics).parent.mkdir(parents=True, exist_ok=True)
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(args.output_metrics, index=False)
        print(f"Saved metrics to {args.output_metrics}")

if __name__ == "__main__":
    main()
