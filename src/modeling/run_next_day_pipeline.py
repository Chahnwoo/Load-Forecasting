#!/usr/bin/env python3
"""
run_next_day_pipeline.py

Trains a model on the historical dataset and predicts next-day load.
"""

import argparse
import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.modeling.train_forecaster import (
    set_seed,
    print_step,
    make_preprocessor,
    build_model,
    REQUIRED_COLUMNS,
    build_features,
    TorchSequenceRegressor
)

def parse_args():
    parser = argparse.ArgumentParser(description="Next-day load forecasting pipeline.")
    parser.add_argument("--historical_csv", type=str, required=True, help="Path to historical dataset (e.g., filtered.csv).")
    parser.add_argument("--next_day_csv", type=str, required=True, help="Path to next day features dataset.")
    parser.add_argument("--output_predictions", type=str, required=True, help="Path to output predictions CSV.")
    
    parser.add_argument(
        "--model",
        type=str,
        default="ridge",
        choices=["linear", "ridge", "hinge_regression", "xgboost", "lstm", "transformer"],
        help="Model type to use."
    )
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--svr_c", type=float, default=1.0)
    parser.add_argument("--svr_epsilon", type=float, default=0.1)
    parser.add_argument("--svr_max_iter", type=int, default=5000)
    
    parser.add_argument("--xgb_n_estimators", type=int, default=300)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--xgb_reg_alpha", type=float, default=0.0)
    parser.add_argument("--xgb_reg_lambda", type=float, default=1.0)
    parser.add_argument("--xgb_min_child_weight", type=float, default=1.0)
    parser.add_argument("--xgb_tree_method", type=str, default="hist")
    
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    print_step("Loading datasets...")
    hist_df = pd.read_csv(args.historical_csv)
    next_df = pd.read_csv(args.next_day_csv)

    # Basic preparation matching `prepare_base_data` in train_forecaster.py
    hist_df = build_features(hist_df)
    next_df = build_features(next_df)

    # Drop bad rows in historical data
    hist_df = hist_df.dropna(subset=["load_mw", "load_previous_week"]).copy()
    
    # next_df won't have load_mw (or it will be nan or missing). So we can't drop on load_mw.
    # Just drop if missing previous week.
    if "load_previous_week" in next_df.columns:
        next_df = next_df.dropna(subset=["load_previous_week"]).copy()
    
    if hist_df.empty:
        raise ValueError("No historical rows remain after dropping NaNs.")
    if next_df.empty:
        raise ValueError("No next-day rows remain after dropping NaNs.")

    print_step("Building features...")
    # Sort and align
    hist_df = hist_df.sort_values(["region", "time_utc"]).reset_index(drop=True)
    next_df = next_df.sort_values(["region", "time_utc"]).reset_index(drop=True)

    y_train = hist_df["load_mw"].values
    train_groups = hist_df["region"].astype(str).values
    next_groups = next_df["region"].astype(str).values

    # Drop columns that are targets or not features
    # Note: next_df has `load_pred_mw` which we drop, and doesn't have `load_mw`
    columns_to_drop = ["load_mw", "time_utc", "time_key", "load_pred_mw"]
    
    X_train_model = hist_df.drop(columns=[c for c in columns_to_drop if c in hist_df.columns], errors="ignore").copy()
    X_next_model = next_df.drop(columns=[c for c in columns_to_drop if c in next_df.columns], errors="ignore").copy()

    # Ensure next_df has exactly the same columns
    missing_cols = set(X_train_model.columns) - set(X_next_model.columns)
    for c in missing_cols:
        X_next_model[c] = 0
    X_next_model = X_next_model[X_train_model.columns]

    preprocessor = make_preprocessor(X_train_model)
    X_train_trans = preprocessor.fit_transform(X_train_model)
    X_next_trans = preprocessor.transform(X_next_model)

    print_step(f"Training model: {args.model}")
    model = build_model(args)
    
    if args.model in {"linear", "ridge", "hinge_regression", "xgboost"}:
        model.fit(X_train_trans, y_train)
        preds = model.predict(X_next_trans)
        out_df = next_df.copy()
        out_df["predicted_load_mw"] = preds
    elif args.model in {"lstm", "transformer"}:
        model.fit(X_train_trans, y_train, groups=train_groups)
        
        # PyTorch sequence models require a sequence! 
        # But `next_df` only has 24 hours. If lookback=24, we need 24 previous hours to predict the 1st next-day hour!
        # Therefore, we MUST concatenate the last `lookback` hours from historical data onto `next_df` for sequence generation.
        print_step("Concatenating recent history for sequence prediction...")
        
        preds_list = []
        out_idx_list = []
        
        for region in next_df["region"].unique():
            hist_region = hist_df[hist_df["region"] == region].tail(args.lookback)
            next_region = next_df[next_df["region"] == region]
            
            combined = pd.concat([hist_region, next_region], ignore_index=True)
            
            # Reprocess
            X_combined_model = combined.drop(columns=[c for c in columns_to_drop if c in combined.columns], errors="ignore")
            for c in missing_cols:
                X_combined_model[c] = 0
            X_combined_model = X_combined_model[X_train_model.columns]
            
            X_comb_trans = preprocessor.transform(X_combined_model)
            comb_groups = combined["region"].astype(str).values
            
            region_preds, valid_idx = model.predict(X_comb_trans, groups=comb_groups)
            
            # The first `lookback` predictions from this `predict()` are shifted (actually valid_idx corresponds to indices >= lookback)
            # So the indices in `valid_idx` that are >= lookback map to `next_region` rows!
            mask = valid_idx >= len(hist_region)
            valid_idx_in_next = valid_idx[mask] - len(hist_region)
            
            preds_list.extend(region_preds[mask])
            
            # Store regional info to reconstruct
            # we need to map valid_idx_in_next to the original next_df indices
            original_indices = next_region.index.values[valid_idx_in_next]
            out_idx_list.extend(original_indices)
            
        preds = np.array(preds_list)
        out_idx = np.array(out_idx_list)
        
        out_df = next_df.iloc[out_idx].copy()
        out_df["predicted_load_mw"] = preds
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    print_step("Saving predictions...")
    Path(args.output_predictions).parent.mkdir(parents=True, exist_ok=True)
    
    keep_cols = ["region", "time_key", "time_utc", "load_pred_mw", "predicted_load_mw"]
    out_df = out_df[[c for c in keep_cols if c in out_df.columns]]
    out_df.to_csv(args.output_predictions, index=False)
    print(f"Saved {len(out_df)} predictions to {args.output_predictions}")

if __name__ == "__main__":
    main()
