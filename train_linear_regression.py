#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm


def print_step(message: str):
    print(f"\n[INFO] {message}")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create model-ready features from the raw dataframe.
    """
    df = df.copy()

    tqdm.write("[INFO] Parsing time_utc...")
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")

    if df["time_utc"].isna().any():
        bad_count = int(df["time_utc"].isna().sum())
        raise ValueError(
            f"Found {bad_count} rows with invalid time_utc values. "
            "Please ensure the time_utc column is parseable as datetimes."
        )

    tqdm.write("[INFO] Building time-derived features...")
    df["year"] = df["time_utc"].dt.year
    df["month"] = df["time_utc"].dt.month
    df["day"] = df["time_utc"].dt.day
    df["day_of_week"] = df["time_utc"].dt.dayofweek
    df["hour"] = df["time_utc"].dt.hour
    df["day_of_year"] = df["time_utc"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    return df


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing for numeric + categorical columns.
    """
    categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute MAPE in percent, ignoring rows where actual == 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0
    if not np.any(mask):
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    mean_actual = np.mean(y_true)
    mean_forecast = np.mean(y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "BIAS": bias,
        "MEAN_ACTUAL": mean_actual,
        "MEAN_FORECAST": mean_forecast,
    }


def pretty_print_metrics(metrics: dict):
    print("\n" + "=" * 80)
    print("VALIDATION METRICS")
    print("=" * 80)

    for key in ["RMSE", "MAE", "MAPE", "BIAS", "MEAN_ACTUAL", "MEAN_FORECAST", "MSE"]:
        value = metrics[key]
        if pd.isna(value):
            formatted = "NaN"
        elif key == "MAPE":
            formatted = f"{value:,.4f}%"
        else:
            formatted = f"{value:,.4f}"

        print(f"{key:<15}: {formatted}")

    print("=" * 80)


def drop_bad_rows(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Drop rows that cannot be used for modeling/evaluation.
    """
    before = len(df)

    missing_target = int(df["load_mw"].isna().sum())
    missing_prev_week = int(df["load_previous_week"].isna().sum())

    df = df.dropna(subset=["load_mw", "load_previous_week"]).copy()

    after = len(df)
    dropped = before - after

    print(
        f"[INFO] {split_name}: dropped {dropped:,} rows "
        f"(missing load_mw={missing_target:,}, missing load_previous_week={missing_prev_week:,})."
    )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Train a linear regression model for load forecasting."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the input CSV containing both training and validation data.",
    )
    parser.add_argument(
        "--output_predictions",
        type=str,
        default="validation_predictions.csv",
        help="Path to save validation predictions CSV.",
    )
    args = parser.parse_args()

    output_path = Path(args.output_predictions)
    
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    progress = tqdm(total=8, desc="Linear regression pipeline", unit="step")

    print_step("Reading CSV...")
    df = pd.read_csv(csv_path)
    progress.update(1)

    required_columns = {
        "region",
        "time_key",
        "time_utc",
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "precipitation",
        "cloud_cover",
        "wind_speed_10m",
        "shortwave_radiation",
        "cdd_65f",
        "hdd_65f",
        "load_mw",
        "is_weekend",
        "US_federal_holidays",
        "state_holidays",
        "load_previous_week",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print_step("Building features...")
    df = build_features(df)
    progress.update(1)

    print_step("Splitting into training and validation sets...")
    cutoff = pd.Timestamp("2025-12-31 23:59:59")
    train_df = df[df["time_utc"] <= cutoff].copy()
    valid_df = df[df["time_utc"] > cutoff].copy()
    progress.update(1)

    if train_df.empty:
        raise ValueError("No training rows found on or before 2025-12-31.")
    if valid_df.empty:
        raise ValueError("No validation rows found after 2025-12-31.")

    print_step("Dropping unusable rows...")
    train_df = drop_bad_rows(train_df, "TRAIN")
    valid_df = drop_bad_rows(valid_df, "VALID")
    progress.update(1)

    if train_df.empty:
        raise ValueError(
            "No usable training rows remain after dropping rows with missing load_mw/load_previous_week."
        )
    if valid_df.empty:
        raise ValueError(
            "No usable validation rows remain after dropping rows with missing load_mw/load_previous_week."
        )

    print_step("Preparing model inputs...")
    y_train = train_df["load_mw"].copy()
    y_valid = valid_df["load_mw"].copy()

    X_train = train_df.drop(columns=["load_mw"]).copy()
    X_valid = valid_df.drop(columns=["load_mw"]).copy()

    columns_to_drop = ["time_utc", "time_key"]
    X_train = X_train.drop(columns=columns_to_drop, errors="ignore")
    X_valid = X_valid.drop(columns=columns_to_drop, errors="ignore")
    progress.update(1)

    print_step("Building preprocessing + linear regression pipeline...")
    preprocessor = make_preprocessor(X_train)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    progress.update(1)

    print_step("Fitting model...")
    model.fit(X_train, y_train)
    progress.update(1)

    print_step("Predicting on validation set and computing metrics...")
    valid_preds = model.predict(X_valid)
    metrics = compute_metrics(y_valid.values, valid_preds)
    progress.update(1)
    progress.close()

    print("\n" + "=" * 80)
    print("LINEAR REGRESSION LOAD FORECASTING RESULTS")
    print("=" * 80)
    print(f"Training rows used     : {len(train_df):,}")
    print(f"Validation rows used   : {len(valid_df):,}")
    print(f"Training end time      : {train_df['time_utc'].max()}")
    print(f"Validation start time  : {valid_df['time_utc'].min()}")
    print(f"Validation end time    : {valid_df['time_utc'].max()}")

    pretty_print_metrics(metrics)

    print_step("Saving validation metrics...")
    metrics_output_path = output_path.with_name(output_path.stem + "_metrics.csv")

    metrics_df = pd.DataFrame(
        [
            {
                "train_rows_used": len(train_df),
                "validation_rows_used": len(valid_df),
                "training_end_time": str(train_df["time_utc"].max()),
                "validation_start_time": str(valid_df["time_utc"].min()),
                "validation_end_time": str(valid_df["time_utc"].max()),
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "MAPE": metrics["MAPE"],
                "BIAS": metrics["BIAS"],
                "MEAN_ACTUAL": metrics["MEAN_ACTUAL"],
                "MEAN_FORECAST": metrics["MEAN_FORECAST"],
                "MSE": metrics["MSE"],
            }
        ]
    )

    metrics_df.to_csv(metrics_output_path, index=False)
    print(f"[INFO] Saved validation metrics to: {metrics_output_path.resolve()}")

    print_step("Saving validation predictions...")
    output_df = valid_df[["region", "time_key", "time_utc", "load_mw"]].copy()
    output_df = output_df.rename(columns={"load_mw": "actual_load_mw"})
    output_df["predicted_load_mw"] = valid_preds
    output_df["error"] = output_df["predicted_load_mw"] - output_df["actual_load_mw"]
    output_df["absolute_error"] = np.abs(output_df["error"])
    output_df["squared_error"] = output_df["error"] ** 2

    nonzero_mask = output_df["actual_load_mw"] != 0
    output_df["ape_percent"] = np.nan
    output_df.loc[nonzero_mask, "ape_percent"] = (
        np.abs(output_df.loc[nonzero_mask, "error"])
        / np.abs(output_df.loc[nonzero_mask, "actual_load_mw"])
    ) * 100.0

    
    output_df.to_csv(output_path, index=False)

    print(f"[INFO] Saved validation predictions to: {output_path.resolve()}")


if __name__ == "__main__":
    main()