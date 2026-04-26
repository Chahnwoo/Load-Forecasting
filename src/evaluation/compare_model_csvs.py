#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_COLUMNS = ["RMSE", "MAE", "MAPE", "BIAS", "MSE"]
PREDICTION_REQUIRED_COLUMNS = {
    "time_utc",
    "actual_load_mw",
    "predicted_load_mw",
}
PREDICTION_OPTIONAL_COLUMNS = {
    "absolute_error",
    "squared_error",
    "ape_percent",
    "region",
    "time_key",
    "error",
}


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_model_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def infer_model_name_from_filename(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in ["_metrics", "_predictions"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return normalize_model_name(stem)


def classify_csv(df: pd.DataFrame) -> Optional[str]:
    cols = set(df.columns)
    if "model" in cols and any(m in cols for m in METRIC_COLUMNS):
        return "metrics"
    if PREDICTION_REQUIRED_COLUMNS.issubset(cols):
        return "predictions"
    return None


def load_csvs(input_dir: Path) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    metrics_by_model: Dict[str, pd.DataFrame] = {}
    predictions_by_model: Dict[str, pd.DataFrame] = {}

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {input_dir}")

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[WARN] Failed to read {csv_path.name}: {exc}")
            continue

        file_type = classify_csv(df)
        if file_type is None:
            print(f"[WARN] Skipping unrecognized CSV format: {csv_path.name}")
            continue

        if file_type == "metrics":
            if "model" in df.columns and not df.empty:
                model_name = normalize_model_name(df.iloc[0]["model"])
            else:
                model_name = infer_model_name_from_filename(csv_path)
            metrics_by_model[model_name] = df.copy()

        elif file_type == "predictions":
            model_name = infer_model_name_from_filename(csv_path)
            predictions_by_model[model_name] = df.copy()

    if not metrics_by_model and not predictions_by_model:
        raise ValueError(
            "No recognized metrics or predictions CSV files were found."
        )

    return metrics_by_model, predictions_by_model


def build_metrics_table(metrics_by_model: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_name, df in metrics_by_model.items():
        if df.empty:
            continue

        row = df.iloc[0].to_dict()
        row["model"] = normalize_model_name(row.get("model", model_name))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    metrics_df = pd.DataFrame(rows)

    preferred_order = (
        ["model", "train_rows_used", "validation_rows_scored",
         "training_end_time", "validation_start_time_scored",
         "validation_end_time_scored"]
        + [c for c in METRIC_COLUMNS if c in metrics_df.columns]
        + [c for c in metrics_df.columns if c not in {
            "model", "train_rows_used", "validation_rows_scored",
            "training_end_time", "validation_start_time_scored",
            "validation_end_time_scored", *METRIC_COLUMNS
        }]
    )

    metrics_df = metrics_df[[c for c in preferred_order if c in metrics_df.columns]]
    metrics_df = metrics_df.sort_values("model").reset_index(drop=True)
    return metrics_df


def compute_prediction_summary(predictions_by_model: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for model_name, df in predictions_by_model.items():
        if df.empty:
            continue

        tmp = df.copy()
        tmp["time_utc"] = pd.to_datetime(tmp["time_utc"], errors="coerce")
        tmp = tmp.dropna(subset=["time_utc", "actual_load_mw", "predicted_load_mw"]).copy()

        if tmp.empty:
            continue

        actual = pd.to_numeric(tmp["actual_load_mw"], errors="coerce")
        pred = pd.to_numeric(tmp["predicted_load_mw"], errors="coerce")
        valid = ~(actual.isna() | pred.isna())
        tmp = tmp.loc[valid].copy()

        if tmp.empty:
            continue

        actual = tmp["actual_load_mw"].astype(float)
        pred = tmp["predicted_load_mw"].astype(float)
        err = pred - actual
        abs_err = err.abs()
        sq_err = err ** 2

        nonzero_actual = actual != 0
        if nonzero_actual.any():
            ape = (abs_err[nonzero_actual] / actual[nonzero_actual].abs()) * 100.0
            mape = float(ape.mean())
        else:
            mape = np.nan

        rows.append({
            "model": model_name,
            "rows": len(tmp),
            "start_time": tmp["time_utc"].min(),
            "end_time": tmp["time_utc"].max(),
            "RMSE_recomputed": float(np.sqrt(sq_err.mean())),
            "MAE_recomputed": float(abs_err.mean()),
            "MAPE_recomputed": mape,
            "BIAS_recomputed": float(err.mean()),
            "MEAN_ACTUAL": float(actual.mean()),
            "MEAN_FORECAST": float(pred.mean()),
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    return out


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def plot_metric_bars(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    if metrics_df.empty:
        return

    for metric in METRIC_COLUMNS:
        if metric not in metrics_df.columns:
            continue

        plot_df = metrics_df[["model", metric]].dropna()
        if plot_df.empty:
            continue

        plt.figure(figsize=(10, 6))
        plt.bar(plot_df["model"], plot_df[metric])
        plt.title(f"{metric} by Model")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"bar_{metric.lower()}.png", dpi=150)
        plt.close()


def prepare_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["time_utc"] = pd.to_datetime(tmp["time_utc"], errors="coerce")
    tmp["actual_load_mw"] = pd.to_numeric(tmp["actual_load_mw"], errors="coerce")
    tmp["predicted_load_mw"] = pd.to_numeric(tmp["predicted_load_mw"], errors="coerce")
    tmp = tmp.dropna(subset=["time_utc", "actual_load_mw", "predicted_load_mw"]).copy()

    if "absolute_error" not in tmp.columns:
        tmp["absolute_error"] = (tmp["predicted_load_mw"] - tmp["actual_load_mw"]).abs()
    else:
        tmp["absolute_error"] = pd.to_numeric(tmp["absolute_error"], errors="coerce")

    return tmp.sort_values("time_utc").reset_index(drop=True)


def find_common_time_window(predictions_by_model: Dict[str, pd.DataFrame]) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    starts = []
    ends = []

    for df in predictions_by_model.values():
        tmp = prepare_prediction_df(df)
        if tmp.empty:
            continue
        starts.append(tmp["time_utc"].min())
        ends.append(tmp["time_utc"].max())

    if not starts or not ends:
        return None, None

    common_start = max(starts)
    common_end = min(ends)

    if common_start > common_end:
        return None, None

    return common_start, common_end


def align_predictions_on_overlap(predictions_by_model: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    common_start, common_end = find_common_time_window(predictions_by_model)
    aligned: Dict[str, pd.DataFrame] = {}

    if common_start is None or common_end is None:
        return aligned

    for model_name, df in predictions_by_model.items():
        tmp = prepare_prediction_df(df)
        tmp = tmp[(tmp["time_utc"] >= common_start) & (tmp["time_utc"] <= common_end)].copy()
        if not tmp.empty:
            aligned[model_name] = tmp

    return aligned


def plot_actual_vs_predicted_lines(
    predictions_by_model: Dict[str, pd.DataFrame],
    output_dir: Path,
    max_models_per_plot: int = 6,
) -> None:
    if not predictions_by_model:
        return

    aligned = align_predictions_on_overlap(predictions_by_model)
    if not aligned:
        print("[WARN] Could not find overlapping time window across prediction files.")
        return

    model_names = sorted(aligned.keys())
    common_actual = aligned[model_names[0]][["time_utc", "actual_load_mw"]].copy()

    # Full comparison plot
    plt.figure(figsize=(14, 7))
    plt.plot(common_actual["time_utc"], common_actual["actual_load_mw"], label="actual", linewidth=2.5)

    for model_name in model_names[:max_models_per_plot]:
        plt.plot(
            aligned[model_name]["time_utc"],
            aligned[model_name]["predicted_load_mw"],
            label=model_name,
            linewidth=1.5,
        )

    plt.title("Actual vs Predicted Load (Common Overlap Window)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "timeseries_actual_vs_predictions_overlap.png", dpi=150)
    plt.close()

    # First week view if enough data
    first_time = common_actual["time_utc"].min()
    cutoff = first_time + pd.Timedelta(days=7)
    plt.figure(figsize=(14, 7))
    short_actual = common_actual[common_actual["time_utc"] <= cutoff]
    plt.plot(short_actual["time_utc"], short_actual["actual_load_mw"], label="actual", linewidth=2.5)

    for model_name in model_names[:max_models_per_plot]:
        short_df = aligned[model_name][aligned[model_name]["time_utc"] <= cutoff]
        plt.plot(
            short_df["time_utc"],
            short_df["predicted_load_mw"],
            label=model_name,
            linewidth=1.5,
        )

    plt.title("Actual vs Predicted Load (First 7 Days of Common Overlap)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "timeseries_actual_vs_predictions_first_7_days.png", dpi=150)
    plt.close()


def plot_error_distributions(predictions_by_model: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    if not predictions_by_model:
        return

    plt.figure(figsize=(12, 7))
    for model_name, df in sorted(predictions_by_model.items()):
        tmp = prepare_prediction_df(df)
        if tmp.empty:
            continue
        err = tmp["predicted_load_mw"] - tmp["actual_load_mw"]
        plt.hist(err, bins=60, alpha=0.45, label=model_name)

    plt.title("Prediction Error Distribution")
    plt.xlabel("Prediction Error (MW)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "hist_error_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 7))
    for model_name, df in sorted(predictions_by_model.items()):
        tmp = prepare_prediction_df(df)
        if tmp.empty:
            continue
        plt.hist(tmp["absolute_error"], bins=60, alpha=0.45, label=model_name)

    plt.title("Absolute Error Distribution")
    plt.xlabel("Absolute Error (MW)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "hist_absolute_error_distribution.png", dpi=150)
    plt.close()


def plot_hourly_error_profiles(predictions_by_model: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    if not predictions_by_model:
        return

    plt.figure(figsize=(12, 7))
    added_any = False

    for model_name, df in sorted(predictions_by_model.items()):
        tmp = prepare_prediction_df(df)
        if tmp.empty:
            continue

        tmp["hour"] = tmp["time_utc"].dt.hour
        hourly = tmp.groupby("hour", as_index=False)["absolute_error"].mean()
        plt.plot(hourly["hour"], hourly["absolute_error"], marker="o", label=model_name)
        added_any = True

    if not added_any:
        plt.close()
        return

    plt.title("Mean Absolute Error by Hour of Day")
    plt.xlabel("Hour of Day (UTC)")
    plt.ylabel("Mean Absolute Error (MW)")
    plt.xticks(range(24))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "line_mae_by_hour.png", dpi=150)
    plt.close()


def plot_daily_error_profiles(predictions_by_model: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    if not predictions_by_model:
        return

    plt.figure(figsize=(14, 7))
    added_any = False

    for model_name, df in sorted(predictions_by_model.items()):
        tmp = prepare_prediction_df(df)
        if tmp.empty:
            continue

        tmp["date"] = tmp["time_utc"].dt.date
        daily = tmp.groupby("date", as_index=False)["absolute_error"].mean()
        plt.plot(pd.to_datetime(daily["date"]), daily["absolute_error"], marker="o", label=model_name)
        added_any = True

    if not added_any:
        plt.close()
        return

    plt.title("Daily Mean Absolute Error")
    plt.xlabel("Date")
    plt.ylabel("Mean Absolute Error (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "line_daily_mae.png", dpi=150)
    plt.close()


def create_overlap_summary(predictions_by_model: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    aligned = align_predictions_on_overlap(predictions_by_model)
    rows = []

    for model_name, tmp in sorted(aligned.items()):
        actual = tmp["actual_load_mw"].astype(float)
        pred = tmp["predicted_load_mw"].astype(float)
        err = pred - actual
        abs_err = err.abs()
        sq_err = err ** 2

        nonzero_actual = actual != 0
        if nonzero_actual.any():
            mape = float(((abs_err[nonzero_actual] / actual[nonzero_actual].abs()) * 100.0).mean())
        else:
            mape = np.nan

        rows.append({
            "model": model_name,
            "rows_in_common_window": len(tmp),
            "common_start_time": tmp["time_utc"].min(),
            "common_end_time": tmp["time_utc"].max(),
            "RMSE_overlap": float(np.sqrt(sq_err.mean())),
            "MAE_overlap": float(abs_err.mean()),
            "MAPE_overlap": mape,
            "BIAS_overlap": float(err.mean()),
        })

    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare forecasting model CSV outputs and generate plots."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing metrics/predictions CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save graphs and summary CSVs. Defaults to <input_dir>/comparison_outputs",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist or is not a directory: {input_dir}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else input_dir / "comparison_outputs"
    )
    safe_mkdir(output_dir)

    metrics_by_model, predictions_by_model = load_csvs(input_dir)

    print(f"[INFO] Found {len(metrics_by_model)} metrics files and {len(predictions_by_model)} prediction files.")

    # Summary tables
    metrics_table = build_metrics_table(metrics_by_model)
    prediction_summary = compute_prediction_summary(predictions_by_model)
    overlap_summary = create_overlap_summary(predictions_by_model)

    if not metrics_table.empty:
        save_dataframe(metrics_table, output_dir / "metrics_summary.csv")
        print(f"[INFO] Wrote metrics summary: {output_dir / 'metrics_summary.csv'}")

    if not prediction_summary.empty:
        save_dataframe(prediction_summary, output_dir / "prediction_summary_recomputed.csv")
        print(f"[INFO] Wrote prediction summary: {output_dir / 'prediction_summary_recomputed.csv'}")

    if not overlap_summary.empty:
        save_dataframe(overlap_summary, output_dir / "prediction_summary_common_overlap.csv")
        print(f"[INFO] Wrote overlap summary: {output_dir / 'prediction_summary_common_overlap.csv'}")

    # Plots
    if not metrics_table.empty:
        plot_metric_bars(metrics_table, output_dir)

    if predictions_by_model:
        plot_actual_vs_predicted_lines(predictions_by_model, output_dir)
        plot_error_distributions(predictions_by_model, output_dir)
        plot_hourly_error_profiles(predictions_by_model, output_dir)
        plot_daily_error_profiles(predictions_by_model, output_dir)

    print(f"[INFO] Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()