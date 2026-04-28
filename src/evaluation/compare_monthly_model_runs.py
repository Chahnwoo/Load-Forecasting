#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# These are the metric columns we will look for in the metrics CSV files.
METRIC_COLUMNS = ["RMSE", "MAE", "MAPE", "BIAS", "MSE"]

# Required columns for prediction CSV files.
PREDICTION_REQUIRED_COLUMNS = {"time_utc", "actual_load_mw", "predicted_load_mw"}


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


def parse_run_filename(path: Path) -> Optional[Tuple[str, str, str]]:
    """
    Parse filenames of the form:
      {model}_YYYY-MM_metrics.csv
      {model}_YYYY-MM_predictions.csv

    Returns:
      (model, month, kind)
    where kind is either "metrics" or "predictions".

    Example:
      hinge_regression_2025-12_metrics.csv
      -> ("hinge_regression", "2025-12", "metrics")
    """
    pattern = r"^(?P<model>.+)_(?P<month>\d{4}-\d{2})_(?P<kind>metrics|predictions)\.csv$"
    m = re.match(pattern, path.name)
    if not m:
        return None

    model = normalize_model_name(m.group("model"))
    month = m.group("month")
    kind = m.group("kind")
    return model, month, kind


def month_to_timestamp(month_str: str) -> pd.Timestamp:
    return pd.to_datetime(month_str, format="%Y-%m")


def classify_prediction_csv(df: pd.DataFrame) -> bool:
    return PREDICTION_REQUIRED_COLUMNS.issubset(set(df.columns))


def load_monthly_files(
    input_dir: Path,
) -> Tuple[Dict[Tuple[str, str], pd.DataFrame], Dict[Tuple[str, str], pd.DataFrame]]:
    """
    Returns:
      metrics_by_key[(model, month)] = df
      predictions_by_key[(model, month)] = df
    """
    metrics_by_key: Dict[Tuple[str, str], pd.DataFrame] = {}
    predictions_by_key: Dict[Tuple[str, str], pd.DataFrame] = {}

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {input_dir}")

    for csv_path in csv_files:
        parsed = parse_run_filename(csv_path)
        if parsed is None:
            print(f"[WARN] Skipping file with unexpected name format: {csv_path.name}")
            continue

        model, month, kind = parsed

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[WARN] Failed to read {csv_path.name}: {exc}")
            continue

        if kind == "metrics":
            metrics_by_key[(model, month)] = df.copy()

        elif kind == "predictions":
            if not classify_prediction_csv(df):
                print(
                    f"[WARN] Skipping predictions file with unexpected columns: {csv_path.name}"
                )
                continue
            predictions_by_key[(model, month)] = df.copy()

    return metrics_by_key, predictions_by_key


def build_monthly_metrics_table(
    metrics_by_key: Dict[Tuple[str, str], pd.DataFrame]
) -> pd.DataFrame:
    """
    Reads monthly metrics CSVs and creates one summary row per (model, month).
    """
    rows = []

    for (model, month), df in metrics_by_key.items():
        if df.empty:
            continue

        # We assume the file has one summary row, but if there are multiple rows,
        # we take the first row.
        row_dict = df.iloc[0].to_dict()

        out = {
            "model": model,
            "month": month,
            "month_ts": month_to_timestamp(month),
        }

        for col in df.columns:
            out[col] = row_dict.get(col)

        # Force model and month from the filename, since that is the reliable source now.
        out["model"] = model
        out["month"] = month
        out["month_ts"] = month_to_timestamp(month)

        rows.append(out)

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)

    # Reorder columns nicely.
    preferred_order = (
        ["model", "month", "month_ts"]
        + [c for c in METRIC_COLUMNS if c in summary.columns]
        + [c for c in summary.columns if c not in {"model", "month", "month_ts", *METRIC_COLUMNS}]
    )

    summary = summary[[c for c in preferred_order if c in summary.columns]]
    summary = summary.sort_values(["month_ts", "model"]).reset_index(drop=True)
    return summary


def prepare_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["time_utc"] = pd.to_datetime(tmp["time_utc"], errors="coerce")
    tmp["actual_load_mw"] = pd.to_numeric(tmp["actual_load_mw"], errors="coerce")
    tmp["predicted_load_mw"] = pd.to_numeric(tmp["predicted_load_mw"], errors="coerce")
    tmp = tmp.dropna(subset=["time_utc", "actual_load_mw", "predicted_load_mw"]).copy()
    return tmp.sort_values("time_utc").reset_index(drop=True)


def recompute_monthly_metrics_from_predictions(
    predictions_by_key: Dict[Tuple[str, str], pd.DataFrame]
) -> pd.DataFrame:
    """
    Recompute metrics from predictions CSVs, producing one row per (model, month).
    """
    rows = []

    for (model, month), df in predictions_by_key.items():
        tmp = prepare_prediction_df(df)
        if tmp.empty:
            continue

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

        rows.append(
            {
                "model": model,
                "month": month,
                "month_ts": month_to_timestamp(month),
                "rows": len(tmp),
                "start_time": tmp["time_utc"].min(),
                "end_time": tmp["time_utc"].max(),
                "RMSE": float(np.sqrt(sq_err.mean())),
                "MAE": float(abs_err.mean()),
                "MAPE": mape,
                "BIAS": float(err.mean()),
                "MSE": float(sq_err.mean()),
                "MEAN_ACTUAL": float(actual.mean()),
                "MEAN_PREDICTED": float(pred.mean()),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["month_ts", "model"]).reset_index(drop=True)
    return out


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def plot_metric_over_months(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """
    Creates a line plot with:
      x-axis = month
      y-axis = metric
      one line per model
    """
    if df.empty or metric not in df.columns:
        return

    plot_df = df[["model", "month", "month_ts", metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])

    if plot_df.empty:
        return

    plt.figure(figsize=(12, 7))

    for model in sorted(plot_df["model"].unique()):
        sub = plot_df[plot_df["model"] == model].sort_values("month_ts")
        if sub.empty:
            continue

        plt.plot(
            sub["month_ts"],
            sub[metric],
            marker="o",
            label=model,
        )

    title = f"{metric} Across Months"
    if title_suffix:
        title += f" ({title_suffix})"

    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_all_metric_lines(df: pd.DataFrame, output_dir: Path, prefix: str = "line") -> None:
    if df.empty:
        return

    for metric in METRIC_COLUMNS:
        if metric in df.columns:
            output_path = output_dir / f"{prefix}_{metric.lower()}.png"
            plot_metric_over_months(df, metric, output_path)


def plot_model_rank_by_month(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    """
    Optional helper plot:
    rank each model by metric within each month (lower is better).
    """
    if df.empty or metric not in df.columns:
        return

    tmp = df[["model", "month", "month_ts", metric]].copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp = tmp.dropna(subset=[metric])

    if tmp.empty:
        return

    tmp["rank"] = tmp.groupby("month_ts")[metric].rank(method="min", ascending=True)

    plt.figure(figsize=(12, 7))
    for model in sorted(tmp["model"].unique()):
        sub = tmp[tmp["model"] == model].sort_values("month_ts")
        plt.plot(sub["month_ts"], sub["rank"], marker="o", label=model)

    plt.title(f"Model Rank Across Months ({metric}, lower is better)")
    plt.xlabel("Month")
    plt.ylabel("Rank")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate monthly model comparison plots from metrics/predictions CSV files."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing files like {model}_YYYY-MM_metrics.csv and {model}_YYYY-MM_predictions.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where output CSVs/plots will be saved. Default: <input_dir>/comparison_outputs",
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

    metrics_by_key, predictions_by_key = load_monthly_files(input_dir)

    print(f"[INFO] Found {len(metrics_by_key)} monthly metrics files.")
    print(f"[INFO] Found {len(predictions_by_key)} monthly predictions files.")

    # Build monthly summary from metrics CSVs
    monthly_metrics = build_monthly_metrics_table(metrics_by_key)
    if not monthly_metrics.empty:
        save_dataframe(monthly_metrics, output_dir / "monthly_metrics_summary.csv")
        print(f"[INFO] Wrote: {output_dir / 'monthly_metrics_summary.csv'}")

        # Line charts from metrics CSVs
        plot_all_metric_lines(monthly_metrics, output_dir, prefix="line")
        print("[INFO] Wrote line plots from metrics CSVs.")

        # Optional rank plot for RMSE
        if "RMSE" in monthly_metrics.columns:
            plot_model_rank_by_month(
                monthly_metrics,
                metric="RMSE",
                output_path=output_dir / "line_rank_rmse.png",
            )
            print(f"[INFO] Wrote: {output_dir / 'line_rank_rmse.png'}")
    else:
        print("[INFO] No monthly metrics summary could be built.")

    # Recompute monthly metrics from prediction files
    recomputed = recompute_monthly_metrics_from_predictions(predictions_by_key)
    if not recomputed.empty:
        save_dataframe(recomputed, output_dir / "monthly_metrics_recomputed_from_predictions.csv")
        print(f"[INFO] Wrote: {output_dir / 'monthly_metrics_recomputed_from_predictions.csv'}")

        # Separate line charts from recomputed metrics
        plot_all_metric_lines(recomputed, output_dir, prefix="line_recomputed")
        print("[INFO] Wrote line plots from predictions CSVs.")

    print(f"[INFO] Done. All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()