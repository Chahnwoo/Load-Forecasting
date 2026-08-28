#!/usr/bin/env python3
"""Compute paired, operating-day diagnostics from frozen strict predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

TIME = "time_utc"
TARGET = "actual_load_mw"
DEFAULT_PREDICTIONS = Path(
    "data/backtesting/evaluation/strict_oct_nov_to_dec/predictions.csv"
)
METHODS = {
    "ridge": "ridge_prediction",
    "raw_caiso": "caiso_dam_forecast_mw",
    "linearly_calibrated_caiso": "caiso_linearly_calibrated_prediction",
    "previous_week": "previous_week_prediction",
}
COMPARISONS = (
    ("ridge", "raw_caiso"),
    ("ridge", "linearly_calibrated_caiso"),
    ("ridge", "previous_week"),
)


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return math.sqrt(float(np.mean((prediction - actual) ** 2)))


def analyze(frame: pd.DataFrame, *, resamples: int = 20_000, seed: int = 202512,
            timezone: str = "America/Los_Angeles") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return daily rows, comparison summaries, and machine-readable metadata."""
    required = {TIME, TARGET, *(METHODS.values())}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"predictions are missing required columns: {missing}")
    data = frame[list(required)].copy()
    data[TIME] = pd.to_datetime(data[TIME], utc=True, errors="coerce")
    numeric = [TARGET, *METHODS.values()]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    if data.isna().any().any() or data[TIME].duplicated().any():
        raise ValueError("predictions must have unique valid timestamps and no null values")
    if not np.isfinite(data[numeric].to_numpy()).all():
        raise ValueError("predictions must contain only finite numeric values")
    data["operating_day"] = data[TIME].dt.tz_convert(timezone).dt.date.astype(str)

    daily_records: list[dict] = []
    for day, part in data.groupby("operating_day", sort=True):
        actual = part[TARGET].to_numpy()
        values = {name: rmse(actual, part[column].to_numpy()) for name, column in METHODS.items()}
        record: dict[str, object] = {"operating_day": day, "hours": len(part)}
        record.update({f"{name}_rmse_mw": value for name, value in values.items()})
        for challenger, baseline in COMPARISONS:
            record[f"{challenger}_improvement_vs_{baseline}_pct"] = (
                (values[baseline] - values[challenger]) / values[baseline] * 100
            )
            record[f"{challenger}_beats_{baseline}"] = values[challenger] < values[baseline]
        daily_records.append(record)
    daily = pd.DataFrame(daily_records)

    rng = np.random.default_rng(seed)
    days = [part for _, part in data.groupby("operating_day", sort=True)]
    sampled_days = rng.integers(0, len(days), size=(resamples, len(days)))
    summaries = []
    for challenger, baseline in COMPARISONS:
        challenger_column, baseline_column = METHODS[challenger], METHODS[baseline]
        actual = data[TARGET].to_numpy()
        overall_baseline = rmse(actual, data[baseline_column].to_numpy())
        estimate = ((overall_baseline - rmse(actual, data[challenger_column].to_numpy())) /
                    overall_baseline)
        counts = np.asarray([len(day) for day in days])
        base_sse = np.asarray([np.square(
            day[baseline_column].to_numpy() - day[TARGET].to_numpy()).sum() for day in days])
        challenger_sse = np.asarray([np.square(
            day[challenger_column].to_numpy() - day[TARGET].to_numpy()).sum() for day in days])
        sample_counts = counts[sampled_days].sum(axis=1)
        base_sample_rmse = np.sqrt(base_sse[sampled_days].sum(axis=1) / sample_counts)
        challenger_sample_rmse = np.sqrt(
            challenger_sse[sampled_days].sum(axis=1) / sample_counts)
        samples = (base_sample_rmse - challenger_sample_rmse) / base_sample_rmse * 100
        daily_improvement = daily[f"{challenger}_improvement_vs_{baseline}_pct"]
        summaries.append({
            "challenger": challenger, "baseline": baseline,
            "overall_challenger_rmse_mw": rmse(actual, data[challenger_column].to_numpy()),
            "overall_baseline_rmse_mw": rmse(actual, data[baseline_column].to_numpy()),
            "rmse_improvement_pct": estimate * 100,
            "bootstrap_ci_lower_pct": float(np.quantile(samples, 0.025)),
            "bootstrap_ci_upper_pct": float(np.quantile(samples, 0.975)),
            "operating_day_wins": int(daily[f"{challenger}_beats_{baseline}"].sum()),
            "operating_days": len(days),
            "median_daily_rmse_improvement_pct": float(daily_improvement.median()),
        })
    summary = pd.DataFrame(summaries)
    metadata = {"bootstrap_resamples": resamples, "seed": seed,
                "operating_day_timezone": timezone, "rows": len(data),
                "operating_days": len(days), "comparisons": summary.to_dict("records")}
    return daily, summary, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: the predictions file's directory")
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=202512)
    parser.add_argument("--timezone", default="America/Los_Angeles")
    args = parser.parse_args()
    if args.bootstrap_resamples <= 0:
        parser.error("--bootstrap-resamples must be positive")
    daily, summary, metadata = analyze(
        pd.read_csv(args.predictions), resamples=args.bootstrap_resamples,
        seed=args.seed, timezone=args.timezone)
    output = args.output_dir or args.predictions.parent
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_paired_diagnostics.csv", index=False)
    summary.to_csv(output / "bootstrap_summary.csv", index=False)
    (output / "bootstrap_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
