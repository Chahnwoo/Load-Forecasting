#!/usr/bin/env python3
"""
merge_collected_data.py

Combine raw CAISO dataset CSVs into one long processed CSV.

Expected input directory:

  data/raw/

Expected input filenames:

  caiso_dataset_YYYYMMDD_to_YYYYMMDD.csv

Default output directory:

  data/processed/

Automatic output filename:

  caiso_dataset_<earliest_start>_to_<latest_end>.csv

Example:

  data/raw/caiso_dataset_20200101_to_20201231.csv
  data/raw/caiso_dataset_20260424_to_20260424.csv

produces:

  data/processed/caiso_dataset_20200101_to_20260424.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Optional

import pandas as pd


DATASET_RE = re.compile(
    r"^caiso_dataset_(\d{8})_to_(\d{8})\.csv$"
)


@dataclass(frozen=True)
class DatasetFile:
    path: str
    start_date: date
    end_date: date


def parse_dataset_filename(path: str) -> Optional[DatasetFile]:
    basename = os.path.basename(path)
    m = DATASET_RE.match(basename)
    if not m:
        return None

    start_s, end_s = m.group(1), m.group(2)
    start_date = datetime.strptime(start_s, "%Y%m%d").date()
    end_date = datetime.strptime(end_s, "%Y%m%d").date()

    if end_date < start_date:
        raise ValueError(f"{path} has end date before start date")

    return DatasetFile(
        path=path,
        start_date=start_date,
        end_date=end_date,
    )


def find_input_files(raw_dir: str) -> List[DatasetFile]:
    """
    Find raw files matching:

      caiso_dataset_YYYYMMDD_to_YYYYMMDD.csv

    Only searches the raw directory.
    """
    pattern = os.path.join(raw_dir, "caiso_dataset_*_to_*.csv")
    raw_paths = sorted(glob.glob(pattern))

    dataset_files: List[DatasetFile] = []

    for path in raw_paths:
        parsed = parse_dataset_filename(path)
        if parsed is None:
            continue
        dataset_files.append(parsed)

    dataset_files.sort(key=lambda x: (x.start_date, x.end_date, x.path))
    return dataset_files


def infer_output_path(
    dataset_files: List[DatasetFile],
    *,
    processed_dir: str,
) -> str:
    if not dataset_files:
        raise RuntimeError("Cannot infer output path because no input files were found.")

    min_start = min(f.start_date for f in dataset_files)
    max_end = max(f.end_date for f in dataset_files)

    filename = (
        f"caiso_dataset_"
        f"{min_start.strftime('%Y%m%d')}_to_"
        f"{max_end.strftime('%Y%m%d')}.csv"
    )

    return os.path.join(processed_dir, filename)


def validate_columns(df: pd.DataFrame, path: str) -> None:
    required = {"region", "time_utc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def load_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_columns(df, path)

    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")

    bad = int(df["time_utc"].isna().sum())
    if bad > 0:
        raise ValueError(f"{path} has {bad} invalid time_utc values")

    return df


def join_datasets(dataset_files: List[DatasetFile]) -> pd.DataFrame:
    if not dataset_files:
        raise RuntimeError("No raw input CSV files found.")

    parts = []

    for dataset_file in dataset_files:
        print(
            f"Loading: {dataset_file.path} "
            f"({dataset_file.start_date} to {dataset_file.end_date})"
        )
        df = load_one_csv(dataset_file.path)
        parts.append(df)

    combined = pd.concat(parts, ignore_index=True)

    before = len(combined)

    combined = (
        combined
        .drop_duplicates(subset=["region", "time_utc"], keep="last")
        .copy()
    )

    after = len(combined)

    combined = (
        combined
        .sort_values(["region", "time_utc"])
        .reset_index(drop=True)
    )

    print(f"Rows before dedup: {before:,}")
    print(f"Rows after dedup:  {after:,}")
    print(f"Removed duplicates: {before - after:,}")

    return combined


def summarize_coverage(combined: pd.DataFrame) -> None:
    print("\nCoverage by region:")

    rows = []

    for region, g in combined.groupby("region", sort=True):
        rows.append({
            "region": region,
            "first_time_utc": g["time_utc"].min(),
            "last_time_utc": g["time_utc"].max(),
            "rows": len(g),
            "missing_load_rows": int(g["load_mw"].isna().sum()) if "load_mw" in g.columns else None,
        })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--raw-dir",
        default="./data/raw",
        help="Directory containing raw CAISO dataset CSVs",
    )

    ap.add_argument(
        "--processed-dir",
        default="./data/processed",
        help="Directory where merged processed CSV should be written",
    )

    ap.add_argument(
        "--output",
        default=None,
        help=(
            "Optional explicit output path. "
            "If omitted, output name is inferred from available raw dataset dates."
        ),
    )

    args = ap.parse_args()

    dataset_files = find_input_files(args.raw_dir)

    if not dataset_files:
        raise RuntimeError(
            f"No files matching caiso_dataset_YYYYMMDD_to_YYYYMMDD.csv found in {args.raw_dir}"
        )

    output_path = args.output or infer_output_path(
        dataset_files,
        processed_dir=args.processed_dir,
    )

    print("Found raw input files:")
    for f in dataset_files:
        print(f"  - {f.path} ({f.start_date} to {f.end_date})")

    print(f"\nOutput path: {output_path}")

    combined = join_datasets(dataset_files)

    summarize_coverage(combined)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    out = combined.copy()
    out["time_utc"] = out["time_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(output_path, index=False)

    print(f"\nSaved merged processed dataset to: {output_path}")
    print(f"Final rows: {len(out):,}")
    print(f"Final columns: {len(out.columns)}")


if __name__ == "__main__":
    main()
