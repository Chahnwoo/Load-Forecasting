#!/usr/bin/env python3
"""
join_caiso_datasets.py

Combine multiple CAISO dataset CSVs into one long CSV.

Example input files:
  caiso_dataset_20200101_to_20201231.csv
  caiso_dataset_20210101_to_20211231.csv
  caiso_dataset_20220101_to_20221231.csv
  caiso_dataset_20230101_to_20231231.csv
  caiso_dataset_20240101_to_20241231.csv

Output:
  caiso_dataset_20200101_to_20241231.csv
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List

import pandas as pd


def find_input_files(data_dir: str) -> List[str]:
    """
    Find files matching:
      caiso_dataset_YYYYMMDD_to_YYYYMMDD.csv
    """
    pattern = os.path.join(data_dir, "caiso_dataset_*_to_*.csv")
    files = sorted(glob.glob(pattern))

    # Avoid accidentally including the final merged output if it already exists
    files = [
        f for f in files
        if os.path.basename(f) != "caiso_dataset_20200101_to_20241231.csv"
    ]
    return files


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


def join_datasets(paths: List[str]) -> pd.DataFrame:
    if not paths:
        raise RuntimeError("No input CSV files found.")

    parts = []
    for path in paths:
        print(f"Loading: {path}")
        df = load_one_csv(path)
        parts.append(df)

    combined = pd.concat(parts, ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset=["region", "time_utc"], keep="last").copy()
    after = len(combined)

    combined = combined.sort_values(["region", "time_utc"]).reset_index(drop=True)

    print(f"Rows before dedup: {before:,}")
    print(f"Rows after dedup:  {after:,}")
    print(f"Removed duplicates: {before - after:,}")

    return combined


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default="./data",
        help="Directory containing yearly CAISO dataset CSVs",
    )
    ap.add_argument(
        "--output",
        default="./data/caiso_dataset_20200101_to_20260421.csv",
        help="Output merged CSV path",
    )
    args = ap.parse_args()

    files = find_input_files(args.data_dir)
    print("Found files:")
    for f in files:
        print(f"  - {f}")

    combined = join_datasets(files)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    out = combined.copy()
    out["time_utc"] = out["time_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(args.output, index=False)

    print(f"\nSaved merged dataset to: {args.output}")
    print(f"Final rows: {len(out):,}")
    print(f"Final columns: {len(out.columns)}")


if __name__ == "__main__":
    main()