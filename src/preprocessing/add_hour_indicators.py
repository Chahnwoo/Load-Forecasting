#!/usr/bin/env python3

import argparse
import pandas as pd


def add_hour_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 24 indicator columns, one for each hour of the day, based on `time_key`.

    Expected `time_key` format:
        YYYY-MM-DD:HH

    Example:
        2026-04-23:17  ->  hour_17 = 1, all other hour_* columns = 0
    """
    if "time_key" not in df.columns:
        raise ValueError("Input file must contain a 'time_key' column.")

    # Parse the hour from the last two characters of time_key
    hour_series = df["time_key"].astype(str).str[-2:]

    # Validate hour format
    invalid_mask = ~hour_series.str.fullmatch(r"\d{2}")
    if invalid_mask.any():
        bad_examples = df.loc[invalid_mask, "time_key"].head(10).tolist()
        raise ValueError(
            "Some 'time_key' values do not end in a valid 2-digit hour. "
            f"Examples: {bad_examples}"
        )

    hours = hour_series.astype(int)

    # Validate hour range
    out_of_range_mask = (hours < 0) | (hours > 23)
    if out_of_range_mask.any():
        bad_examples = df.loc[out_of_range_mask, "time_key"].head(10).tolist()
        raise ValueError(
            "Some parsed hours are outside the range 00-23. "
            f"Examples: {bad_examples}"
        )

    # Add one-hot / indicator columns
    for h in range(24):
        col_name = f"hour_{h:02d}"
        df[col_name] = (hours == h).astype(int)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Add 24 hour-of-day indicator variables to a CSV dataset."
    )
    parser.add_argument(
        "input_csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        default="with_hour_indicators.csv",
        help="Path to output CSV file (default: with_hour_indicators.csv)"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Optional sanity check against expected columns
    expected_columns = [
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
    ]

    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        print("Warning: the following expected columns were not found:")
        for col in missing:
            print(f"  - {col}")
        print("Continuing anyway, as only 'time_key' is required.")

    df = add_hour_indicators(df)

    df.to_csv(args.output_csv, index=False)
    print(f"Saved updated dataset to: {args.output_csv}")


if __name__ == "__main__":
    main()