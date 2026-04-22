#!/usr/bin/env python3
"""
Build next-day, population-weighted hourly feature forecasts for the CAISO-style dataset.

Inputs
------
1) station_population_weights.csv
   Required columns:
   - region
   - station_name
   - latitude
   - longitude
   - population_weight

2) (Optional) historical dataset CSV with the same schema as the provided training data.
   Used only to backfill load_previous_week from the observation 7 days earlier.

What this script forecasts
--------------------------
Forecasted / deterministic columns:
- region
- time_key
- time_utc
- temperature_2m                  (deg F)
- apparent_temperature            (deg F)
- relative_humidity_2m            (%)
- precipitation                   (mm)
- cloud_cover                     (%)
- wind_speed_10m                  (km/h)
- shortwave_radiation             (W/m^2)
- cdd_65f
- hdd_65f
- is_weekend
- US_federal_holidays
- state_holidays

Not directly forecasted here:
- load_mw                         -> left as NaN
- load_previous_week              -> backfilled from historical CSV if available, else NaN

API source
----------
Open-Meteo Forecast API:
https://open-meteo.com/en/docs

The script queries hourly forecasts in UTC so the resulting rows line up cleanly with
the training dataset's `time_utc` column.
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# These match the forecastable weather columns in the provided dataset.
HOURLY_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
]


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def retry_get_json(
    url: str,
    params: Dict[str, object],
    timeout: int = 60,
    max_retries: int = 5,
    backoff_base: float = 1.5,
) -> Dict:
    """GET JSON with simple exponential backoff."""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                raise requests.HTTPError(f"429 Too Many Requests: {resp.text[:200]}")
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            sleep_s = backoff_base ** (attempt - 1)
            logging.warning(
                "Request failed on attempt %d/%d. Sleeping %.1fs. Error: %s",
                attempt,
                max_retries,
                sleep_s,
                exc,
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_exc}")


def normalize_region_weights(weights_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "region",
        "station_name",
        "latitude",
        "longitude",
        "population_weight",
    }
    missing = required_cols - set(weights_df.columns)
    if missing:
        raise ValueError(f"station weights file is missing required columns: {sorted(missing)}")

    weights_df = weights_df.copy()
    weights_df["population_weight"] = pd.to_numeric(
        weights_df["population_weight"], errors="raise"
    )

    sums = weights_df.groupby("region")["population_weight"].transform("sum")
    if (sums <= 0).any():
        bad_regions = weights_df.loc[sums <= 0, "region"].unique().tolist()
        raise ValueError(f"Non-positive total weights for region(s): {bad_regions}")

    weights_df["population_weight"] = weights_df["population_weight"] / sums
    return weights_df


def infer_target_next_day_utc(target_date_utc: Optional[str]) -> pd.Timestamp:
    """
    Return the UTC date whose 24 hourly rows we want.

    If target_date_utc is None, use tomorrow in UTC.
    """
    if target_date_utc:
        d = pd.Timestamp(target_date_utc)
        if d.tzinfo is not None:
            d = d.tz_convert("UTC").normalize().tz_localize(None)
        else:
            d = d.normalize()
        return d

    now_utc = pd.Timestamp.now(tz="UTC")
    return (now_utc + pd.Timedelta(days=1)).normalize().tz_localize(None)


def fetch_station_forecast_hourly(
    latitude: float,
    longitude: float,
    forecast_days: int = 3,
) -> pd.DataFrame:
    """
    Fetch hourly forecast in UTC for one station.

    We request UTC explicitly so all stations align on the same hourly timestamps.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "forecast_days": forecast_days,
    }

    payload = retry_get_json(OPEN_METEO_FORECAST_URL, params=params)
    hourly = payload.get("hourly")
    if not hourly:
        raise ValueError(f"No hourly forecast returned for ({latitude}, {longitude}).")

    df = pd.DataFrame(hourly)
    if "time" not in df.columns:
        raise ValueError(f"Hourly forecast is missing `time` for ({latitude}, {longitude}).")

    df = df.rename(columns={"time": "time_utc"})
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)

    # Ensure all expected variables exist.
    for col in HOURLY_VARS:
        if col not in df.columns:
            raise ValueError(
                f"Open-Meteo response for ({latitude}, {longitude}) is missing column: {col}"
            )

    return df[["time_utc"] + HOURLY_VARS].copy()


def build_station_level_forecasts(weights_df: pd.DataFrame) -> pd.DataFrame:
    station_frames: List[pd.DataFrame] = []

    for row in weights_df.itertuples(index=False):
        logging.info(
            "Fetching forecast for region=%s station=%s (lat=%.4f lon=%.4f weight=%.6f)",
            row.region,
            row.station_name,
            row.latitude,
            row.longitude,
            row.population_weight,
        )
        station_df = fetch_station_forecast_hourly(
            latitude=float(row.latitude),
            longitude=float(row.longitude),
            forecast_days=3,
        )
        station_df["region"] = row.region
        station_df["station_name"] = row.station_name
        station_df["population_weight"] = float(row.population_weight)
        station_frames.append(station_df)

    if not station_frames:
        raise ValueError("No stations found in weights file.")

    station_forecasts = pd.concat(station_frames, ignore_index=True)
    return station_forecasts


def weighted_average_by_region_hour(station_forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute population-weighted averages for each region and hour.
    """
    def agg(group: pd.DataFrame) -> pd.Series:
        weights = group["population_weight"].to_numpy(dtype=float)
        out = {}
        for col in HOURLY_VARS:
            vals = group[col].to_numpy(dtype=float)
            mask = ~pd.isna(vals)
            if mask.sum() == 0:
                out[col] = math.nan
            else:
                w = weights[mask]
                v = vals[mask]
                w = w / w.sum()
                out[col] = float((v * w).sum())
        return pd.Series(out)

    result = (
        station_forecasts
        .groupby(["region", "time_utc"], as_index=False)
        .apply(agg, include_groups=False)
        .reset_index()
    )

    # Depending on pandas version, groupby.apply may leave a synthetic level/index.
    if "level_0" in result.columns:
        result = result.drop(columns=["level_0"])
    if "index" in result.columns and set(HOURLY_VARS).issubset(result.columns):
        # safe no-op if not synthetic
        pass

    # Keep only desired columns and sort
    keep_cols = ["region", "time_utc"] + HOURLY_VARS
    result = result[keep_cols].sort_values(["region", "time_utc"]).reset_index(drop=True)
    return result


def add_degree_day_features(df: pd.DataFrame, base_f: float = 65.0) -> pd.DataFrame:
    df = df.copy()
    df["cdd_65f"] = (df["temperature_2m"] - base_f).clip(lower=0)
    df["hdd_65f"] = (base_f - df["temperature_2m"]).clip(lower=0)
    return df


def california_state_holiday_indicator(dates: Iterable[pd.Timestamp]) -> pd.Series:
    """
    Try to compute California state holidays using the `holidays` package.
    Fallback: use US federal holidays if the package is unavailable.

    Install with:
        pip install holidays
    """
    dates = pd.to_datetime(pd.Series(list(dates))).dt.normalize()

    try:
        import holidays  # type: ignore

        years = sorted(set(dates.dt.year.tolist()))
        ca_holidays = holidays.US(state="CA", years=years)
        return dates.isin(pd.to_datetime(list(ca_holidays.keys()))).astype(int)
    except Exception as exc:
        logging.warning(
            "Could not import/use `holidays` for California state holidays. "
            "Falling back to US federal holidays only. Error: %s",
            exc,
        )
        cal = USFederalHolidayCalendar()
        start = dates.min()
        end = dates.max() + pd.Timedelta(days=1)
        federal = cal.holidays(start=start, end=end)
        return dates.isin(federal).astype(int)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["time_utc"], utc=False)
    dates = ts.dt.normalize()

    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    cal = USFederalHolidayCalendar()
    federal_holidays = cal.holidays(start=dates.min(), end=dates.max() + pd.Timedelta(days=1))
    df["US_federal_holidays"] = dates.isin(federal_holidays).astype(int)

    df["state_holidays"] = california_state_holiday_indicator(dates)
    return df


def add_time_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["time_utc"], utc=False)
    df["time_key"] = ts.dt.strftime("%Y-%m-%d:%H")
    return df


def backfill_load_previous_week(
    forecast_df: pd.DataFrame,
    historical_csv_path: Optional[Path],
) -> pd.DataFrame:
    """
    Fill load_previous_week from `load_mw` observed exactly 7 days earlier in the historical data.

    If no historical CSV is supplied or if the needed hour is unavailable, values remain NaN.
    """
    forecast_df = forecast_df.copy()
    forecast_df["load_previous_week"] = pd.NA

    if historical_csv_path is None:
        logging.info("No historical dataset provided; leaving load_previous_week as NaN.")
        return forecast_df

    hist = pd.read_csv(historical_csv_path, usecols=["region", "time_utc", "load_mw"])
    hist["time_utc"] = pd.to_datetime(hist["time_utc"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)

    lookup = hist.rename(columns={"load_mw": "load_previous_week"}).copy()
    lookup["time_utc"] = lookup["time_utc"] + pd.Timedelta(days=7)

    merged = forecast_df.merge(
        lookup,
        on=["region", "time_utc"],
        how="left",
        suffixes=("", "_hist"),
    )

    if "load_previous_week_hist" in merged.columns:
        merged["load_previous_week"] = merged["load_previous_week_hist"]
        merged = merged.drop(columns=["load_previous_week_hist"])

    return merged


def finalize_schema(
    df: pd.DataFrame,
    target_date_utc: pd.Timestamp,
) -> pd.DataFrame:
    df = df.copy()

    # Keep only the target UTC day (24 hours).
    start = target_date_utc
    end = target_date_utc + pd.Timedelta(days=1)
    mask = (df["time_utc"] >= start) & (df["time_utc"] < end)
    df = df.loc[mask].copy()

    df = add_degree_day_features(df)
    df = add_calendar_features(df)
    df = add_time_keys(df)

    # `load_mw` is the target / unknown future realized load, so keep as NaN.
    df["load_mw"] = pd.NA

    final_cols = [
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

    # Make sure every column exists.
    for col in final_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[final_cols].sort_values(["region", "time_utc"]).reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate next-day population-weighted hourly feature forecasts."
    )
    parser.add_argument(
        "--weights-csv",
        type=Path,
        required=True,
        help="Path to station_population_weights.csv",
    )
    parser.add_argument(
        "--historical-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to historical dataset CSV with `region`, `time_utc`, and `load_mw`. "
            "Used to backfill `load_previous_week`."
        ),
    )
    parser.add_argument(
        "--target-date-utc",
        type=str,
        default=None,
        help=(
            "UTC date to generate in YYYY-MM-DD format. "
            "Default: tomorrow in UTC."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Where to write the forecast dataset CSV.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    target_date_utc = infer_target_next_day_utc(args.target_date_utc)
    logging.info("Target UTC date: %s", target_date_utc.date())

    weights_df = pd.read_csv(args.weights_csv)
    weights_df = normalize_region_weights(weights_df)

    station_forecasts = build_station_level_forecasts(weights_df)
    region_hourly = weighted_average_by_region_hour(station_forecasts)

    forecast_df = finalize_schema(region_hourly, target_date_utc)
    forecast_df = backfill_load_previous_week(forecast_df, args.historical_csv)

    # Re-assert final schema ordering after merge.
    final_cols = [
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
    forecast_df = forecast_df[final_cols].sort_values(["region", "time_utc"]).reset_index(drop=True)

    # Sanity checks
    expected_regions = sorted(weights_df["region"].unique().tolist())
    actual_regions = sorted(forecast_df["region"].unique().tolist())
    if actual_regions != expected_regions:
        raise ValueError(f"Region mismatch. expected={expected_regions} actual={actual_regions}")

    counts = forecast_df.groupby("region").size()
    bad_counts = counts[counts != 24]
    if not bad_counts.empty:
        raise ValueError(
            "Each region should have exactly 24 hourly rows for the target UTC day. "
            f"Bad counts: {bad_counts.to_dict()}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(args.output_csv, index=False)

    logging.info("Wrote %d rows to %s", len(forecast_df), args.output_csv)
    logging.info("Rows per region: %s", counts.to_dict())


if __name__ == "__main__":
    main()
