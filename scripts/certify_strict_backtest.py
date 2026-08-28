#!/usr/bin/env python3
"""Read-only certification checks for built strict backtest months and raw caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.acquire_december_2025_caiso import raw_identifiers, read_oasis_zip
from scripts.acquire_gdex_gfs_0p25 import cached_subset
from src.backtesting.acquisition import sha256_file
from src.backtesting.strict_dataset import (
    ACTUAL_ITEM, DAM_ITEM, ENTITY, normalize_caiso, operating_intervals,
)

EXPECTED_ROWS = {"2025-10": 744, "2025-11": 721, "2025-12": 744}
EXPECTED_WEATHER = {"2025-10": 3720, "2025-11": 3605, "2025-12": 3720}


def _utc(frame: pd.DataFrame, column: str, product: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"{product} is missing {column}")
    result = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if result.isna().any():
        raise ValueError(f"{product} contains invalid {column}")
    return result


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"missing required product: {path}")
    return pd.read_csv(path)


def certify_month(root: Path, month: str, *, raw_gdex: bool = False,
                  raw_caiso: bool = False) -> dict:
    """Fail closed on a strict product invariant and return certified counts."""
    strict = root / month / "processed/strict"
    rows = _read(strict / "strict_rows.csv")
    actuals = _read(strict / "actuals.csv")
    dam = _read(strict / "dam_forecasts.csv")
    weather = _read(strict / "weather_vintages.csv")
    expected = operating_intervals(month)
    count = EXPECTED_ROWS.get(month, len(expected))
    if len(expected) != count or len(rows) != count:
        raise ValueError(f"{month}: expected {count} strict rows, found {len(rows)}")
    times = _utc(rows, "time_utc", "strict_rows")
    if times.duplicated().any() or set(times) != set(expected.time_utc):
        raise ValueError(f"{month}: strict target hours are not exact and unique")
    operating_day = rows.get("operating_day")
    if operating_day is None or operating_day.nunique() != expected.operating_day.nunique():
        raise ValueError(f"{month}: operating-day coverage is incomplete")
    cutoffs = _utc(rows, "forecast_cutoff_utc", "strict_rows")
    if rows.assign(_cutoff=cutoffs).groupby("operating_day")["_cutoff"].nunique().ne(1).any():
        raise ValueError(f"{month}: an operating day has more than one cutoff")
    weather_available = _utc(rows, "weather_available_at_utc", "strict_rows")
    load_available = _utc(rows, "load_previous_week_available_time_utc", "strict_rows")
    load_source = _utc(rows, "load_previous_week_source_time_utc", "strict_rows")
    if (weather_available > cutoffs).any() or (load_available > cutoffs).any():
        raise ValueError(f"{month}: a feature was unavailable at its cutoff")
    if not load_source.eq(times - pd.Timedelta(days=7)).all():
        raise ValueError(f"{month}: previous-week load is not the exact UTC hour seven days earlier")
    if rows.isna().any().any() or actuals.isna().any().any() or dam.isna().any().any() or weather.isna().any().any():
        raise ValueError(f"{month}: strict products contain nulls")
    actual = normalize_caiso(actuals, ACTUAL_ITEM)
    forecast = normalize_caiso(dam, DAM_ITEM)
    if len(actual) != count or len(forecast) != count:
        raise ValueError(f"{month}: DAM/ACTUAL coverage is not exact")
    actual_times, dam_times = set(actual.interval_start_utc), set(forecast.interval_start_utc)
    if actual_times != set(times) or dam_times != set(times):
        raise ValueError(f"{month}: DAM/ACTUAL target keys differ from strict rows")
    valid = _utc(weather, "valid_time_utc", "weather_vintages")
    available = _utc(weather, "available_at_utc", "weather_vintages")
    cutoff_by_time = dict(zip(times, cutoffs))
    if any(available.iloc[i] > cutoff_by_time.get(valid.iloc[i], pd.Timestamp.min.tz_localize("UTC"))
           for i in range(len(weather))):
        raise ValueError(f"{month}: selected station weather was unavailable at cutoff")
    station_count = weather["station"].nunique() if "station" in weather else 0
    expected_weather = EXPECTED_WEATHER.get(month, count * station_count)
    if len(weather) != expected_weather or station_count != 5 or valid.nunique() != count:
        raise ValueError(f"{month}: expected {expected_weather} selected weather rows, 5 stations, and {count} hours")

    report = {"month": month, "strict_rows": len(rows),
              "operating_days": int(operating_day.nunique()),
              "dam_targets": len(forecast), "actual_targets": len(actual),
              "selected_weather_rows": len(weather), "stations": station_count,
              "selected_weather_hours": int(valid.nunique())}
    if raw_gdex:
        files = sorted((root / month / "raw/gdex_ncss").glob("*.nc"))
        for path in files:
            cached_subset(path, path.name[:-3])
        sidecars = list((root / month / "raw/gdex_ncss").glob("*.nc.provenance.json"))
        if len(files) != len(sidecars):
            raise ValueError(f"{month}: raw GDEX cache has incomplete file/sidecar pairs")
        report.update(raw_gdex_subsets=len(files), raw_gdex_sidecars=len(sidecars),
                      raw_gdex_deep_validated=len(files))
    if raw_caiso:
        files = sorted((root / month / "raw/caiso").glob("*.zip"))
        for path in files:
            sidecar = path.with_suffix(path.suffix + ".provenance.json")
            if not sidecar.is_file():
                raise ValueError(f"{month}: missing CAISO sidecar for {path.name}")
            metadata = json.loads(sidecar.read_text())
            if metadata.get("checksum") != sha256_file(path):
                raise ValueError(f"{month}: CAISO checksum mismatch for {path.name}")
            identifiers = raw_identifiers(read_oasis_zip(path))
            allowed = {(ENTITY, DAM_ITEM, "DAM"), (ENTITY, ACTUAL_ITEM, "ACTUAL")}
            if not any((x["TAC_AREA_NAME"], x["XML_DATA_ITEM"], x["MARKET_RUN_ID"]) in allowed
                       for x in identifiers):
                raise ValueError(f"{month}: unexpected CAISO identifiers in {path.name}")
        sidecars = list((root / month / "raw/caiso").glob("*.zip.provenance.json"))
        if len(files) != len(sidecars):
            raise ValueError(f"{month}: raw CAISO cache has incomplete file/sidecar pairs")
        report.update(raw_caiso_acquisitions=len(files), raw_caiso_validated=len(files))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=["2025-10", "2025-11", "2025-12"])
    parser.add_argument("--output-root", type=Path, default=Path("data/backtesting"))
    parser.add_argument("--validate-raw-gdex", action="store_true")
    parser.add_argument("--validate-raw-caiso", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    reports = [certify_month(args.output_root, month, raw_gdex=args.validate_raw_gdex,
                             raw_caiso=args.validate_raw_caiso) for month in args.months]
    payload = {"status": "passed", "months": reports}
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
