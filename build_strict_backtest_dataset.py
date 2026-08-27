#!/usr/bin/env python3
"""Build strict products from provenance-bearing source extracts (no metrics)."""

import argparse
import sys

import pandas as pd

from src.backtesting.strict_dataset import StrictBacktestBuilder, StrictDataError, write_products


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True, help="CAISO operating month, YYYY-MM")
    parser.add_argument("--entity", default="caiso", choices=["caiso"])
    parser.add_argument("--forecast-cutoff-local", default="07:00", metavar="HH:MM")
    parser.add_argument("--actuals-csv", required=True)
    parser.add_argument("--dam-csv", required=True)
    parser.add_argument("--weather-vintages-csv",
                        help="NCEI DSI 6182 GFS Grid 004 (0.5 degree) full-cycle forecasts with provenance")
    parser.add_argument("--load-history-csv", required=True)
    parser.add_argument("--output-dir", default="data/strict_backtest")
    args = parser.parse_args()
    if not args.weather_vintages_csv:
        parser.error("strict mode requires --weather-vintages-csv; realized Archive API weather is forbidden")
    try:
        products = StrictBacktestBuilder(args.month, args.forecast_cutoff_local).build(
            pd.read_csv(args.actuals_csv), pd.read_csv(args.dam_csv),
            pd.read_csv(args.weather_vintages_csv), pd.read_csv(args.load_history_csv))
        write_products(products, args.output_dir)
    except StrictDataError as exc:
        print(f"strict dataset blocked: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
