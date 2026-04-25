#!/usr/bin/env python3
"""
add_calendar_features.py

Adds calendar and lag features to the CAISO dataset.

New columns:
    is_weekend
    US_federal_holidays
    state_holidays
    load_previous_week

Output:
    data/processed/revised_caiso_dataset_20200101_to_20260421.csv
"""

from pathlib import Path
import pandas as pd
from pandas.tseries.holiday import (
    USFederalHolidayCalendar,
    AbstractHolidayCalendar,
    Holiday,
    nearest_workday,
    next_monday
)
from pandas.tseries.offsets import DateOffset
from pandas.tseries.holiday import MO, TH

INPUT_FILE = "data/processed/caiso_dataset_20200101_to_20260421.csv"
OUTPUT_FILE = "data/processed/revised_caiso_dataset_20200101_to_20260421.csv"


# -----------------------------
# California State Holidays
# -----------------------------
class CaliforniaHolidayCalendar(AbstractHolidayCalendar):
    rules = [

        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),

        Holiday(
            "MartinLutherKingJrDay",
            month=1,
            day=1,
            offset=DateOffset(weekday=MO(3))
        ),

        Holiday(
            "PresidentsDay",
            month=2,
            day=1,
            offset=DateOffset(weekday=MO(3))
        ),

        Holiday("CesarChavezDay", month=3, day=31, observance=nearest_workday),

        Holiday(
            "MemorialDay",
            month=5,
            day=31,
            offset=DateOffset(weekday=MO(-1))
        ),

        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),

        Holiday(
            "LaborDay",
            month=9,
            day=1,
            offset=DateOffset(weekday=MO(1))
        ),

        Holiday("VeteransDay", month=11, day=11, observance=nearest_workday),

        Holiday(
            "Thanksgiving",
            month=11,
            day=1,
            offset=DateOffset(weekday=TH(4))
        ),

        Holiday(
            "DayAfterThanksgiving",
            month=11,
            day=1,
            offset=[DateOffset(weekday=TH(4)), DateOffset(days=1)]
        ),

        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


# -----------------------------
# Main Feature Builder
# -----------------------------
def main():

    df = pd.read_csv(INPUT_FILE)

    df["time_utc"] = pd.to_datetime(df["time_utc"])

    df = df.sort_values(["region", "time_utc"]).reset_index(drop=True)

    # -----------------------------
    # Weekend feature
    # -----------------------------
    df["is_weekend"] = (df["time_utc"].dt.dayofweek >= 5).astype(int)

    # -----------------------------
    # Federal holidays
    # -----------------------------
    fed_cal = USFederalHolidayCalendar()

    fed_holidays = fed_cal.holidays(
        start=df["time_utc"].min(),
        end=df["time_utc"].max()
    )

    df["US_federal_holidays"] = (
        df["time_utc"].dt.normalize().isin(fed_holidays)
    ).astype(int)

    # -----------------------------
    # California state holidays
    # -----------------------------
    ca_cal = CaliforniaHolidayCalendar()

    state_holidays = ca_cal.holidays(
        start=df["time_utc"].min(),
        end=df["time_utc"].max()
    )

    df["state_holidays"] = (
        df["time_utc"].dt.normalize().isin(state_holidays)
    ).astype(int)

    # -----------------------------
    # Previous week load
    # -----------------------------
    lookup = df[["region", "time_utc", "load_mw"]].copy()

    lookup["time_utc"] = lookup["time_utc"] + pd.Timedelta(days=7)

    lookup = lookup.rename(
        columns={"load_mw": "load_previous_week"}
    )

    df = df.merge(
        lookup,
        on=["region", "time_utc"],
        how="left"
    )

    # -----------------------------
    # Save dataset
    # -----------------------------
    Path("data").mkdir(exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
