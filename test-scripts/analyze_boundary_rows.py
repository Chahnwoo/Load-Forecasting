import sys
import pandas as pd


def analyze_boundary_rows(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    # Drop MWD
    df = df[df["region"] != "mwd"].copy()

    # Parse times
    df["timestamp"] = pd.to_datetime(df["time_key"], format="%Y-%m-%d:%H", errors="raise")

    # Parse time_utc if available
    if "time_utc" in df.columns:
        df["time_utc_parsed"] = pd.to_datetime(df["time_utc"], errors="coerce")
    else:
        df["time_utc_parsed"] = pd.NaT

    # Previous-week target
    df["previous_week_timestamp"] = df["timestamp"] - pd.Timedelta(days=7)

    # Region bounds
    region_bounds = (
        df.groupby("region")
        .agg(region_start=("timestamp", "min"), region_end=("timestamp", "max"))
        .reset_index()
    )
    df = df.merge(region_bounds, on="region", how="left")

    # Lookup for exact within-region previous-week match
    lookup = df[["region", "timestamp"]].drop_duplicates().rename(
        columns={"timestamp": "lookup_timestamp"}
    )

    merged = df.merge(
        lookup,
        how="left",
        left_on=["region", "previous_week_timestamp"],
        right_on=["region", "lookup_timestamp"],
    )

    merged["previous_week_exists"] = merged["lookup_timestamp"].notna()

    # Rows with no previous-week match
    missing = merged.loc[~merged["previous_week_exists"]].copy()

    # Global timestamp existence check (ignoring region)
    all_times = set(df["timestamp"].dropna().unique())
    missing["prev_week_exists_any_region"] = missing["previous_week_timestamp"].isin(all_times)

    # Classification
    def classify_row(row):
        if row["previous_week_timestamp"] < row["region_start"]:
            return "before_region_start"
        if row["previous_week_timestamp"] > row["region_end"]:
            return "after_region_end"
        if row["prev_week_exists_any_region"]:
            return "prev_week_time_exists_in_other_regions_only"
        # DST suspicion: March / November are classic local-time trouble months
        if row["timestamp"].month in (3, 11):
            return "dst_or_calendar_alignment_issue"
        return "unknown_no_match_within_region"

    missing["missing_reason"] = missing.apply(classify_row, axis=1)

    # Helpful derived fields
    missing["year_month"] = missing["timestamp"].dt.to_period("M").astype(str)
    missing["weekday"] = missing["timestamp"].dt.day_name()
    missing["hour"] = missing["timestamp"].dt.hour
    missing["days_from_region_start"] = (
        (missing["timestamp"] - missing["region_start"]) / pd.Timedelta(days=1)
    )

    if missing["time_utc_parsed"].notna().any():
        missing["utc_offset_hours"] = (
            (missing["timestamp"] - missing["time_utc_parsed"].dt.tz_localize(None))
            / pd.Timedelta(hours=1)
        )

    print("=" * 90)
    print("OVERALL")
    print("=" * 90)
    print(f"Rows after dropping mwd: {len(df)}")
    print(f"Rows missing exact previous-week match: {len(missing)}")

    print("\n" + "=" * 90)
    print("MISSING REASONS")
    print("=" * 90)
    reason_counts = missing["missing_reason"].value_counts()
    print(reason_counts.to_string())

    print("\n" + "=" * 90)
    print("MISSING REASONS BY REGION")
    print("=" * 90)
    reason_region = (
        missing.groupby(["region", "missing_reason"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    print(reason_region.to_string())

    print("\n" + "=" * 90)
    print("MISSING REASONS BY MONTH")
    print("=" * 90)
    reason_month = (
        missing.groupby(["year_month", "missing_reason"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    print(reason_month.to_string())

    print("\n" + "=" * 90)
    print("COUNTS BY HOUR")
    print("=" * 90)
    hour_counts = missing.groupby(["hour", "missing_reason"]).size().unstack(fill_value=0)
    print(hour_counts.to_string())

    print("\n" + "=" * 90)
    print("COUNTS BY WEEKDAY")
    print("=" * 90)
    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]
    weekday_counts = (
        missing.groupby(["weekday", "missing_reason"])
        .size()
        .unstack(fill_value=0)
        .reindex(weekday_order, fill_value=0)
    )
    print(weekday_counts.to_string())

    if "utc_offset_hours" in missing.columns:
        print("\n" + "=" * 90)
        print("UTC OFFSET DISTRIBUTION FOR MISSING ROWS")
        print("=" * 90)
        print(missing["utc_offset_hours"].value_counts(dropna=False).sort_index().to_string())

    # Samples by reason
    sample_cols = [
        "region",
        "time_key",
        "time_utc",
        "timestamp",
        "previous_week_timestamp",
        "region_start",
        "region_end",
        "year_month",
        "weekday",
        "hour",
        "load_previous_week",
        "missing_reason",
    ]
    if "utc_offset_hours" in missing.columns:
        sample_cols.append("utc_offset_hours")

    print("\n" + "=" * 90)
    print("SAMPLE ROWS BY REASON")
    print("=" * 90)
    for reason in missing["missing_reason"].value_counts().index:
        print(f"\n--- {reason} ---")
        sample = missing.loc[missing["missing_reason"] == reason, sample_cols].head(20)
        print(sample.to_string(index=False))

    # Zoom in on recent months
    recent = missing.loc[missing["timestamp"] >= pd.Timestamp("2026-02-01")].copy()
    print("\n" + "=" * 90)
    print("RECENT MISSING ROWS (2026-02-01 AND LATER)")
    print("=" * 90)
    print(f"Count: {len(recent)}")
    if len(recent) > 0:
        recent_summary = (
            recent.groupby(["year_month", "missing_reason"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        print(recent_summary.to_string())

        print("\nRecent sample:")
        print(recent[sample_cols].head(40).to_string(index=False))

    # Optional: save missing rows for inspection
    missing.to_csv("boundary_rows_analysis.csv", index=False)
    print("\nSaved detailed missing-row analysis to: boundary_rows_analysis.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_boundary_rows.py path/to/dropped.csv")
        sys.exit(1)

    analyze_boundary_rows(sys.argv[1])