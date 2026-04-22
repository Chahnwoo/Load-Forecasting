import sys
import pandas as pd


def explore_missing_previous_week(csv_path: str) -> None:
    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(csv_path)

    required_cols = {
        "region",
        "time_key",
        "load_mw",
        "load_previous_week",
        "is_weekend",
        "US_federal_holidays",
        "state_holidays",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    # Parse timestamp from time_key format YYYY-MM-DD:HH
    df["timestamp"] = pd.to_datetime(df["time_key"], format="%Y-%m-%d:%H", errors="raise")
    df["previous_week_timestamp"] = df["timestamp"] - pd.Timedelta(weeks=1)

    # -----------------------------
    # Find rows whose previous-week row exists or not
    # -----------------------------
    lookup = df[["region", "timestamp", "load_mw"]].rename(
        columns={"timestamp": "lookup_timestamp", "load_mw": "prev_week_load_mw"}
    )

    merged = df.merge(
        lookup,
        how="left",
        left_on=["region", "previous_week_timestamp"],
        right_on=["region", "lookup_timestamp"],
    )

    merged["previous_week_row_exists"] = merged["prev_week_load_mw"].notna()
    missing_prev = merged.loc[~merged["previous_week_row_exists"]].copy()

    # -----------------------------
    # Basic summary
    # -----------------------------
    total_rows = len(merged)
    total_missing = len(missing_prev)
    total_found = int(merged["previous_week_row_exists"].sum())

    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total rows: {total_rows}")
    print(f"Rows with previous-week row present: {total_found}")
    print(f"Rows without previous-week row: {total_missing}")
    print(f"Percent without previous-week row: {100 * total_missing / total_rows:.2f}%")

    # -----------------------------
    # Region-level missingness
    # -----------------------------
    region_summary = (
        merged.groupby("region")
        .agg(
            total_rows=("region", "size"),
            missing_previous_week=("previous_week_row_exists", lambda s: (~s).sum()),
        )
        .reset_index()
    )
    region_summary["pct_missing"] = (
        100 * region_summary["missing_previous_week"] / region_summary["total_rows"]
    )
    region_summary = region_summary.sort_values(
        ["missing_previous_week", "pct_missing"], ascending=[False, False]
    )

    print("\n" + "=" * 80)
    print("MISSINGNESS BY REGION")
    print("=" * 80)
    print(region_summary.to_string(index=False))

    # -----------------------------
    # Dataset date range overall
    # -----------------------------
    print("\n" + "=" * 80)
    print("OVERALL TIME RANGE")
    print("=" * 80)
    print(f"Earliest timestamp in dataset: {merged['timestamp'].min()}")
    print(f"Latest timestamp in dataset:   {merged['timestamp'].max()}")

    # -----------------------------
    # Earliest / latest timestamp by region
    # -----------------------------
    region_time_range = (
        merged.groupby("region")
        .agg(
            earliest_timestamp=("timestamp", "min"),
            latest_timestamp=("timestamp", "max"),
            total_rows=("timestamp", "size"),
        )
        .reset_index()
        .sort_values("earliest_timestamp")
    )

    print("\n" + "=" * 80)
    print("TIME RANGE BY REGION")
    print("=" * 80)
    print(region_time_range.to_string(index=False))

    # -----------------------------
    # Are missing rows mostly at the start of each region?
    # -----------------------------
    first_available_per_region = merged.groupby("region")["timestamp"].min().rename("region_start")
    missing_prev = missing_prev.merge(first_available_per_region, on="region", how="left")
    missing_prev["hours_since_region_start"] = (
        (missing_prev["timestamp"] - missing_prev["region_start"]) / pd.Timedelta(hours=1)
    )

    print("\n" + "=" * 80)
    print("HOW FAR FROM REGION START ARE THE MISSING ROWS?")
    print("=" * 80)
    print("This helps determine whether the missing rows are mostly just the first 7 days.")
    print(
        missing_prev["hours_since_region_start"]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        .to_string()
    )

    near_start_counts = {
        "within_first_24h": int((missing_prev["hours_since_region_start"] < 24).sum()),
        "within_first_7d": int((missing_prev["hours_since_region_start"] < 24 * 7).sum()),
        "within_first_8d": int((missing_prev["hours_since_region_start"] < 24 * 8).sum()),
        "after_first_7d": int((missing_prev["hours_since_region_start"] >= 24 * 7).sum()),
    }

    print("\nCounts of missing rows by distance from region start:")
    for k, v in near_start_counts.items():
        print(f"{k}: {v}")

    # -----------------------------
    # Time-based patterns among missing rows
    # -----------------------------
    missing_prev["year_month"] = missing_prev["timestamp"].dt.to_period("M").astype(str)
    missing_prev["weekday"] = missing_prev["timestamp"].dt.day_name()
    missing_prev["hour"] = missing_prev["timestamp"].dt.hour

    print("\n" + "=" * 80)
    print("MISSING ROWS BY MONTH")
    print("=" * 80)
    month_counts = missing_prev["year_month"].value_counts().sort_index()
    print(month_counts.to_string())

    print("\n" + "=" * 80)
    print("MISSING ROWS BY WEEKDAY")
    print("=" * 80)
    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]
    weekday_counts = missing_prev["weekday"].value_counts().reindex(weekday_order, fill_value=0)
    print(weekday_counts.to_string())

    print("\n" + "=" * 80)
    print("MISSING ROWS BY HOUR")
    print("=" * 80)
    hour_counts = missing_prev["hour"].value_counts().sort_index()
    print(hour_counts.to_string())

    # -----------------------------
    # Weekend / holiday patterns
    # -----------------------------
    print("\n" + "=" * 80)
    print("MISSING ROWS BY WEEKEND / HOLIDAY FLAGS")
    print("=" * 80)

    for col in ["is_weekend", "US_federal_holidays", "state_holidays"]:
        print(f"\n{col}:")
        print(missing_prev[col].value_counts(dropna=False).to_string())

    # Compare missing rate by flag against full dataset
    print("\n" + "=" * 80)
    print("MISSING RATE BY WEEKEND / HOLIDAY FLAGS")
    print("=" * 80)

    for col in ["is_weekend", "US_federal_holidays", "state_holidays"]:
        summary = (
            merged.groupby(col)
            .agg(
                total_rows=("region", "size"),
                missing_rows=("previous_week_row_exists", lambda s: (~s).sum()),
            )
            .reset_index()
        )
        summary["pct_missing"] = 100 * summary["missing_rows"] / summary["total_rows"]
        print(f"\n{col}:")
        print(summary.to_string(index=False))

    # -----------------------------
    # Check for within-region hourly gaps
    # -----------------------------
    print("\n" + "=" * 80)
    print("HOURLY GAP CHECK BY REGION")
    print("=" * 80)
    print("If a region has missing hours, that can also create missing previous-week matches.")

    gap_summaries = []
    for region, g in merged.groupby("region"):
        g = g.sort_values("timestamp").copy()
        diffs = g["timestamp"].diff()
        gaps = diffs[diffs > pd.Timedelta(hours=1)]

        gap_summaries.append({
            "region": region,
            "num_gaps_gt_1h": int(gaps.shape[0]),
            "largest_gap_hours": (
                gaps.max() / pd.Timedelta(hours=1) if not gaps.empty else 0
            ),
            "expected_hourly_rows_if_contiguous": int(
                ((g["timestamp"].max() - g["timestamp"].min()) / pd.Timedelta(hours=1)) + 1
            ),
            "actual_rows": int(len(g)),
        })

    gap_summary_df = pd.DataFrame(gap_summaries).sort_values(
        ["num_gaps_gt_1h", "largest_gap_hours"], ascending=[False, False]
    )
    gap_summary_df["missing_hours_within_span"] = (
        gap_summary_df["expected_hourly_rows_if_contiguous"] - gap_summary_df["actual_rows"]
    )

    print(gap_summary_df.to_string(index=False))

    # -----------------------------
    # Show sample missing rows
    # -----------------------------
    print("\n" + "=" * 80)
    print("SAMPLE ROWS WITHOUT PREVIOUS-WEEK DATA")
    print("=" * 80)
    sample_cols = [
        "region", "time_key", "timestamp", "previous_week_timestamp",
        "load_mw", "load_previous_week",
        "is_weekend", "US_federal_holidays", "state_holidays",
        "hours_since_region_start",
    ]
    print(
        missing_prev.sort_values(["region", "timestamp"])[sample_cols]
        .head(50)
        .to_string(index=False)
    )

    # -----------------------------
    # Optional diagnostic conclusion
    # -----------------------------
    print("\n" + "=" * 80)
    print("QUICK INTERPRETATION HELP")
    print("=" * 80)

    within_first_week = near_start_counts["within_first_7d"]
    if total_missing > 0:
        pct_first_week = 100 * within_first_week / total_missing
        print(
            f"{within_first_week} out of {total_missing} missing rows "
            f"({pct_first_week:.2f}%) occur within the first 7 days of each region's timeline."
        )

        if pct_first_week > 95:
            print(
                "This strongly suggests the missing previous-week rows are mostly expected: "
                "the dataset simply does not go back far enough for the earliest week."
            )
        else:
            print(
                "A nontrivial share of missing rows occurs after the first 7 days, "
                "so there may be gaps or missing hours within one or more regions."
            )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python explore_missing_previous_week.py path/to/dataset.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    explore_missing_previous_week(csv_path)