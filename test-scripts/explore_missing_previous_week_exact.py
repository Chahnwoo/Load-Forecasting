import sys
import pandas as pd


def explore_missing_previous_week_exact(csv_path: str) -> None:
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

    # -----------------------------
    # Drop problematic region
    # -----------------------------
    df = df[df["region"] != "mwd"].copy()

    # -----------------------------
    # Parse ONE canonical timestamp source
    # -----------------------------
    df["timestamp"] = pd.to_datetime(df["time_key"], format="%Y-%m-%d:%H", errors="raise")
    df["previous_week_timestamp"] = df["timestamp"] - pd.Timedelta(days=7)

    # -----------------------------
    # Build exact key set for lookup
    # -----------------------------
    # Using tuples avoids merge-related surprises and ensures strict equality
    exact_keys = set(zip(df["region"], df["timestamp"]))

    # Exact existence check
    df["previous_week_exists"] = [
        (region, prev_ts) in exact_keys
        for region, prev_ts in zip(df["region"], df["previous_week_timestamp"])
    ]

    missing = df.loc[~df["previous_week_exists"]].copy()

    # -----------------------------
    # Region bounds for classification
    # -----------------------------
    region_bounds = (
        df.groupby("region")
        .agg(region_start=("timestamp", "min"), region_end=("timestamp", "max"))
        .reset_index()
    )
    missing = missing.merge(region_bounds, on="region", how="left")

    def classify_row(row):
        if row["previous_week_timestamp"] < row["region_start"]:
            return "before_region_start"
        if row["previous_week_timestamp"] > row["region_end"]:
            return "after_region_end"
        return "unexpected_no_match"

    missing["missing_reason"] = missing.apply(classify_row, axis=1)

    # -----------------------------
    # Derived columns
    # -----------------------------
    missing["year_month"] = missing["timestamp"].dt.to_period("M").astype(str)
    missing["weekday"] = missing["timestamp"].dt.day_name()
    missing["hour"] = missing["timestamp"].dt.hour

    # -----------------------------
    # Overall summary
    # -----------------------------
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Rows after dropping mwd: {len(df)}")
    print(f"Rows with exact previous-week row present: {int(df['previous_week_exists'].sum())}")
    print(f"Rows without exact previous-week row: {len(missing)}")
    print(f"Percent without exact previous-week row: {100 * len(missing) / len(df):.2f}%")

    # -----------------------------
    # Missingness by region
    # -----------------------------
    region_summary = (
        df.groupby("region")
        .agg(
            total_rows=("region", "size"),
            missing_previous_week=("previous_week_exists", lambda s: (~s).sum()),
        )
        .reset_index()
    )
    region_summary["pct_missing"] = (
        100 * region_summary["missing_previous_week"] / region_summary["total_rows"]
    )

    print("\n" + "=" * 80)
    print("MISSINGNESS BY REGION")
    print("=" * 80)
    print(region_summary.to_string(index=False))

    # -----------------------------
    # Missing reasons
    # -----------------------------
    print("\n" + "=" * 80)
    print("MISSING REASONS")
    print("=" * 80)
    print(missing["missing_reason"].value_counts().to_string())

    # -----------------------------
    # Missing rows by month
    # -----------------------------
    print("\n" + "=" * 80)
    print("MISSING ROWS BY MONTH")
    print("=" * 80)
    print(missing["year_month"].value_counts().sort_index().to_string())

    # -----------------------------
    # Missing rows by weekday
    # -----------------------------
    print("\n" + "=" * 80)
    print("MISSING ROWS BY WEEKDAY")
    print("=" * 80)
    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]
    weekday_counts = missing["weekday"].value_counts().reindex(weekday_order, fill_value=0)
    print(weekday_counts.to_string())

    # -----------------------------
    # Missing rows by hour
    # -----------------------------
    print("\n" + "=" * 80)
    print("MISSING ROWS BY HOUR")
    print("=" * 80)
    print(missing["hour"].value_counts().sort_index().to_string())

    # -----------------------------
    # Weekend / holiday flags among missing rows
    # -----------------------------
    print("\n" + "=" * 80)
    print("MISSING ROWS BY WEEKEND / HOLIDAY FLAGS")
    print("=" * 80)
    for col in ["is_weekend", "US_federal_holidays", "state_holidays"]:
        print(f"\n{col}:")
        print(missing[col].value_counts(dropna=False).to_string())

    # -----------------------------
    # Sample missing rows
    # -----------------------------
    print("\n" + "=" * 80)
    print("SAMPLE MISSING ROWS")
    print("=" * 80)
    sample_cols = [
        "region",
        "time_key",
        "timestamp",
        "previous_week_timestamp",
        "load_mw",
        "load_previous_week",
        "missing_reason",
    ]
    print(
        missing.sort_values(["region", "timestamp"])[sample_cols]
        .head(50)
        .to_string(index=False)
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python explore_missing_previous_week_exact.py path/to/dataset.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    explore_missing_previous_week_exact(csv_path)