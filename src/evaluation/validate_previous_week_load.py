import pandas as pd
import sys


def validate_load_previous_week(csv_path: str) -> None:
    # Read CSV
    df = pd.read_csv(csv_path)

    # Required columns
    required_cols = {"time_key", "region", "load_mw", "load_previous_week"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Parse time_key of the form YYYY-MM-DD:HH
    df["timestamp"] = pd.to_datetime(df["time_key"], format="%Y-%m-%d:%H", errors="raise")

    # Compute the timestamp exactly one week earlier
    df["previous_week_timestamp"] = df["timestamp"] - pd.Timedelta(weeks=1)

    # Build lookup table from (region, timestamp) -> load_mw
    lookup = df[["region", "timestamp", "load_mw"]].rename(
        columns={"timestamp": "lookup_timestamp", "load_mw": "expected_load_previous_week"}
    )

    # Merge each row with its previous-week row in the same region
    merged = df.merge(
        lookup,
        how="left",
        left_on=["region", "previous_week_timestamp"],
        right_on=["region", "lookup_timestamp"],
    )

    # Rows where previous-week data exists
    has_previous_week_data = merged["expected_load_previous_week"].notna()

    # Among those, check correctness
    is_correct = has_previous_week_data & (
        merged["load_previous_week"] == merged["expected_load_previous_week"]
    )

    num_correct = int(is_correct.sum())
    num_with_previous_week_data = int(has_previous_week_data.sum())
    num_incorrect = int((has_previous_week_data & ~is_correct).sum())

    print(f"Total rows in dataset: {len(merged)}")
    print(f"Rows with previous-week data available: {num_with_previous_week_data}")
    print(f'Rows where "load_previous_week" is correct: {num_correct}')
    print(f'Rows where "load_previous_week" is incorrect: {num_incorrect}')

    # Optional: print incorrect rows for inspection
    incorrect_rows = merged.loc[has_previous_week_data & ~is_correct, [
        "time_key",
        "region",
        "load_mw",
        "load_previous_week",
        "expected_load_previous_week",
    ]]

    if not incorrect_rows.empty:
        print("\nSample incorrect rows:")
        print(incorrect_rows.head(20).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_load_previous_week.py path/to/your_file.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    validate_load_previous_week(csv_file)