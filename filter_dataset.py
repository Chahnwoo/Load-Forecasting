import sys
import pandas as pd


def filter_rows(input_file: str) -> None:
    df = pd.read_csv(input_file)

    required_cols = {"load_mw", "load_previous_week"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    filtered = df.dropna(subset=["load_mw", "load_previous_week"]).copy()
    filtered.to_csv("filtered.csv", index=False)

    print("Saved filtered dataset to filtered.csv")
    print(f"Original rows: {len(df):,}")
    print(f"Filtered rows: {len(filtered):,}")
    print(f"Dropped rows: {len(df) - len(filtered):,}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python filter_dataset.py path/to/dataset.csv")
        sys.exit(1)

    filter_rows(sys.argv[1])