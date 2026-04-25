import sys
from pathlib import Path
import pandas as pd


def filter_rows(input_file: str) -> None:
    df = pd.read_csv(input_file)

    required_cols = {"load_mw", "load_previous_week"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    filtered = df.dropna(subset=["load_mw", "load_previous_week"]).copy()
    output_file = Path("data/processed/filtered.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_file, index=False)

    print(f"Saved filtered dataset to {output_file}")
    print(f"Original rows: {len(df):,}")
    print(f"Filtered rows: {len(filtered):,}")
    print(f"Dropped rows: {len(df) - len(filtered):,}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/preprocessing/filter_dataset.py path/to/dataset.csv")
        sys.exit(1)

    filter_rows(sys.argv[1])
