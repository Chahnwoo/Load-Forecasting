import sys
import pandas as pd


def summarize_missing_blocks(series: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """
    Given a boolean Series indexed in row order, where True means missing,
    return contiguous missing blocks as:
        (start_timestamp, end_timestamp, block_length)
    """
    blocks = []
    if series.empty:
        return blocks

    # Identify runs of equal values
    group_id = series.ne(series.shift()).cumsum()

    for _, idx in series.groupby(group_id).groups.items():
        block = series.loc[idx]
        if bool(block.iloc[0]):  # missing block
            blocks.append((block.index[0], block.index[-1], len(block)))

    return blocks


def audit_dataset(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    required_cols = {
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
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    # Parse time columns
    df["timestamp_from_key"] = pd.to_datetime(df["time_key"], format="%Y-%m-%d:%H", errors="raise")
    df["timestamp_utc"] = pd.to_datetime(df["time_utc"], errors="raise")

    # Sort
    df = df.sort_values(["region", "timestamp_utc"]).reset_index(drop=True)

    print("=" * 90)
    print("OVERALL DATASET SUMMARY")
    print("=" * 90)
    print(f"Rows: {len(df):,}")
    print(f"Regions: {sorted(df['region'].unique().tolist())}")
    print(f"Earliest UTC timestamp: {df['timestamp_utc'].min()}")
    print(f"Latest UTC timestamp:   {df['timestamp_utc'].max()}")

    # Check time_key vs time_utc consistency
    mismatch_mask = df["timestamp_from_key"] != df["timestamp_utc"]
    print("\n" + "=" * 90)
    print("TIME KEY / TIME UTC CONSISTENCY")
    print("=" * 90)
    print(f"Rows where parsed time_key != parsed time_utc: {int(mismatch_mask.sum()):,}")
    if mismatch_mask.any():
        print(df.loc[mismatch_mask, ["region", "time_key", "time_utc"]].head(20).to_string(index=False))

    # Per-region audit
    print("\n" + "=" * 90)
    print("PER-REGION AUDIT")
    print("=" * 90)

    region_rows = []

    for region, g in df.groupby("region", sort=True):
        g = g.sort_values("timestamp_utc").copy()

        expected_rows = int(
            ((g["timestamp_utc"].max() - g["timestamp_utc"].min()) / pd.Timedelta(hours=1)) + 1
        )
        actual_rows = len(g)

        num_duplicates = int(g["timestamp_utc"].duplicated().sum())

        diffs = g["timestamp_utc"].diff()
        num_gaps = int((diffs > pd.Timedelta(hours=1)).sum())
        largest_gap_hours = (
            float((diffs[diffs > pd.Timedelta(hours=1)].max() / pd.Timedelta(hours=1)))
            if num_gaps > 0 else 0.0
        )

        load_nonnull = g["load_mw"].notna()
        n_load_missing = int((~load_nonnull).sum())
        n_load_present = int(load_nonnull.sum())

        if n_load_present > 0:
            first_nonnull = g.loc[load_nonnull, "timestamp_utc"].min()
            last_nonnull = g.loc[load_nonnull, "timestamp_utc"].max()
        else:
            first_nonnull = pd.NaT
            last_nonnull = pd.NaT

        # Build missing blocks indexed by timestamp
        missing_series = pd.Series(
            (~load_nonnull).to_numpy(),
            index=g["timestamp_utc"].to_numpy()
        )
        missing_blocks = summarize_missing_blocks(missing_series)

        region_rows.append({
            "region": region,
            "actual_rows": actual_rows,
            "expected_hourly_rows": expected_rows,
            "duplicate_timestamps": num_duplicates,
            "num_gaps_gt_1h": num_gaps,
            "largest_gap_hours": largest_gap_hours,
            "load_present_rows": n_load_present,
            "load_missing_rows": n_load_missing,
            "first_load_present_utc": first_nonnull,
            "last_load_present_utc": last_nonnull,
        })

        print(f"\nRegion: {region}")
        print(f"  Rows: {actual_rows:,} (expected hourly rows in span: {expected_rows:,})")
        print(f"  Duplicate timestamps: {num_duplicates:,}")
        print(f"  Gaps > 1 hour: {num_gaps:,}")
        print(f"  Largest gap (hours): {largest_gap_hours}")
        print(f"  load_mw present: {n_load_present:,}")
        print(f"  load_mw missing: {n_load_missing:,}")
        print(f"  First non-null load_mw: {first_nonnull}")
        print(f"  Last non-null load_mw:  {last_nonnull}")

        if missing_blocks:
            print("  Missing load_mw blocks:")
            for start, end, length in missing_blocks[:10]:
                print(f"    {start}  ->  {end}   ({length:,} hours)")
            if len(missing_blocks) > 10:
                print(f"    ... and {len(missing_blocks) - 10} more blocks")
        else:
            print("  Missing load_mw blocks: none")

    region_summary = pd.DataFrame(region_rows)

    print("\n" + "=" * 90)
    print("REGION SUMMARY TABLE")
    print("=" * 90)
    print(region_summary.to_string(index=False))

    # Weather missingness
    weather_cols = [
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "precipitation",
        "cloud_cover",
        "wind_speed_10m",
        "shortwave_radiation",
        "cdd_65f",
        "hdd_65f",
    ]

    print("\n" + "=" * 90)
    print("COLUMN MISSINGNESS SUMMARY")
    print("=" * 90)
    for col in ["load_mw"] + weather_cols:
        n_missing = int(df[col].isna().sum())
        print(f"{col:24s} missing = {n_missing:,}")

    # Degree-day sanity check
    print("\n" + "=" * 90)
    print("DEGREE-DAY SANITY CHECK")
    print("=" * 90)

    cdd_expected = (df["temperature_2m"].astype(float) - 65.0).clip(lower=0.0)
    hdd_expected = (65.0 - df["temperature_2m"].astype(float)).clip(lower=0.0)

    cdd_bad = ~((df["cdd_65f"].astype(float) - cdd_expected).abs() < 1e-9)
    hdd_bad = ~((df["hdd_65f"].astype(float) - hdd_expected).abs() < 1e-9)

    print(f"Rows where cdd_65f does not match temperature_2m: {int(cdd_bad.sum()):,}")
    print(f"Rows where hdd_65f does not match temperature_2m: {int(hdd_bad.sum()):,}")

    # Focused load cutoff analysis
    print("\n" + "=" * 90)
    print("FOCUSED LOAD CUTOFF ANALYSIS")
    print("=" * 90)

    for region, g in df.groupby("region", sort=True):
        g = g.sort_values("timestamp_utc").copy()
        nonnull = g["load_mw"].notna()

        if nonnull.any():
            last_good_idx = g.loc[nonnull, "timestamp_utc"].max()
            after_last_good = g.loc[g["timestamp_utc"] > last_good_idx].copy()
            print(f"\nRegion: {region}")
            print(f"  Last non-null load_mw timestamp: {last_good_idx}")
            print(f"  Rows after last non-null timestamp: {len(after_last_good):,}")
            if len(after_last_good) > 0:
                print(f"  All rows after that missing load_mw? {bool(after_last_good['load_mw'].isna().all())}")
                print("  First 5 rows after cutoff:")
                print(
                    after_last_good[["region", "time_key", "time_utc", "load_mw"]]
                    .head(5)
                    .to_string(index=False)
                )
        else:
            print(f"\nRegion: {region}")
            print("  No non-null load_mw rows at all.")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python audit_caiso_dataset.py path/to/caiso_dataset.csv")
        sys.exit(1)

    audit_dataset(sys.argv[1])