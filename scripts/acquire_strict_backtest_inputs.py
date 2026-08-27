#!/usr/bin/env python3
"""Run the local strict CAISO and GDEX acquisition workflow."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", default="2025-12")
    parser.add_argument("--output-root", default="data/backtesting")
    parser.add_argument("--stations", default="data/stations_population_weights.csv")
    parser.add_argument("--caiso-source-document", action="append", required=True)
    parser.add_argument("--gdex-source-document", action="append", required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    caiso = [sys.executable, "scripts/acquire_december_2025_caiso.py", "--month", args.month,
             "--output-root", str(root)]
    for document in args.caiso_source_document:
        caiso.extend(["--source-document", document])
    subprocess.run(caiso, check=True)
    gdex = [sys.executable, "scripts/acquire_gdex_gfs_0p25.py", "--month", args.month,
            "--output-root", str(root), "--stations", args.stations]
    for document in args.gdex_source_document:
        gdex.extend(["--source-document", document])
    subprocess.run(gdex, check=True)
    manifests = [json.loads((root / args.month / name).read_text()) for name in
                 ("acquisition_manifest.caiso.json", "acquisition_manifest.gfs.json")]
    combined = {"schema_version": "strict-acquisition-manifest-v1", "month": args.month,
                "created_at_utc": max(item["created_at_utc"] for item in manifests),
                "source_documents": sum((item["source_documents"] for item in manifests), []),
                "acquisitions": sum((item["acquisitions"] for item in manifests), [])}
    (root / args.month / "acquisition_manifest.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    processed = root / args.month / "processed"
    subprocess.run([sys.executable, "build_strict_backtest_dataset.py", "--month", args.month,
                    "--actuals-csv", str(processed / "actuals.csv"),
                    "--dam-csv", str(processed / "caiso_dam.csv"),
                    "--weather-vintages-csv", str(processed / "weather_vintages.csv"),
                    "--load-history-csv", str(processed / "load_history.csv"),
                    "--output-dir", str(processed / "strict")], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
