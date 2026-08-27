#!/usr/bin/env python3
"""Download and extract NCEI DSI 6182 GFS 0.5-degree Grid 004 forecasts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.backtesting.acquisition import (NCEI_GRID004_ROOT, NCEI_GRID004_SOURCE,
                                         grid004_object_url, hourly_from_grid004,
                                         parse_grid004_name, sha256_file, write_manifest)
from src.backtesting.strict_dataset import operating_intervals

FIELDS = {
    "temperature_2m": r":TMP:2 m above ground:",
    "relative_humidity_2m": r":RH:2 m above ground:",
    "cloud_cover": r":TCDC:entire atmosphere:",
    "u10": r":UGRD:10 m above ground:",
    "v10": r":VGRD:10 m above ground:",
    "precipitation": r":APCP:surface:",
    "shortwave_radiation": r":DSWRF:surface:",
}


def required_objects(month: str) -> list[str]:
    """Use the newest +10h-available cycle and 3h brackets for all targets."""
    origins = operating_intervals(month)
    names = set()
    for _, day in origins.groupby("operating_day"):
        cutoff = day.forecast_cutoff_utc.iloc[0]
        cycle = cutoff.floor("6h")
        while cycle + pd.Timedelta(hours=10) > cutoff:
            cycle -= pd.Timedelta(hours=6)
        for valid in day.time_utc:
            lead = int((valid - cycle).total_seconds() / 3600)
            for bracket in {lead - lead % 3, lead + (-lead) % 3}:
                if not 0 <= bracket <= 192:
                    raise RuntimeError(f"Grid 004 cannot cover {valid} from eligible cycle {cycle}")
                names.add(f"gfs_4_{cycle:%Y%m%d}_{cycle:%H}00_{bracket:03d}.grb2")
    return sorted(names)


def download(session: requests.Session, url: str, path: Path, retries: int = 4) -> str:
    sidecar = path.with_suffix(path.suffix + ".provenance.json")
    if path.exists() or sidecar.exists():
        if not (path.exists() and sidecar.exists()):
            raise RuntimeError(f"incomplete prior download state for {path}")
        metadata = json.loads(sidecar.read_text())
        if metadata.get("source_object") != url or metadata.get("checksum") != sha256_file(path):
            raise RuntimeError(f"existing Grid 004 provenance mismatch: {path}")
        return metadata["retrieved_at_utc"]
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=(20, 600), stream=True)
            response.raise_for_status()
            temporary = path.with_suffix(path.suffix + ".part")
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        stream.write(chunk)
            temporary.replace(path)
            retrieved = datetime.now(timezone.utc).isoformat()
            sidecar.write_text(json.dumps({"source_object": url, "retrieved_at_utc": retrieved,
                                           "checksum": sha256_file(path)}, indent=2) + "\n")
            return retrieved
        except requests.RequestException:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def inventory(path: Path) -> list[str]:
    result = subprocess.run(["wgrib2", str(path), "-s"], check=True, capture_output=True, text=True)
    return result.stdout.splitlines()


def period_hours(metadata: str, statistic: str) -> int:
    match = re.search(r":(\d+)-(\d+) hour " + statistic + r" fcst", metadata)
    if not match:
        raise RuntimeError(f"field lacks explicit interval {statistic} semantics: {metadata}")
    return int(match.group(2)) - int(match.group(1))


def point_value(path: Path, pattern: str, longitude: float, latitude: float) -> tuple[float, str]:
    lines = [line for line in inventory(path) if re.search(pattern, line)]
    if len(lines) != 1:
        raise RuntimeError(f"expected one GRIB field matching {pattern!r} in {path}, found {len(lines)}")
    field_no = lines[0].split(":", 1)[0]
    result = subprocess.run(["wgrib2", str(path), "-d", field_no, "-lon", str(longitude), str(latitude)],
                            check=True, capture_output=True, text=True)
    match = re.search(r"val=([-+0-9.eE]+)", result.stdout)
    if not match:
        raise RuntimeError(f"could not extract station point from {path}: {result.stdout}")
    return float(match.group(1)), lines[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", default="2025-12")
    parser.add_argument("--output-root", default="data/backtesting")
    parser.add_argument("--stations", default="data/stations_population_weights.csv")
    parser.add_argument("--source-document", action="append", required=True,
                        help="Immutable local copy or URL for NCEI DSI 6182 documentation (repeatable)")
    args = parser.parse_args()
    month_dir = Path(args.output_root) / args.month
    raw_dir, processed = month_dir / "raw/gfs", month_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True); processed.mkdir(parents=True, exist_ok=True)
    stations = pd.read_csv(args.stations).query("region == 'caiso'")
    records, extracted = [], []
    with requests.Session() as session:
        for name in required_objects(args.month):
            parsed, url, target = parse_grid004_name(name), grid004_object_url(name), raw_dir / name
            retrieved = download(session, url, target)
            checksum = sha256_file(target)
            for station in stations.itertuples():
                values, metadata = {}, {}
                for field, pattern in FIELDS.items():
                    values[field], metadata[field] = point_value(target, pattern, station.longitude, station.latitude)
                values["temperature_2m"] -= 273.15  # GRIB TMP is kelvin; model contract is Celsius.
                values["wind_speed_10m"] = (values.pop("u10") ** 2 + values.pop("v10") ** 2) ** 0.5
                extracted.append({"station": station.station_name, "latitude": station.latitude,
                                  "longitude": station.longitude, "population_weight": station.population_weight,
                                  **parsed, **values, "precipitation_period_hours": period_hours(metadata["precipitation"], "acc"),
                                  "shortwave_period_hours": period_hours(metadata["shortwave_radiation"], "ave"),
                                  "source": NCEI_GRID004_SOURCE, "model": "gfs-grid004-0p5",
                                  "source_object": url, "checksum": checksum,
                                  "available_at_utc": parsed["model_init_time_utc"] + pd.Timedelta(hours=10),
                                  "availability_policy": "gfs_init_plus_10h_v1"})
            records.append({"source": NCEI_GRID004_SOURCE, "source_document": "NCEI DSI 6182",
                            "retrieval_mechanism": "NCEI direct historical archive", "source_object": url,
                            "archive_filename": name, "local_path": str(target), "checksum": checksum,
                            "retrieved_at_utc": retrieved, **parsed,
                            "availability_policy": "gfs_init_plus_10h_v1",
                            "available_at_utc": parsed["model_init_time_utc"] + pd.Timedelta(hours=10)})
    three_hourly = pd.DataFrame(extracted)
    hourly_parts = []
    for _, cycle in three_hourly.groupby("model_init_time_utc"):
        hourly = hourly_from_grid004(cycle)
        init = pd.to_datetime(hourly.model_init_time_utc, utc=True)
        hourly["forecast_lead_hours"] = ((pd.to_datetime(hourly.valid_time_utc, utc=True) - init)
                                         .dt.total_seconds() / 3600)
        hourly["available_at_utc"] = init + pd.Timedelta(hours=10)
        hourly["availability_policy"] = "gfs_init_plus_10h_v1"
        hourly["source"] = NCEI_GRID004_SOURCE
        hourly["model"] = "gfs-grid004-0p5"
        for station, indices in hourly.groupby("station").groups.items():
            source = cycle[cycle.station == station]
            hourly.loc[indices, "population_weight"] = source.population_weight.iloc[0]
            # All members belong to one initialization cycle. Retaining the
            # complete object set keeps interpolated values auditable to both
            # bracketing forecast products without collapsing station rows.
            hourly.loc[indices, "source_object"] = "|".join(sorted(source.source_object.unique()))
            hourly.loc[indices, "checksum"] = "|".join(sorted(source.checksum.unique()))
        hourly_parts.append(hourly)
    pd.concat(hourly_parts, ignore_index=True).to_csv(processed / "weather_vintages.csv", index=False)
    write_manifest(month_dir / "acquisition_manifest.gfs.json", records, month=args.month,
                   source_documents=args.source_document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
