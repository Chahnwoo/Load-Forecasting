"""Offline-testable acquisition primitives for strict backtest inputs."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

CAISO_ENDPOINT = "https://oasis.caiso.com/oasisapi/SingleZip"
CAISO_QUERY = "SLD_FCST"
NCEI_GRID004_SOURCE = "noaa-ncei-gfs-grid004-0p5"
NCEI_GRID004_ROOT = (
    "https://www.ncei.noaa.gov/data/global-forecast-system/access/historical/forecast"
)
GRID004_RE = re.compile(r"^gfs_4_(\d{8})_(00|06|12|18)00_(\d{3})\.grb2$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def caiso_request(start_utc: pd.Timestamp, end_utc: pd.Timestamp,
                  market_run_id: str = "DAM") -> tuple[str, dict[str, str]]:
    """Return the exact OASIS request used for one immutable raw response."""
    if market_run_id not in {"DAM", "ACTUAL"}:
        raise ValueError("CAISO market_run_id must be one of: ACTUAL, DAM")

    def oasis_time(value: pd.Timestamp) -> str:
        return pd.Timestamp(value).tz_convert("UTC").strftime("%Y%m%dT%H:%M-0000")

    params = {
        "queryname": CAISO_QUERY,
        "startdatetime": oasis_time(start_utc),
        "enddatetime": oasis_time(end_utc),
        "version": "1",
        "resultformat": "6",
        "market_run_id": market_run_id,
    }
    return f"{CAISO_ENDPOINT}?{urlencode(params)}", params


def parse_grid004_name(name: str) -> dict[str, object]:
    match = GRID004_RE.fullmatch(Path(name).name)
    if not match:
        raise ValueError(f"not an NCEI Grid 004 forecast object: {name}")
    date, cycle, lead = match.groups()
    init = pd.Timestamp(datetime.strptime(date + cycle, "%Y%m%d%H"), tz="UTC")
    lead_hours = int(lead)
    if lead_hours > 192 or lead_hours % 3:
        raise ValueError(f"Grid 004 forecast hour must be 000..192 in three-hour steps: {name}")
    return {"model_init_time_utc": init, "forecast_lead_hours": lead_hours,
            "valid_time_utc": init + pd.Timedelta(hours=lead_hours), "filename": Path(name).name}


def grid004_object_url(name: str) -> str:
    parsed = parse_grid004_name(name)
    day = parsed["model_init_time_utc"].strftime("%Y%m%d")
    return f"{NCEI_GRID004_ROOT}/{day[:6]}/{day}/{parsed['filename']}"


def hourly_from_grid004(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert one station/cycle's 3-hour Grid 004 values to hourly values.

    Instantaneous fields are time-interpolated. APCP is a three-hour amount and
    is distributed uniformly over its interval. DSWRF is a three-hour mean flux
    and is held constant over its interval. Callers must first verify the GRIB
    statistical-process metadata represented by ``*_period_hours``.
    """
    required = {"station", "model_init_time_utc", "valid_time_utc", "temperature_2m",
                "relative_humidity_2m", "cloud_cover", "wind_speed_10m",
                "precipitation", "precipitation_period_hours", "shortwave_radiation",
                "shortwave_period_hours"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Grid 004 extraction missing fields: {sorted(missing)}")
    groups = []
    instantaneous = ["temperature_2m", "relative_humidity_2m", "cloud_cover", "wind_speed_10m"]
    for (_, init), part in frame.groupby(["station", "model_init_time_utc"]):
        part = part.copy().sort_values("valid_time_utc").set_index(pd.to_datetime(part.valid_time_utc, utc=True))
        if part.index.duplicated().any() or not part.index.to_series().diff().dropna().eq(pd.Timedelta(hours=3)).all():
            raise ValueError("Grid 004 cycle must be complete and contiguous at three-hour resolution")
        if not part["precipitation_period_hours"].eq(3).all():
            raise ValueError("APCP must have verified three-hour accumulation semantics")
        if not part["shortwave_period_hours"].eq(3).all():
            raise ValueError("DSWRF must have verified three-hour average semantics")
        idx = pd.date_range(part.index.min(), part.index.max(), freq="h")
        out = pd.DataFrame(index=idx)
        for column in instantaneous:
            out[column] = pd.to_numeric(part[column]).reindex(idx).interpolate(method="time")
        # A value at T describes (T-3h, T]; map it to the three ending hours.
        out["precipitation"] = 0.0
        out["shortwave_radiation"] = float("nan")
        for end, row in part.iloc[1:].iterrows():
            hours = pd.date_range(end - pd.Timedelta(hours=2), end, freq="h")
            out.loc[hours, "precipitation"] = row.precipitation / 3.0
            out.loc[hours, "shortwave_radiation"] = row.shortwave_radiation
        out["station"] = part.station.iloc[0]
        out["model_init_time_utc"] = init
        out["valid_time_utc"] = out.index
        temp_f = out.temperature_2m * 9 / 5 + 32
        out["cdd_65f"] = (temp_f - 65).clip(lower=0)
        out["hdd_65f"] = (65 - temp_f).clip(lower=0)
        # NOAA heat-index and wind-chill regressions in their defined ranges.
        rh = out.relative_humidity_2m
        hi = (-42.379 + 2.04901523 * temp_f + 10.14333127 * rh
              - .22475541 * temp_f * rh - .00683783 * temp_f**2
              - .05481717 * rh**2 + .00122874 * temp_f**2 * rh
              + .00085282 * temp_f * rh**2 - .00000199 * temp_f**2 * rh**2)
        wind_mph = out.wind_speed_10m * 2.2369362921
        wind_chill = (35.74 + .6215 * temp_f - 35.75 * wind_mph**.16
                      + .4275 * temp_f * wind_mph**.16)
        apparent_f = temp_f.where(~((temp_f <= 50) & (wind_mph > 3)), wind_chill)
        apparent_f = apparent_f.where(~((temp_f >= 80) & (rh >= 40)), hi)
        out["apparent_temperature"] = (apparent_f - 32) * 5 / 9
        groups.append(out.reset_index(drop=True))
    return pd.concat(groups, ignore_index=True)


def write_manifest(path: str | Path, records: list[dict], *, month: str,
                   source_documents: list[str] | None = None) -> None:
    if not source_documents or any(not str(item).strip() for item in source_documents):
        raise ValueError("acquisition manifest requires immutable source-document references")
    for index, record in enumerate(records):
        missing = [key for key in ("source", "checksum", "retrieved_at_utc") if not record.get(key)]
        if missing:
            raise ValueError(f"acquisition record {index} missing provenance: {missing}")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(record["checksum"])):
            raise ValueError(f"acquisition record {index} has invalid SHA-256 provenance")
    payload = {"schema_version": "strict-acquisition-manifest-v1", "month": month,
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_documents": source_documents, "acquisitions": records}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
