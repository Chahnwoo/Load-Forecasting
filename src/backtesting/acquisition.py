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
GDEX_SOURCE = "nsf-ncar-gdex-gfs-0p25"
GDEX_MODEL = "gfs-0p25"
GDEX_DATASET = "d084001"
GDEX_REFERENCE = "https://doi.org/10.5065/D65D8PWK"
GDEX_ROOT = "https://tds.gdex.ucar.edu/thredds"
GDEX_RE = re.compile(r"^gfs\.0p25\.(\d{8})(00|06|12|18)\.f(\d{3})\.grib2$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def caiso_request(start_utc: pd.Timestamp, end_utc: pd.Timestamp,
                  market_run_id: str = "DAM") -> tuple[str, dict[str, str]]:
    if market_run_id not in {"DAM", "ACTUAL"}:
        raise ValueError("CAISO market_run_id must be one of: ACTUAL, DAM")
    def oasis_time(value: pd.Timestamp) -> str:
        return pd.Timestamp(value).tz_convert("UTC").strftime("%Y%m%dT%H:%M-0000")
    params = {"queryname": CAISO_QUERY, "startdatetime": oasis_time(start_utc),
              "enddatetime": oasis_time(end_utc), "version": "1", "resultformat": "6",
              "market_run_id": market_run_id}
    return f"{CAISO_ENDPOINT}?{urlencode(params)}", params


def parse_gdex_name(name: str) -> dict[str, object]:
    """Parse a canonical d084001 object name and independently validate its cycle."""
    filename = Path(name).name
    match = GDEX_RE.fullmatch(filename)
    if not match:
        raise ValueError(f"not a GDEX d084001 GFS forecast object: {name}")
    date, cycle, lead = match.groups()
    try:
        init = pd.Timestamp(datetime.strptime(date + cycle, "%Y%m%d%H"), tz="UTC")
    except ValueError as exc:
        raise ValueError(f"invalid GDEX calendar cycle: {name}") from exc
    lead_hours = int(lead)
    if (lead_hours > 384 or
            (lead_hours <= 240 and lead_hours % 3) or
            (lead_hours > 240 and lead_hours % 12)):
        raise ValueError(f"invalid GDEX forecast lead: {name}")
    return {"model_init_time_utc": init, "forecast_lead_hours": lead_hours,
            "valid_time_utc": init + pd.Timedelta(hours=lead_hours), "filename": filename}


def gdex_object_url(name: str) -> str:
    parsed = parse_gdex_name(name)
    day = parsed["model_init_time_utc"].strftime("%Y%m%d")
    return f"{GDEX_ROOT}/fileServer/files/g/{GDEX_DATASET}/{day[:4]}/{day}/{parsed['filename']}"


def gdex_ncss_url(name: str, variables: list[str], *, north: float, south: float,
                  east: float, west: float) -> tuple[str, dict[str, str]]:
    """Construct a deterministic small-box NCSS request for an exact backing object."""
    parsed = parse_gdex_name(name)
    if not variables or any(not str(v).strip() for v in variables):
        raise ValueError("NCSS requires discovered variable names")
    if not (-90 <= south <= north <= 90 and -180 <= west <= east <= 180):
        raise ValueError("invalid NCSS bounding box")
    day = parsed["model_init_time_utc"].strftime("%Y%m%d")
    valid_time = parsed["model_init_time_utc"] + pd.Timedelta(
        hours=parsed["forecast_lead_hours"])
    if valid_time != parsed["valid_time_utc"]:
        raise ValueError(f"GDEX valid time does not match backing object identity: {name}")
    base = f"{GDEX_ROOT}/ncss/grid/files/g/{GDEX_DATASET}/{day[:4]}/{day}/{parsed['filename']}"
    params = {"var": ",".join(sorted(set(variables))), "north": str(north), "south": str(south),
              "east": str(east), "west": str(west), "horizStride": "1", "accept": "netCDF",
              "time": valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"), "addLatLon": "true"}
    return f"{base}?{urlencode(params)}", params


def hourly_from_gdex(frame: pd.DataFrame) -> pd.DataFrame:
    """Build hourly fields using per-product GRIB statistical interval metadata."""
    required = {"station", "model_init_time_utc", "valid_time_utc", "temperature_2m",
                "relative_humidity_2m", "cloud_cover", "u_wind_10m", "v_wind_10m",
                "precipitation", "precipitation_period_hours", "shortwave_radiation",
                "shortwave_period_hours"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"GDEX extraction missing fields: {sorted(missing)}")
    results = []
    instantaneous = ["temperature_2m", "relative_humidity_2m", "cloud_cover", "u_wind_10m", "v_wind_10m"]
    for (_, init), part in frame.groupby(["station", "model_init_time_utc"]):
        part = part.copy().sort_values("valid_time_utc")
        part.index = pd.to_datetime(part.valid_time_utc, utc=True)
        if part.index.duplicated().any() or len(part) < 2:
            raise ValueError("GDEX cycle must contain unique adjacent forecast products")
        spacing = part.index.to_series().diff().dropna()
        if spacing.nunique() != 1 or spacing.iloc[0] > pd.Timedelta(hours=3):
            raise ValueError("GDEX cycle must be complete and contiguous at its forecast cadence")
        idx = pd.date_range(part.index.min(), part.index.max(), freq="h")
        out = pd.DataFrame(index=idx)
        for column in instantaneous:
            out[column] = pd.to_numeric(part[column]).reindex(idx).interpolate(method="time", limit_area="inside")
        out["precipitation"] = float("nan"); out["shortwave_radiation"] = float("nan")
        for end, row in part.iterrows():
            pp, sp = int(row.precipitation_period_hours), int(row.shortwave_period_hours)
            if pp <= 0 or sp <= 0:
                raise ValueError("statistical periods must be positive metadata-derived hours")
            precipitation_hours = pd.date_range(end-pd.Timedelta(hours=pp-1), end, freq="h")
            shortwave_hours = pd.date_range(end-pd.Timedelta(hours=sp-1), end, freq="h")
            # A statistical interval may begin before the requested output range. Preserve
            # its true duration while assigning only the hours represented in that range.
            out.loc[precipitation_hours.intersection(idx), "precipitation"] = row.precipitation / pp
            out.loc[shortwave_hours.intersection(idx), "shortwave_radiation"] = row.shortwave_radiation
        out["wind_speed_10m"] = (out.u_wind_10m ** 2 + out.v_wind_10m ** 2) ** .5
        out["station"] = part.station.iloc[0]; out["model_init_time_utc"] = init; out["valid_time_utc"] = out.index
        temp_f = out.temperature_2m * 9 / 5 + 32
        out["cdd_65f"] = (temp_f - 65).clip(lower=0); out["hdd_65f"] = (65-temp_f).clip(lower=0)
        rh, mph = out.relative_humidity_2m, out.wind_speed_10m * 2.2369362921
        hi = (-42.379+2.04901523*temp_f+10.14333127*rh-.22475541*temp_f*rh-.00683783*temp_f**2-.05481717*rh**2+.00122874*temp_f**2*rh+.00085282*temp_f*rh**2-.00000199*temp_f**2*rh**2)
        wc = 35.74+.6215*temp_f-35.75*mph**.16+.4275*temp_f*mph**.16
        apparent = temp_f.where(~((temp_f <= 50) & (mph > 3)), wc).where(~((temp_f >= 80) & (rh >= 40)), hi)
        out["apparent_temperature"] = (apparent-32)*5/9
        results.append(out.reset_index(drop=True))
    return pd.concat(results, ignore_index=True)


def write_manifest(path: str | Path, records: list[dict], *, month: str,
                   source_documents: list[str] | None = None) -> None:
    if not source_documents or any(not str(item).strip() for item in source_documents):
        raise ValueError("acquisition manifest requires immutable source-document references")
    for index, record in enumerate(records):
        missing = [key for key in ("source", "checksum", "retrieved_at_utc") if not record.get(key)]
        if record.get("source") == GDEX_SOURCE:
            gdex_keys = ("dataset", "dataset_reference", "backing_source_object",
                         "backing_fileserver_url", "model_init_time_utc", "forecast_lead_hours",
                         "valid_time_utc", "ncss_request_url", "request_parameters",
                         "raw_subset_local_filename", "availability_policy", "available_at_utc")
            missing.extend(key for key in gdex_keys if record.get(key) in (None, "", {}))
        if missing: raise ValueError(f"acquisition record {index} missing provenance: {missing}")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(record["checksum"])):
            raise ValueError(f"acquisition record {index} has invalid SHA-256 provenance")
    payload = {"schema_version": "strict-acquisition-manifest-v1", "month": month,
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_documents": source_documents, "acquisitions": records}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str)+"\n", encoding="utf-8")
