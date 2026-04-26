#!/usr/bin/env python3
"""
collect_caiso_dataset_gridstatus_dotenv.py

Hourly data collection pipeline (MODULAR) for CAISO-related regions:
  regions = ['caiso', 'pge', 'sce', 'sdge', 'vea', 'mwd']

Collects, per region and per hour (UTC):
- Time key: YYYY-MM-DD:HH (UTC)
- Weighted-average weather features (up to 5 station points per region)
  via Open-Meteo Historical (archive) API
- Historical load (CAISO EMS hourly load XLSX library, with optional GridStatus API fallback)
- Derived degree-day features (CDD/HDD) using Fahrenheit temperature

Main fixes in this rewrite:
1. More robust CAISO XLSX parsing:
   - checks ALL candidate sheets instead of returning the first parseable one
   - supports alternate column names through alias mapping
   - prints workbook-level schema diagnostics
2. Explicit handling of missing load regions:
   - by default, raises an error if you request a region but no load column exists
   - optional flag lets you keep NaNs instead
3. Honest handling of MWD:
   - MWD is kept as a target region for weather
   - but historical EMS hourly load files often do NOT expose an MWD column
   - this script will not fabricate MWD load from another region
4. More consistent HTTP behavior:
   - uses a shared session for text/json/file downloads
5. Better diagnostics:
   - prints which load columns were found after parsing
   - prints missing-load summaries before writing output

Dependencies:
  pip install pandas requests openpyxl numpy python-dotenv
  # Optional but recommended for recent CAISO TAC load fallback:
  pip install gridstatusio

Usage:
  python src/data_collection/collect_caiso_dataset_gridstatus_dotenv.py --start 2024-01-01 --end 2024-01-07

If station weights CSV is missing and you want fallback behavior:
  python src/data_collection/collect_caiso_dataset_gridstatus_dotenv.py --start 2024-01-01 --end 2024-01-07 --allow-fallback

If you want the script to continue even when some requested regions have no load column:
  python src/data_collection/collect_caiso_dataset_gridstatus_dotenv.py --start 2024-01-01 --end 2024-01-07 --allow-missing-load-regions

To use GridStatus as a fallback for recent/missing CAISO load data:
  # Create a local .env file containing:
  # GRIDSTATUS_API_KEY=your_key_here
  python src/data_collection/collect_caiso_dataset_gridstatus_dotenv.py --start 2025-01-01 --end 2026-04-30 --load-source caiso_then_gridstatus

You can also point to a non-default environment file:
  python src/data_collection/collect_caiso_dataset_gridstatus_dotenv.py --start 2025-01-01 --end 2026-04-30 --env-file ./secrets/.env

To use GridStatus first and skip the CAISO XLSX scrape unless GridStatus fails:
  python src/data_collection/collect_caiso_dataset_gridstatus_dotenv.py --start 2025-01-01 --end 2026-04-30 --load-source gridstatus_then_caiso
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "load-forecasting-pipeline/2.0"})

# ----------------------------
# Config
# ----------------------------

CAISO_LOAD_LIBRARY_URL = "https://www.caiso.com/library/historical-ems-hourly-load"

OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

TZ_UTC = "UTC"
TZ_MARKET = "America/Los_Angeles"

TARGET_REGIONS = ["caiso", "pge", "sce", "sdge", "vea", "mwd"]

# Precomputed station weights file (recommended)
STATION_WEIGHTS_CSV = "./data/stations_population_weights.csv"

# Only used in fallback mode
REGION_PLACE_QUERIES: Dict[str, List[str]] = {
    "caiso": ["Los Angeles, CA", "San Francisco, CA", "San Diego, CA", "Sacramento, CA", "Fresno, CA"],
    "pge":   ["San Francisco, CA", "San Jose, CA", "Oakland, CA", "Sacramento, CA", "Fresno, CA"],
    "sce":   ["Los Angeles, CA", "Riverside, CA", "San Bernardino, CA", "Bakersfield, CA", "Palm Springs, CA"],
    "sdge":  ["San Diego, CA", "Chula Vista, CA", "Oceanside, CA", "El Cajon, CA", "Escondido, CA"],
    "vea":   ["Lancaster, CA", "Palmdale, CA", "Victorville, CA", "Barstow, CA", "Ridgecrest, CA"],
    "mwd":   ["Los Angeles, CA", "Long Beach, CA", "Anaheim, CA", "Santa Ana, CA", "Riverside, CA"],
}

# Only used in fallback mode
REGION_STATION_FALLBACKS: Dict[str, List[Tuple[str, float, float]]] = {
    "caiso": [
        ("Los Angeles", 34.0522, -118.2437),
        ("San Francisco", 37.7749, -122.4194),
        ("San Diego", 32.7157, -117.1611),
        ("Sacramento", 38.5816, -121.4944),
        ("Fresno", 36.7378, -119.7871),
    ],
    "pge": [
        ("San Francisco", 37.7749, -122.4194),
        ("San Jose", 37.3382, -121.8863),
        ("Oakland", 37.8044, -122.2711),
        ("Sacramento", 38.5816, -121.4944),
        ("Fresno", 36.7378, -119.7871),
    ],
    "sce": [
        ("Los Angeles", 34.0522, -118.2437),
        ("Riverside", 33.9806, -117.3755),
        ("San Bernardino", 34.1083, -117.2898),
        ("Bakersfield", 35.3733, -119.0187),
        ("Palm Springs", 33.8303, -116.5453),
    ],
    "sdge": [
        ("San Diego", 32.7157, -117.1611),
        ("Chula Vista", 32.6401, -117.0842),
        ("Oceanside", 33.1959, -117.3795),
        ("El Cajon", 32.7948, -116.9625),
        ("Escondido", 33.1192, -117.0864),
    ],
    "vea": [
        ("Lancaster", 34.6868, -118.1542),
        ("Palmdale", 34.5794, -118.1165),
        ("Victorville", 34.5361, -117.2912),
        ("Barstow", 34.8958, -117.0173),
        ("Ridgecrest", 35.6225, -117.6709),
    ],
    "mwd": [
        ("Los Angeles", 34.0522, -118.2437),
        ("Long Beach", 33.7701, -118.1937),
        ("Anaheim", 33.8366, -117.9143),
        ("Santa Ana", 33.7455, -117.8677),
        ("Riverside", 33.9806, -117.3755),
    ],
}

WEATHER_HOURLY_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
]

# Canonical load columns we want from CAISO files
CANONICAL_LOAD_COLS = ["caiso", "pge", "sce", "sdge", "vea", "mwd"]

# More robust alias mapping for historical workbook schema differences
LOAD_COLUMN_ALIASES: Dict[str, str] = {
    # CAISO total
    "caiso": "caiso",
    "caiso total": "caiso",
    "total": "caiso",
    "iso": "caiso",
    "system": "caiso",
    "system total": "caiso",

    # PGE
    "pge": "pge",
    "pge tac": "pge",
    "pge-tac": "pge",
    "pg&e": "pge",
    "pg&e tac": "pge",
    "pg&e-tac": "pge",

    # SCE
    "sce": "sce",
    "sce tac": "sce",
    "sce-tac": "sce",
    "southern california edison": "sce",

    # SDGE
    "sdge": "sdge",
    "sdg&e": "sdge",
    "sdge tac": "sdge",
    "sdge-tac": "sdge",
    "sdg&e tac": "sdge",
    "sdg&e-tac": "sdge",

    # VEA
    "vea": "vea",
    "vea tac": "vea",
    "vea-tac": "vea",

    # MWD
    "mwd": "mwd",
    "mwd tac": "mwd",
    "mwd-tac": "mwd",
}

GRIDSTATUS_DATASET = "caiso_load_hourly"
GRIDSTATUS_API_KEY_ENV = "GRIDSTATUS_API_KEY"
DEFAULT_ENV_FILE = ".env"

# GridStatus caiso_load_hourly uses TAC area names like PGE-TAC, SCE-TAC, etc.
GRIDSTATUS_TAC_AREA_ALIASES: Dict[str, str] = {
    "caiso": "caiso",
    "caiso tac": "caiso",
    "caiso-tac": "caiso",
    "pge": "pge",
    "pge tac": "pge",
    "pge-tac": "pge",
    "pg&e": "pge",
    "pg&e tac": "pge",
    "pg&e-tac": "pge",
    "sce": "sce",
    "sce tac": "sce",
    "sce-tac": "sce",
    "sdge": "sdge",
    "sdge tac": "sdge",
    "sdge-tac": "sdge",
    "sdg&e": "sdge",
    "sdg&e tac": "sdge",
    "sdg&e-tac": "sdge",
    "vea": "vea",
    "vea tac": "vea",
    "vea-tac": "vea",
    "mwd": "mwd",
    "mwd tac": "mwd",
    "mwd-tac": "mwd",
}

DATE_COLUMN_ALIASES = {
    "date", "opr_dt", "opr date", "trading date", "trade date", "market date"
}
HR_COLUMN_ALIASES = {
    "hr", "hour", "opr_hr", "he", "hour ending", "he hour"
}


# ----------------------------
# Utility / validation
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def parse_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def utc_hourly_index(start_date: date, end_date: date) -> pd.DatetimeIndex:
    """
    Exact UTC hourly index for the requested window, guaranteed 24h/day:
      [start_date 00:00, end_date 23:00] UTC.
    Returned as tz-naive timestamps interpreted as UTC.
    """
    start = pd.Timestamp(start_date.strftime("%Y-%m-%d 00:00:00"))
    end = pd.Timestamp(end_date.strftime("%Y-%m-%d 23:00:00"))
    return pd.date_range(start=start, end=end, freq="h")

def time_key_utc(ts_utc_naive: pd.Timestamp) -> str:
    return pd.to_datetime(ts_utc_naive).strftime("%Y-%m-%d:%H")

def c_to_f(x: pd.Series) -> pd.Series:
    return x.astype(float) * 9.0 / 5.0 + 32.0

def require_openpyxl() -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency: openpyxl is required to parse CAISO .xlsx load files.\n"
            "Install with:\n"
            "  pip install openpyxl\n"
            "or (conda):\n"
            "  conda install -c conda-forge openpyxl\n"
        ) from e

def _stable_params_key(url: str, params: dict) -> str:
    items = sorted((str(k), str(v)) for k, v in params.items())
    blob = url + "?" + "&".join([f"{k}={v}" for k, v in items])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def _cache_path(cache_dir: str, key: str) -> str:
    ensure_dir(cache_dir)
    return os.path.join(cache_dir, f"{key}.json")

def http_get_json(
    url: str,
    params: dict,
    *,
    timeout: int = 60,
    max_retries: int = 8,
    cache_dir: Optional[str] = "./data/cache/open_meteo_json",
) -> dict:
    """
    HTTP GET -> JSON with:
      - disk cache
      - explicit handling of 429
      - exponential backoff + jitter
    """
    path = None
    if cache_dir is not None:
        key = _stable_params_key(url, params)
        path = _cache_path(cache_dir, key)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)

    last_exc = None
    base_sleep = 1.5

    for attempt in range(max_retries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = base_sleep * (2 ** attempt)
                else:
                    wait = base_sleep * (2 ** attempt)

                wait = min(180.0, wait) + random.uniform(0.0, 0.5)
                print(f"[http] 429 rate-limited. sleeping {wait:.1f}s then retrying ({attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue

            r.raise_for_status()
            js = r.json()

            if path is not None:
                with open(path, "w") as f:
                    json.dump(js, f)

            return js

        except Exception as e:
            last_exc = e
            wait = min(180.0, base_sleep * (2 ** attempt)) + random.uniform(0.0, 0.5)
            print(f"[http] error: {type(e).__name__}: {e} | sleeping {wait:.1f}s ({attempt + 1}/{max_retries})")
            time.sleep(wait)

    raise RuntimeError(f"GET failed after {max_retries} retries: {url} params={params}") from last_exc

def http_get_text(url: str, *, timeout: int = 60, max_retries: int = 5) -> str:
    last_exc = None
    for attempt in range(max_retries):
        try:
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_exc = e
            wait = min(60.0, 1.5 * (2 ** attempt)) + random.uniform(0.0, 0.5)
            print(f"[http] text error: {type(e).__name__}: {e} | sleeping {wait:.1f}s ({attempt + 1}/{max_retries})")
            time.sleep(wait)
    raise RuntimeError(f"GET failed after {max_retries} retries: {url}") from last_exc

def download_file(url: str, dest_path: str, *, timeout: int = 120) -> None:
    ensure_dir(os.path.dirname(dest_path))
    if os.path.exists(dest_path):
        return

    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)

def _normalize_colname(c: str) -> str:
    s = str(c).strip().lower()
    s = re.sub(r"[_\-/]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _coerce_date_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(s, errors="coerce")

# def _market_hour_start_to_utc_naive(market_time_naive: pd.Series) -> pd.Series:
#     """
#     Interpret naive timestamps as America/Los_Angeles, then convert to UTC naive.
#     """
#     localized = market_time_naive.dt.tz_localize(
#         TZ_MARKET,
#         ambiguous="NaT",
#         nonexistent="shift_forward",
#     )
#     mask = localized.isna()
#     if mask.any():
#         localized.loc[mask] = market_time_naive.loc[mask].dt.tz_localize(
#             TZ_MARKET,
#             ambiguous=True,
#             nonexistent="shift_forward",
#         )
#     return localized.dt.tz_convert(TZ_UTC).dt.tz_localize(None)

def _market_hour_start_to_utc_naive(market_time_naive: pd.Series) -> pd.Series:
    """
    Interpret naive timestamps as America/Los_Angeles, then convert to UTC naive.
    Handles ambiguous DST hours without chained assignment.
    """
    first_pass = market_time_naive.dt.tz_localize(
        TZ_MARKET,
        ambiguous="NaT",
        nonexistent="shift_forward",
    )

    if first_pass.isna().any():
        second_pass = market_time_naive.dt.tz_localize(
            TZ_MARKET,
            ambiguous=True,
            nonexistent="shift_forward",
        )
        localized = first_pass.where(~first_pass.isna(), second_pass)
    else:
        localized = first_pass

    return localized.dt.tz_convert(TZ_UTC).dt.tz_localize(None)


# ----------------------------
# Station + population weights
# ----------------------------

@dataclass(frozen=True)
class Station:
    name: str
    latitude: float
    longitude: float

def _normalize_weights(weights: List[float]) -> List[float]:
    ws = [max(0.0, float(w)) for w in weights]
    s = float(sum(ws))
    if s <= 0:
        return [1.0 / len(ws)] * len(ws)
    return [w / s for w in ws]

def load_station_weights_csv(path: str = STATION_WEIGHTS_CSV) -> pd.DataFrame:
    """
    Expected columns:
      region, station_name, latitude, longitude, population_weight
    """
    if not os.path.exists(path):
        raise RuntimeError(
            f"Missing station weights file: {path}\n"
            "Generate it with:\n"
            "  python build_station_population_weights.py\n"
        )

    df = pd.read_csv(path)
    needed = {"region", "station_name", "latitude", "longitude", "population_weight"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Station weights CSV missing columns: {sorted(missing)}")

    df["region"] = df["region"].astype(str).str.lower()
    df["population_weight"] = pd.to_numeric(df["population_weight"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["region", "station_name", "latitude", "longitude", "population_weight"]).copy()
    return df

def collect_region_stations_and_weights_from_csv(
    region: str,
    *,
    start_date: date,
    end_date: date,
    max_stations: int = 5,
    weights_csv: str = STATION_WEIGHTS_CSV,
) -> Tuple[List[Station], List[float], pd.DataFrame]:
    _ = (start_date, end_date)

    df = load_station_weights_csv(weights_csv)
    df_r = df[df["region"] == region.lower()].copy()
    if df_r.empty:
        raise RuntimeError(f"No station weights found for region={region} in {weights_csv}")

    df_r = df_r.sort_values("population_weight", ascending=False).head(max_stations).copy()
    weights = _normalize_weights(df_r["population_weight"].tolist())
    df_r["population_weight"] = weights

    stations = [
        Station(
            name=str(row.station_name),
            latitude=float(row.latitude),
            longitude=float(row.longitude),
        )
        for row in df_r.itertuples(index=False)
    ]

    meta = df_r.copy()
    meta["region"] = region.lower()
    meta["source"] = "stations_population_weights_csv"
    return stations, weights, meta


# ----------------------------
# Fallback station selection (optional)
# ----------------------------

@dataclass(frozen=True)
class GeoStation:
    name: str
    latitude: float
    longitude: float
    population: float

def geocode_place(place_query: str) -> Optional[GeoStation]:
    js = http_get_json(
        OPEN_METEO_GEOCODE_URL,
        params={"name": place_query, "count": 1, "language": "en", "format": "json"},
        timeout=30,
    )
    results = js.get("results") or []
    if not results:
        return None
    r0 = results[0]
    pop = float(r0.get("population") or 0.0)
    return GeoStation(
        name=str(r0.get("name") or place_query),
        latitude=float(r0["latitude"]),
        longitude=float(r0["longitude"]),
        population=pop,
    )

def collect_region_stations_fallback(
    region: str,
    *,
    start_date: date,
    end_date: date,
    max_stations: int = 5,
) -> Tuple[List[Station], List[float], pd.DataFrame]:
    _ = (start_date, end_date)

    queries = REGION_PLACE_QUERIES.get(region, [])[:max_stations]
    geo: List[GeoStation] = []
    for q in queries:
        try:
            st = geocode_place(q)
        except Exception:
            st = None
        if st is not None:
            geo.append(st)

    if not geo:
        fb = REGION_STATION_FALLBACKS.get(region, [])[:max_stations]
        if not fb:
            raise RuntimeError(f"No fallback stations configured for region={region}")
        stations = [Station(name=n, latitude=lat, longitude=lon) for (n, lat, lon) in fb]
        weights = [1.0 / len(stations)] * len(stations)
        meta = pd.DataFrame([{
            "region": region.lower(),
            "station_name": s.name,
            "latitude": s.latitude,
            "longitude": s.longitude,
            "population_weight": w,
            "population_raw": 0.0,
            "source": "fallback_coords_uniform",
        } for s, w in zip(stations, weights)])
        return stations, weights, meta

    pops = [max(0.0, s.population) for s in geo]
    weights = _normalize_weights(pops)
    stations = [Station(name=s.name, latitude=s.latitude, longitude=s.longitude) for s in geo]
    meta = pd.DataFrame([{
        "region": region.lower(),
        "station_name": s.name,
        "latitude": s.latitude,
        "longitude": s.longitude,
        "population_weight": w,
        "population_raw": s.population,
        "source": "open_meteo_geocode",
    } for s, w in zip(geo, weights)])
    return stations, weights, meta


# ----------------------------
# Weather collection (UTC-aligned)
# ----------------------------

def fetch_weather_for_station(
    station: Station,
    *,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Fetch hourly weather for ONE station from Open-Meteo archive API.
    Requests timezone=UTC so every day has 24 UTC hours.
    """
    params = {
        "latitude": station.latitude,
        "longitude": station.longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "timezone": TZ_UTC,
    }
    js = http_get_json(OPEN_METEO_ARCHIVE_URL, params=params, timeout=90)
    hourly = js.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        raise RuntimeError(f"No hourly data returned for station={station.name} ({station.latitude},{station.longitude})")

    df = pd.DataFrame({"time_utc": pd.to_datetime(times)})
    for v in WEATHER_HOURLY_VARS:
        df[v] = hourly.get(v)

    full_idx = utc_hourly_index(start_date, end_date)
    df = df.set_index("time_utc").reindex(full_idx).reset_index().rename(columns={"index": "time_utc"})
    return df

def collect_weighted_region_weather(
    region: str,
    *,
    start_date: date,
    end_date: date,
    allow_fallback: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        stations, weights, meta = collect_region_stations_and_weights_from_csv(
            region, start_date=start_date, end_date=end_date, max_stations=5, weights_csv=STATION_WEIGHTS_CSV
        )
    except Exception as e:
        if not allow_fallback:
            raise
        print(f"[warn] station weights CSV failed for region={region}; falling back ({type(e).__name__}: {e})")
        stations, weights, meta = collect_region_stations_fallback(
            region, start_date=start_date, end_date=end_date, max_stations=5
        )

    station_dfs = [fetch_weather_for_station(s, start_date=start_date, end_date=end_date) for s in stations]

    out = pd.DataFrame({"time_utc": utc_hourly_index(start_date, end_date)})
    for v in WEATHER_HOURLY_VARS:
        out[v] = 0.0

    for df_s, w in zip(station_dfs, weights):
        merged = out.merge(df_s, on="time_utc", how="left", suffixes=("", "_s"))
        for v in WEATHER_HOURLY_VARS:
            merged[v] = merged[v].astype(float) + w * merged[f"{v}_s"].astype(float)
            merged.drop(columns=[f"{v}_s"], inplace=True)
        out = merged

    out["temperature_2m"] = c_to_f(out["temperature_2m"])
    out["apparent_temperature"] = c_to_f(out["apparent_temperature"])
    out["region"] = region
    return out, meta


# ----------------------------
# CAISO load collection
# ----------------------------

def extract_xlsx_links_from_caiso_library(html: str, base_url: str = "https://www.caiso.com") -> List[str]:
    hrefs = re.findall(r'href\s*=\s*"([^"]+\.xlsx)"', html, flags=re.IGNORECASE)
    hrefs += re.findall(r"href\s*=\s*'([^']+\.xlsx)'", html, flags=re.IGNORECASE)
    links = [urljoin(base_url, h) for h in hrefs]

    seen = set()
    out = []
    for u in links:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out

def _xlsx_maybe_relevant(url: str, start_date: date, end_date: date) -> bool:
    years = re.findall(r"(?:19|20)\d{2}", url)
    if not years:
        return True
    years_i = {int(y) for y in years}
    return any(start_date.year <= y <= end_date.year for y in years_i)

def _find_header_row(raw: pd.DataFrame) -> Optional[int]:
    scan = min(len(raw), 80)
    for i in range(scan):
        row = raw.iloc[i].astype(str).map(_normalize_colname).tolist()
        has_date = any(x in DATE_COLUMN_ALIASES for x in row)
        has_hr = any(x in HR_COLUMN_ALIASES for x in row)
        if has_date and has_hr:
            return i
    return None

def _standardize_load_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename historical workbook headers to canonical names when possible.
    """
    rename_map = {}
    for c in df.columns:
        norm = _normalize_colname(c)
        if norm in LOAD_COLUMN_ALIASES:
            rename_map[c] = LOAD_COLUMN_ALIASES[norm]
        elif norm in DATE_COLUMN_ALIASES:
            rename_map[c] = "date"
        elif norm in HR_COLUMN_ALIASES:
            rename_map[c] = "hr"

    df = df.rename(columns=rename_map).copy()
    df.columns = [_normalize_colname(c) for c in df.columns]
    return df

def _extract_candidate_from_sheet(xls: pd.ExcelFile, sheet: str) -> Optional[pd.DataFrame]:
    """
    Try to parse one sheet into:
      time_utc + any recognized load columns
    Returns None if this sheet doesn't look usable.
    """
    try:
        raw = pd.read_excel(xls, sheet_name=sheet, header=None)
    except Exception:
        return None

    header_row = _find_header_row(raw)
    if header_row is None:
        return None

    try:
        df = pd.read_excel(xls, sheet_name=sheet, header=header_row)
    except Exception:
        return None

    df = _standardize_load_columns(df)

    if "date" not in df.columns or "hr" not in df.columns:
        return None

    df["date"] = _coerce_date_series(df["date"])
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["date", "hr"]).copy()
    if df.empty:
        return None

    # CAISO files are usually hour-ending. HE1 corresponds to interval start 00:00.
    market_time = df["date"] + pd.to_timedelta(df["hr"].astype(int) - 1, unit="h")
    df["time_utc"] = _market_hour_start_to_utc_naive(market_time)

    out = df[["time_utc"]].copy()
    found_any = False
    for col in CANONICAL_LOAD_COLS:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
            found_any = True

    if not found_any:
        return None

    out = out.drop_duplicates(subset=["time_utc"]).sort_values("time_utc")
    return out

def _score_candidate_load_df(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Higher is better:
    1) number of recognized load columns
    2) number of rows
    """
    load_cols = [c for c in df.columns if c != "time_utc"]
    return (len(load_cols), len(df))

def parse_caiso_load_xlsx(path: str, *, debug: bool = False) -> pd.DataFrame:
    """
    Parse the BEST candidate sheet in a workbook.
    Unlike the old version, this does not return the first parseable sheet.
    """
    xls = pd.ExcelFile(path)
    candidates: List[Tuple[str, pd.DataFrame]] = []

    for sheet in xls.sheet_names:
        cand = _extract_candidate_from_sheet(xls, sheet)
        if cand is not None:
            candidates.append((sheet, cand))

    if not candidates:
        raise RuntimeError(f"Could not parse CAISO load XLSX schema: {os.path.basename(path)}")

    # Choose the richest candidate
    candidates.sort(key=lambda x: _score_candidate_load_df(x[1]), reverse=True)
    best_sheet, best_df = candidates[0]

    if debug:
        summaries = []
        for sheet, cand in candidates:
            cols = [c for c in cand.columns if c != "time_utc"]
            summaries.append(f"{sheet}: cols={cols} rows={len(cand):,}")
        print(f"[parse_xlsx] {os.path.basename(path)} chose sheet='{best_sheet}' | " + " | ".join(summaries))

    return best_df

def _merge_workbook_parts(parts: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge workbook/file parts on time_utc, preferring non-null values from later parts
    without losing columns found in only some workbooks.
    """
    if not parts:
        raise RuntimeError("No parsed load parts to merge.")

    out = pd.DataFrame({"time_utc": pd.to_datetime(pd.Series(dtype="datetime64[ns]"))})
    for part in parts:
        part = part.copy()
        part["time_utc"] = pd.to_datetime(part["time_utc"])
        if out.empty:
            out = part
            continue

        merged = out.merge(part, on="time_utc", how="outer", suffixes=("", "_new"))
        for col in list(merged.columns):
            if col.endswith("_new"):
                base = col[:-4]
                if base in merged.columns:
                    merged[base] = merged[base].combine_first(merged[col])
                else:
                    merged[base] = merged[col]
                merged.drop(columns=[col], inplace=True)
        out = merged

    out = out.sort_values("time_utc").drop_duplicates(subset=["time_utc"], keep="last")
    return out


def load_dotenv_file(env_file: Optional[str] = DEFAULT_ENV_FILE) -> None:
    """
    Load environment variables from a .env file, if available.

    This keeps secrets like GRIDSTATUS_API_KEY out of the script and out of
    shell history. Existing environment variables are not overwritten, so values
    exported in your terminal still take precedence over the .env file.
    """
    if not env_file:
        return

    if not os.path.exists(env_file):
        print(f"[dotenv] no {env_file} file found; using existing environment variables only")
        return

    try:
        from dotenv import load_dotenv
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency: python-dotenv is required to load API keys from a .env file.\n"
            "Install it with:\n"
            "  pip install python-dotenv\n"
            "Or set GRIDSTATUS_API_KEY directly in your shell environment."
        ) from e

    loaded = load_dotenv(dotenv_path=env_file, override=False)
    if loaded:
        print(f"[dotenv] loaded environment variables from {env_file}")
    else:
        print(f"[dotenv] checked {env_file}, but no variables were loaded")


def _get_gridstatus_client(api_key: Optional[str] = None):
    """
    Construct a GridStatus.io client lazily so the base pipeline does not require
    gridstatusio unless the GridStatus load source is actually used.
    """
    try:
        from gridstatusio import GridStatusClient
    except ImportError as e:
        raise RuntimeError(
            "Missing optional dependency: gridstatusio is required for GridStatus load fallback.\n"
            "Install it with:\n"
            "  pip install gridstatusio\n"
            f"Then put your API key in .env as:\n  {GRIDSTATUS_API_KEY_ENV}=your_key_here"
        ) from e

    key = api_key or os.environ.get(GRIDSTATUS_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"GridStatus API key not found. Add {GRIDSTATUS_API_KEY_ENV}=your_key_here to .env, "
            f"set {GRIDSTATUS_API_KEY_ENV} in your shell, or pass --gridstatus-api-key."
        )

    # gridstatusio has had a few constructor signatures over time. Try the most
    # explicit modern form first, then fall back to the positional form shown in
    # the course email.
    try:
        return GridStatusClient(api_key=key, return_format="pandas")
    except TypeError:
        try:
            return GridStatusClient(key, return_format="pandas")
        except TypeError:
            return GridStatusClient(key)

def _normalize_gridstatus_colname(c: str) -> str:
    s = str(c).strip().lower()
    s = re.sub(r"[\s\-/]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def _normalize_tac_area_name(x: object) -> Optional[str]:
    s = str(x).strip().lower()
    s = re.sub(r"[_/]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    compact_dash = s.replace(" ", "-")

    if s in GRIDSTATUS_TAC_AREA_ALIASES:
        return GRIDSTATUS_TAC_AREA_ALIASES[s]
    if compact_dash in GRIDSTATUS_TAC_AREA_ALIASES:
        return GRIDSTATUS_TAC_AREA_ALIASES[compact_dash]

    if s.startswith("pge") or s.startswith("pg&e"):
        return "pge"
    if s.startswith("sce"):
        return "sce"
    if s.startswith("sdge") or s.startswith("sdg&e"):
        return "sdge"
    if s.startswith("vea"):
        return "vea"
    if s.startswith("mwd"):
        return "mwd"
    if s.startswith("caiso") or s in {"system", "total", "iso"}:
        return "caiso"
    return None

def _choose_gridstatus_load_value_column(df: pd.DataFrame) -> str:
    """
    GridStatus' caiso_load_hourly schema normally contains a MW load column.
    This function keeps the script resilient if that column is named load,
    load_mw, value, demand, etc.
    """
    preferred = [
        "load_mw",
        "load",
        "demand_mw",
        "demand",
        "mw",
        "value",
    ]
    for col in preferred:
        if col in df.columns:
            return col

    forbidden = {
        "interval_start_local", "interval_start_utc", "interval_end_local", "interval_end_utc",
        "tac_area_name", "tac_area", "area", "market", "source", "publish_time", "publish_time_utc",
    }
    numeric_candidates = [
        c for c in df.columns
        if c not in forbidden and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    raise RuntimeError(
        "Could not identify the GridStatus load value column. "
        f"Columns after normalization were: {list(df.columns)}"
    )

def _gridstatus_time_to_utc_naive(df: pd.DataFrame) -> pd.Series:
    if "interval_start_utc" in df.columns:
        return pd.to_datetime(df["interval_start_utc"], errors="coerce", utc=True).dt.tz_convert(None)

    if "interval_start_local" in df.columns:
        local = pd.to_datetime(df["interval_start_local"], errors="coerce")
        if getattr(local.dt, "tz", None) is not None:
            return local.dt.tz_convert(TZ_UTC).dt.tz_localize(None)
        localized = local.dt.tz_localize(TZ_MARKET, ambiguous="infer", nonexistent="shift_forward")
        return localized.dt.tz_convert(TZ_UTC).dt.tz_localize(None)

    raise RuntimeError(
        "GridStatus data did not include interval_start_utc or interval_start_local. "
        f"Columns were: {list(df.columns)}"
    )

def _standardize_gridstatus_load_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert GridStatus caiso_load_hourly long-form TAC data into the same wide
    format used by the CAISO XLSX parser:
      time_utc, caiso, pge, sce, sdge, vea, mwd
    """
    if raw is None or len(raw) == 0:
        raise RuntimeError("GridStatus returned no load rows.")

    df = raw.copy()
    df.columns = [_normalize_gridstatus_colname(c) for c in df.columns]

    if "tac_area_name" not in df.columns:
        area_candidates = [c for c in ["tac_area", "area", "zone", "load_zone"] if c in df.columns]
        if area_candidates:
            df = df.rename(columns={area_candidates[0]: "tac_area_name"})
        else:
            raise RuntimeError(
                "GridStatus data did not include tac_area_name or a recognizable TAC area column. "
                f"Columns were: {list(df.columns)}"
            )

    value_col = _choose_gridstatus_load_value_column(df)
    df["time_utc"] = _gridstatus_time_to_utc_naive(df)
    df["region"] = df["tac_area_name"].apply(_normalize_tac_area_name)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["time_utc", "region", value_col]).copy()

    if df.empty:
        raise RuntimeError("After standardizing GridStatus TAC rows, no usable load rows remained.")

    wide = (
        df.pivot_table(
            index="time_utc",
            columns="region",
            values=value_col,
            aggfunc="last",
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    keep = ["time_utc"] + [c for c in CANONICAL_LOAD_COLS if c in wide.columns]
    return wide[keep].sort_values("time_utc").reset_index(drop=True)

def collect_gridstatus_load(
    *,
    start_date: date,
    end_date: date,
    api_key: Optional[str] = None,
    debug: bool = True,
) -> pd.DataFrame:
    """
    Fetch CAISO TAC hourly load from GridStatus.io and return the same wide UTC
    shape as collect_caiso_load().

    This is primarily useful for recent dates when CAISO's historical EMS XLSX
    library is unavailable, incomplete, or difficult to scrape.
    """
    client = _get_gridstatus_client(api_key)

    if debug:
        print(f"[gridstatus_load] fetching dataset={GRIDSTATUS_DATASET} start={start_date} end={end_date}")

    raw = client.get_dataset(
        dataset=GRIDSTATUS_DATASET,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        timezone="market",
    )

    load = _standardize_gridstatus_load_frame(raw)
    utc_idx = utc_hourly_index(start_date, end_date)
    load = load.set_index("time_utc").reindex(utc_idx).reset_index().rename(columns={"index": "time_utc"})

    if debug:
        found_cols = [c for c in load.columns if c != "time_utc"]
        print(f"[gridstatus_load] final columns after pivot/reindex: {found_cols}")
        for c in found_cols:
            n_missing = int(load[c].isna().sum())
            print(f"[gridstatus_load] missing {c}: {n_missing:,} / {len(load):,}")

    return load

def _count_missing_requested_load(load_wide: pd.DataFrame, regions: List[str]) -> int:
    missing = 0
    for r in regions:
        if r not in load_wide.columns:
            missing += len(load_wide)
        else:
            missing += int(load_wide[r].isna().sum())
    return missing

def _combine_load_sources(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer non-null values from primary; fill remaining holes from fallback.
    """
    out = primary.copy()
    out["time_utc"] = pd.to_datetime(out["time_utc"])
    fb = fallback.copy()
    fb["time_utc"] = pd.to_datetime(fb["time_utc"])

    merged = out.merge(fb, on="time_utc", how="outer", suffixes=("", "_fallback"))
    for col in CANONICAL_LOAD_COLS:
        fb_col = f"{col}_fallback"
        if col in merged.columns and fb_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[fb_col])
            merged.drop(columns=[fb_col], inplace=True)
        elif fb_col in merged.columns:
            merged[col] = merged[fb_col]
            merged.drop(columns=[fb_col], inplace=True)
    return merged.sort_values("time_utc").reset_index(drop=True)

def collect_load_with_strategy(
    *,
    start_date: date,
    end_date: date,
    regions: List[str],
    load_source: str,
    gridstatus_api_key: Optional[str] = None,
    debug: bool = True,
) -> pd.DataFrame:
    """
    Load-source strategy:
      caiso                 : only CAISO historical XLSX files
      gridstatus            : only GridStatus.io
      caiso_then_gridstatus : use CAISO first; fill missing/failures from GridStatus
      gridstatus_then_caiso : use GridStatus first; fill missing/failures from CAISO
    """
    valid = {"caiso", "gridstatus", "caiso_then_gridstatus", "gridstatus_then_caiso"}
    if load_source not in valid:
        raise ValueError(f"Unknown load_source={load_source!r}. Expected one of {sorted(valid)}")

    if load_source == "caiso":
        return collect_caiso_load(start_date=start_date, end_date=end_date, debug=debug)

    if load_source == "gridstatus":
        return collect_gridstatus_load(
            start_date=start_date,
            end_date=end_date,
            api_key=gridstatus_api_key,
            debug=debug,
        )

    if load_source == "caiso_then_gridstatus":
        try:
            primary = collect_caiso_load(start_date=start_date, end_date=end_date, debug=debug)
            before = _count_missing_requested_load(primary, regions)
            if before == 0:
                if debug:
                    print("[load_source] CAISO XLSX load complete; GridStatus fallback not needed.")
                return primary
            print(f"[load_source] CAISO XLSX load has {before:,} missing requested cells; trying GridStatus fallback.")
            fallback = collect_gridstatus_load(
                start_date=start_date,
                end_date=end_date,
                api_key=gridstatus_api_key,
                debug=debug,
            )
            combined = _combine_load_sources(primary, fallback)
            after = _count_missing_requested_load(combined, regions)
            print(f"[load_source] GridStatus fallback filled {before - after:,} requested load cells.")
            return combined
        except Exception as e:
            print(f"[load_source] CAISO XLSX load failed ({type(e).__name__}: {e}); trying GridStatus fallback.")
            return collect_gridstatus_load(
                start_date=start_date,
                end_date=end_date,
                api_key=gridstatus_api_key,
                debug=debug,
            )

    # gridstatus_then_caiso
    try:
        primary = collect_gridstatus_load(
            start_date=start_date,
            end_date=end_date,
            api_key=gridstatus_api_key,
            debug=debug,
        )
        before = _count_missing_requested_load(primary, regions)
        if before == 0:
            if debug:
                print("[load_source] GridStatus load complete; CAISO XLSX fallback not needed.")
            return primary
        print(f"[load_source] GridStatus load has {before:,} missing requested cells; trying CAISO XLSX fallback.")
        fallback = collect_caiso_load(start_date=start_date, end_date=end_date, debug=debug)
        combined = _combine_load_sources(primary, fallback)
        after = _count_missing_requested_load(combined, regions)
        print(f"[load_source] CAISO XLSX fallback filled {before - after:,} requested load cells.")
        return combined
    except Exception as e:
        print(f"[load_source] GridStatus load failed ({type(e).__name__}: {e}); trying CAISO XLSX fallback.")
        return collect_caiso_load(start_date=start_date, end_date=end_date, debug=debug)

def collect_caiso_load(
    *,
    start_date: date,
    end_date: date,
    cache_dir: str = "./data/cache/caiso_load_xlsx",
    debug: bool = True,
) -> pd.DataFrame:
    require_openpyxl()

    # Buffer in market-time terms, then later slice on UTC index
    market_start = start_date - timedelta(days=1)
    market_end = end_date + timedelta(days=1)

    html = http_get_text(CAISO_LOAD_LIBRARY_URL)
    links = extract_xlsx_links_from_caiso_library(html, base_url="https://www.caiso.com")
    links = [u for u in links if _xlsx_maybe_relevant(u, market_start, market_end)]

    if debug:
        print(f"[caiso_load] found {len(links)} xlsx links, first={links[0] if links else None}")

    if not links:
        raise RuntimeError("No .xlsx links found on CAISO load library page (after filtering).")

    utc_idx = utc_hourly_index(start_date, end_date)
    t0 = utc_idx[0] - pd.Timedelta(hours=48)
    t1 = utc_idx[-1] + pd.Timedelta(hours=48)

    parsed_parts: List[pd.DataFrame] = []
    for url in links:
        fn = url.split("/")[-1]
        path = os.path.join(cache_dir, fn)
        try:
            download_file(url, path)
            part = parse_caiso_load_xlsx(path, debug=debug)
            part = part[(part["time_utc"] >= t0) & (part["time_utc"] <= t1)].copy()
            if part.empty:
                continue
            parsed_parts.append(part)
            if debug:
                cols = [c for c in part.columns if c != "time_utc"]
                print(f"[caiso_load] parsed OK: {fn} rows={len(part):,} cols={cols}")
        except Exception as e:
            if debug:
                print(f"[caiso_load] skip: {fn} ({type(e).__name__}: {e})")
            continue

    if not parsed_parts:
        raise RuntimeError("Failed to parse any CAISO load XLSX files.")

    load = _merge_workbook_parts(parsed_parts)
    load["time_utc"] = pd.to_datetime(load["time_utc"])

    # Restrict to buffered UTC range, then reindex to strict requested range
    load = load[(load["time_utc"] >= t0) & (load["time_utc"] <= t1)].copy()
    load = load.set_index("time_utc").reindex(utc_idx).reset_index().rename(columns={"index": "time_utc"})

    if debug:
        found_cols = [c for c in load.columns if c != "time_utc"]
        print(f"[caiso_load] final columns after merge/reindex: {found_cols}")
        for c in found_cols:
            n_missing = int(load[c].isna().sum())
            print(f"[caiso_load] missing {c}: {n_missing:,} / {len(load):,}")

    return load


# ----------------------------
# Derived features + final assembly
# ----------------------------

def add_degree_days_f(df: pd.DataFrame) -> pd.DataFrame:
    if "temperature_2m" not in df.columns:
        raise ValueError("Expected temperature_2m in dataframe for degree-day computation.")
    t = df["temperature_2m"].astype(float)
    df["cdd_65f"] = (t - 65.0).clip(lower=0.0)
    df["hdd_65f"] = (65.0 - t).clip(lower=0.0)
    return df

def build_region_frame(
    region: str,
    *,
    start_date: date,
    end_date: date,
    load_wide: pd.DataFrame,
    allow_fallback: bool,
    allow_missing_load_regions: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    weather, meta = collect_weighted_region_weather(
        region,
        start_date=start_date,
        end_date=end_date,
        allow_fallback=allow_fallback,
    )

    load_col = region.lower()

    if load_col in load_wide.columns:
        merged = weather.merge(load_wide[["time_utc", load_col]], on="time_utc", how="left")
        merged.rename(columns={load_col: "load_mw"}, inplace=True)
    else:
        msg = (
            f"Requested region '{region}' but no corresponding load column was found in parsed CAISO load data. "
            f"Available load columns: {[c for c in load_wide.columns if c != 'time_utc']}. "
            f"This commonly happens for 'mwd' because the historical EMS source often does not include it as a standalone column."
        )
        if allow_missing_load_regions:
            print(f"[warn] {msg} Filling load_mw with NaN.")
            merged = weather.copy()
            merged["load_mw"] = np.nan
        else:
            raise RuntimeError(msg)

    merged["load_mw"] = pd.to_numeric(merged["load_mw"], errors="coerce")
    merged = add_degree_days_f(merged)
    merged["time_key"] = merged["time_utc"].apply(time_key_utc)

    keep = (
        ["region", "time_key", "time_utc"]
        + WEATHER_HOURLY_VARS
        + ["cdd_65f", "hdd_65f", "load_mw"]
    )
    region_df = merged[keep].copy()
    return region_df, meta

def summarize_dataset_missingness(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, g in dataset.groupby("region", sort=True):
        n = len(g)
        n_missing = int(g["load_mw"].isna().sum())
        rows.append({
            "region": region,
            "rows": n,
            "missing_load_rows": n_missing,
            "missing_load_fraction": n_missing / n if n else np.nan,
        })
    return pd.DataFrame(rows).sort_values("region").reset_index(drop=True)

def build_dataset(
    regions: List[str],
    *,
    start_date: date,
    end_date: date,
    allow_fallback: bool,
    allow_missing_load_regions: bool,
    load_source: str,
    gridstatus_api_key: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    load_wide = collect_load_with_strategy(
        start_date=start_date,
        end_date=end_date,
        regions=regions,
        load_source=load_source,
        gridstatus_api_key=gridstatus_api_key,
        debug=True,
    )

    parts: List[pd.DataFrame] = []
    metas: List[pd.DataFrame] = []

    for r in regions:
        df_r, meta_r = build_region_frame(
            r,
            start_date=start_date,
            end_date=end_date,
            load_wide=load_wide,
            allow_fallback=allow_fallback,
            allow_missing_load_regions=allow_missing_load_regions,
        )
        parts.append(df_r)
        metas.append(meta_r)

    dataset = pd.concat(parts, ignore_index=True)
    station_meta = pd.concat(metas, ignore_index=True)

    dataset.sort_values(["region", "time_utc"], inplace=True)
    return dataset, station_meta

def save_outputs(
    dataset: pd.DataFrame,
    station_meta: pd.DataFrame,
    *,
    start_date: date,
    end_date: date,
    out_dir: str = "./data/raw",
) -> Tuple[str, str]:
    ensure_dir(out_dir)

    fn1 = f"caiso_dataset_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
    fn2 = f"station_weights_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
    p1 = os.path.join(out_dir, fn1)
    p2 = os.path.join(out_dir, fn2)

    out = dataset.copy()
    out["time_utc"] = pd.to_datetime(out["time_utc"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(p1, index=False)

    station_meta.to_csv(p2, index=False)
    return p1, p2


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument(
        "--regions",
        default=",".join(TARGET_REGIONS),
        help=f"Comma-separated regions. Default: {','.join(TARGET_REGIONS)}",
    )
    ap.add_argument(
        "--allow-fallback",
        action="store_true",
        help="If station weights CSV is missing/broken, fall back to geocode/fallback coords.",
    )
    ap.add_argument(
        "--allow-missing-load-regions",
        action="store_true",
        help="Continue even if a requested region has no parsed load column; fill load_mw with NaN instead of raising.",
    )
    ap.add_argument(
        "--load-source",
        choices=["caiso", "gridstatus", "caiso_then_gridstatus", "gridstatus_then_caiso"],
        default="caiso_then_gridstatus",
        help=(
            "Load data source strategy. Default caiso_then_gridstatus keeps the historical CAISO XLSX path "
            "but uses GridStatus.io to fill recent/missing load data."
        ),
    )
    ap.add_argument(
        "--gridstatus-api-key",
        default=None,
        help=(
            f"GridStatus.io API key. Prefer putting {GRIDSTATUS_API_KEY_ENV}=... in .env "
            "instead of passing this on the command line."
        ),
    )
    ap.add_argument(
        "--env-file",
        default=DEFAULT_ENV_FILE,
        help=(
            "Path to dotenv file used to load environment variables before GridStatus is called. "
            f"Default: {DEFAULT_ENV_FILE}"
        ),
    )
    args = ap.parse_args()

    load_dotenv_file(args.env_file)

    start_date = parse_yyyy_mm_dd(args.start)
    end_date = parse_yyyy_mm_dd(args.end)
    if end_date < start_date:
        raise ValueError("end must be >= start")

    regions = [r.strip().lower() for r in args.regions.split(",") if r.strip()]
    for r in regions:
        if r not in TARGET_REGIONS:
            raise ValueError(f"Unknown region '{r}'. Expected one of: {TARGET_REGIONS}")

    dataset, station_meta = build_dataset(
        regions=regions,
        start_date=start_date,
        end_date=end_date,
        allow_fallback=args.allow_fallback,
        allow_missing_load_regions=args.allow_missing_load_regions,
        load_source=args.load_source,
        gridstatus_api_key=args.gridstatus_api_key,
    )

    missing_summary = summarize_dataset_missingness(dataset)
    print("\n[dataset] load missingness by region:")
    print(missing_summary.to_string(index=False))

    p1, p2 = save_outputs(
        dataset,
        station_meta,
        start_date=start_date,
        end_date=end_date,
    )

    print(f"\nSaved dataset: {p1}")
    print(f"Saved station weights: {p2}")
    print(f"Rows: {len(dataset):,} | Columns: {len(dataset.columns)}")

if __name__ == "__main__":
    main()
