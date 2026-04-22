#!/usr/bin/env python3
"""
build_station_population_weights.py

Build a stable station list + population-based weights for each CAISO region.

Why this exists:
- Open-Meteo geocoding "population" is often missing -> weights become uniform.
- Geocoding can fail transiently -> no stations.
- We want deterministic stations and weights that can be reused by daily pipelines.

Approach:
1) Use deterministic fallback station coordinates per region (up to 5).
2) For each station, estimate "population near station" using WorldPop stats API:
     https://api.worldpop.org/v1/services/stats?dataset=wpgppop&year=2020&geojson=...&runasync=false
   The response includes total_population for the polygon. :contentReference[oaicite:1]{index=1}
3) Normalize station populations into weights (sum to 1 per region).
4) Save to ./data/stations_population_weights.csv

Dependencies:
  pip install pandas requests

Usage:
  python build_station_population_weights.py
  # optional:
  python build_station_population_weights.py --radius-km 30 --year 2020
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests


# ----------------------------
# Config
# ----------------------------

# WorldPop API docs: root https://api.worldpop.org/v1/services and services/stats described here. :contentReference[oaicite:2]{index=2}
WORLDPOP_STATS_URL = "https://api.worldpop.org/v1/services/stats"
WORLDPOP_TASK_URL = "https://api.worldpop.org/v1/tasks/{taskid}"

OUT_PATH = "./data/stations_population_weights.csv"

# Deterministic stations (up to 5 per region)
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


# ----------------------------
# HTTP helpers
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def http_get_json(url: str, params: dict, *, timeout: int = 60) -> dict:
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "load-forecasting/1.0"})
    r.raise_for_status()
    return r.json()

def http_get_json_no_params(url: str, *, timeout: int = 60) -> dict:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "load-forecasting/1.0"})
    r.raise_for_status()
    return r.json()


# ----------------------------
# Geometry: circle polygon (approx)
# ----------------------------

def _km_to_deg_lat(km: float) -> float:
    return km / 111.0

def _km_to_deg_lon(km: float, lat: float) -> float:
    return km / (111.0 * math.cos(math.radians(lat)) + 1e-9)

def circle_polygon_lonlat(lon: float, lat: float, radius_km: float, n_points: int = 48) -> List[List[float]]:
    """
    Returns a closed ring polygon (lon,lat) approximating a circle.
    """
    dlat = _km_to_deg_lat(radius_km)
    dlon = _km_to_deg_lon(radius_km, lat)

    ring: List[List[float]] = []
    for i in range(n_points):
        th = 2.0 * math.pi * (i / n_points)
        y = lat + dlat * math.sin(th)
        x = lon + dlon * math.cos(th)
        ring.append([x, y])

    # close ring
    ring.append(ring[0])
    return ring

def make_geojson_circle(lon: float, lat: float, radius_km: float) -> dict:
    ring = circle_polygon_lonlat(lon, lat, radius_km)
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        ],
    }


# ----------------------------
# WorldPop population query
# ----------------------------

def worldpop_total_population_for_circle(
    *,
    lon: float,
    lat: float,
    radius_km: float,
    year: int,
    dataset: str = "wpgppop",
    runasync: bool = False,
    poll_seconds: float = 0.8,
    max_polls: int = 40,
) -> float:
    """
    Uses WorldPop stats API to get total_population within a circle polygon.
    - dataset wpgppop and year 2000-2020 are documented. :contentReference[oaicite:3]{index=3}
    """
    geojson = make_geojson_circle(lon, lat, radius_km)

    params = {
        "dataset": dataset,
        "year": year,
        "geojson": json.dumps(geojson),
        "runasync": "true" if runasync else "false",
    }

    js = http_get_json(WORLDPOP_STATS_URL, params=params, timeout=60)

    # If server returns finished immediately
    if js.get("status") == "finished" and isinstance(js.get("data"), dict):
        val = js["data"].get("total_population")
        try:
            return float(val)
        except Exception:
            return 0.0

    # If server created an async task
    taskid = js.get("taskid")
    if not taskid:
        # Some error shape
        return 0.0

    task_url = WORLDPOP_TASK_URL.format(taskid=taskid)
    for _ in range(max_polls):
        tjs = http_get_json_no_params(task_url, timeout=60)
        if tjs.get("status") == "finished" and isinstance(tjs.get("data"), dict):
            val = tjs["data"].get("total_population")
            try:
                return float(val)
            except Exception:
                return 0.0
        if tjs.get("error") is True:
            return 0.0
        time.sleep(poll_seconds)

    return 0.0


# ----------------------------
# Weights
# ----------------------------

def normalize_weights(populations: List[float]) -> List[float]:
    pops = [max(0.0, float(p)) for p in populations]
    total = sum(pops)
    if total <= 0:
        return [1.0 / len(pops)] * len(pops)
    return [p / total for p in pops]


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-km", type=float, default=30.0, help="Radius around each station used for population total.")
    ap.add_argument("--year", type=int, default=2020, help="WorldPop year (wpgppop supports 2000-2020).")
    ap.add_argument("--max-stations", type=int, default=5, help="Max stations per region.")
    ap.add_argument("--dataset", type=str, default="wpgppop", help="WorldPop dataset (default wpgppop).")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    rows = []
    for region, sts in REGION_STATION_FALLBACKS.items():
        for (name, lat, lon) in sts[: args.max_stations]:
            rows.append({"region": region, "station_name": name, "latitude": lat, "longitude": lon})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No fallback stations configured.")

    pops = []
    for r in df.itertuples(index=False):
        if args.debug:
            print(f"[worldpop] region={r.region} station={r.station_name} lat={r.latitude:.4f} lon={r.longitude:.4f}")
        p = worldpop_total_population_for_circle(
            lon=float(r.longitude),
            lat=float(r.latitude),
            radius_km=float(args.radius_km),
            year=int(args.year),
            dataset=str(args.dataset),
            runasync=False,
        )
        pops.append(p)

    df["population_est"] = pops

    df["population_weight"] = 0.0
    for region, grp in df.groupby("region", sort=False):
        w = normalize_weights(grp["population_est"].tolist())
        df.loc[grp.index, "population_weight"] = w

    ensure_dir(os.path.dirname(OUT_PATH))
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    # quick sanity print
    print(df.groupby("region")["population_weight"].sum())

if __name__ == "__main__":
    main()