# Strict CAISO day-ahead backtest data contract

## Historical forecast source

Strict mode uses NSF NCAR GDEX dataset **d084001**, the cycle-addressable NCEP
GFS 0.25-degree historical forecast grids. Its source and model identifiers are
`nsf-ncar-gdex-gfs-0p25` and `gfs-0p25`. NCEI Grid 004 direct acquisition was
abandoned because that older 0.5-degree contract was not a reliable path for
the required December archive; the obsolete `noaa-ncei-gfs-grid004-0p5`,
READY pseudo-analysis, realized weather, and reanalysis sources are rejected.
The dataset reference recorded by the implementation is
<https://doi.org/10.5065/D65D8PWK>; its applicability to this exact GDEX
collection must be confirmed during certification.

The following December-adjacent backing object was independently verified by
HEAD (HTTP 200):

`https://tds.gdex.ucar.edu/thredds/fileServer/files/g/d084001/2025/20251130/gfs.0p25.2025113000.f030.grib2`

It is about 542 MB. The acquisition path **never downloads that full global
object**. It discovers variable names from the object's NCSS `dataset.xml`,
then requests NetCDF4 only for a tightly padded box around the repository's
CAISO stations through `/thredds/ncss/grid/files/g/d084001/...`. The exact raw
NCSS response bytes are retained and SHA-256 checksummed.

## Frozen origin and complete cycles

Every operating day has a D-1 07:00 Pacific frozen origin. A cycle is eligible
only when `model_init_time_utc + 10h <= forecast_cutoff_utc`, under the retained
`gfs_init_plus_10h_v1` conservative policy. THREDDS `Last-Modified` is archive
ingest/publication metadata and is never interpreted as operational
availability. Selection chooses the newest eligible initialization that is
complete across every station for every target; it cannot combine a partial
new cycle with an older one.

Canonical names are `gfs.0p25.YYYYMMDDHH.fFFF.grib2`, where the cycle must be
00/06/12/18 UTC. The name determines initialization and forecast lead, and
valid time must equal their sum. Malformed dates, cycles, and leads fail closed.

## NCSS, temporal semantics, and provenance

The required physical fields are 2 m temperature and relative humidity, total
cloud cover, 10 m U/V wind, total precipitation, and downward shortwave
radiation. Names are matched from NCSS metadata rather than guessed. Each
NetCDF variable's GRIB attributes must expose the precipitation accumulation
or shortwave averaging interval. Positive metadata-derived periods are used:
accumulations are divided over `(T-period,T]`, means are held over that
interval, and instantaneous fields are interpolated only between contiguous
forecast products in one initialization. Wind speed is derived from U/V.
CDD/HDD and apparent temperature are calculated only after hourly fields exist.

Station rows and repository population weights remain intact through
validation and are aggregated only afterward. Each acquired subset manifest
record preserves dataset/reference, backing name and fileServer URL,
initialization/lead/valid time, NCSS URL and parameters, discovered variables,
raw subset filename, retrieval timestamp, checksum of the exact consumed
bytes, source-document references, availability policy, and availability.
Hourly interpolated station rows retain all contributing subset URLs and
checksums.

## Local commands

Install `requirements.txt` (including xarray and netCDF4). First run this exact
single-object smoke test; it retrieves only a CAISO-sized subset:

```bash
python scripts/acquire_gdex_gfs_0p25.py --month 2025-12 --smoke-test \
  --stations data/stations_population_weights.csv \
  --source-document https://rda.ucar.edu/datasets/d084001/
```

Then run the one-command acquisition and assembly workflow:

```bash
python scripts/acquire_strict_backtest_inputs.py --month 2025-12 \
  --stations data/stations_population_weights.csv \
  --caiso-source-document /immutable/docs/caiso-1210-v19.8.pdf \
  --gdex-source-document https://rda.ucar.edu/datasets/d084001/
```

The weather-only acquisition command is:

```bash
python scripts/acquire_gdex_gfs_0p25.py --month 2025-12 \
  --stations data/stations_population_weights.csv \
  --source-document https://rda.ucar.edu/datasets/d084001/
```

Raw subsets reside in `data/backtesting/2025-12/raw/gdex_ncss`; sidecars make
the workflow resumable and reject URL/checksum mismatches. Manifests and
`processed/weather_vintages.csv` are written beneath the month directory.
No metrics are calculated.

## Remaining certification requirements

Before certifying December, validate the live `dataset.xml` descriptions and
unique matches for all seven fields; confirm coordinate/dimension names,
units, nearest-grid behavior, GRIB interval attributes (including lead-zero
products and any interval changes), and NCSS response content type; verify the
recorded DOI/reference; checksum every subset; validate the combined manifest;
and prove all 744 target hours have one complete eligible cycle and all station
weights. Retain immutable GDEX and CAISO documentation. The +10-hour
availability policy remains an explicit conservative assumption until an
authoritative operational historical availability bound is found.
