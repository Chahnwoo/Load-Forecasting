# Strict CAISO day-ahead backtest data contract

## Status (December 2025)

The repository has a strict, fail-closed assembly path, but it **cannot yet
produce a trustworthy December 2025 dataset from the repository's checked-in
sources**.  Existing
`data/cache/open_meteo_json` files and the collection pipeline use Open-Meteo's
Archive API.  Those values describe realized/reanalysis weather and have no
forecast issue/model-initialization time.  They are not accepted by the strict
builder and will never be relabelled as day-ahead forecasts.

Open-Meteo also offers a Historical Forecast API.  Its convenient time series
does not expose a selectable issue/model-run timestamp for every returned
value, so we cannot prove the required `issue_time_utc <= forecast_cutoff_utc`
from that response alone.  It is therefore not the default strict source.

### Selected historical weather archive

Use the officially archived **NOAA NCEI Global Forecast System 0.5 Degree,
Grid 004, DSI 6182** forecast product. Its archive is cycle-and-lead
addressable (`gfs_4_YYYYMMDD_HH00_FFF.grb2`), with 00/06/12/18 UTC cycles,
a 0.5-degree grid, three-hour forecast products, and forecast hours +000
through +192. The strict source identifier is
`noaa-ncei-gfs-grid004-0p5`; the model identifier is `gfs-grid004-0p5`.

The previously proposed permanent NCEI archive for 0.25-degree PGRB2 was
declined. Operational 0.25-degree cloud objects are only a short rolling
archive and no longer contain December 2025. They must not be used. Likewise,
ARL/READY `gfs0p25` daily files are pseudo-analyses assembled from successive
near-analysis forecasts and do not preserve identifiable full cycles.

The Grid 004 acquisition first computes the cycle allowed by the frozen origin
and `gfs_init_plus_10h_v1`, then requests only three-hour objects needed to
bracket the operating day's target hours. It parses the filename back into
initialization, lead, and valid time and records the exact NCEI URL, archive
filename, retrieval time, and SHA-256 checksum. A missing object or malformed
filename is fatal; analysis files are never substituted.

The weather timestamps have deliberately different meanings:

* `model_init_time_utc` is the GFS cycle.
* `forecast_lead_hours` is encoded by the archived Grid 004 filename.
* `valid_time_utc` equals initialization plus forecast lead.
* `available_at_utc` remains initialization plus ten hours under
  `gfs_init_plus_10h_v1`; retrieval time is never backdated into availability.
* `source_object` and `checksum` identify the exact bytes used.

The historical inventory does not provide authoritative publication time for
every object. The conservative +10 hour policy therefore remains unchanged.
At a December 07:00 PST (15:00 UTC) D-1 origin it normally selects the prior
00 UTC cycle, not the 06 UTC cycle. Selection always tests
`available_at_utc <= forecast_cutoff_utc` and requires one complete cycle
across all stations.

### Three-hour to hourly weather policy

Station points use the repository's CAISO locations and population weights.
GRIB records are inspected with `wgrib2`; the extraction fails unless exactly
one expected field exists and APCP/DSWRF inventory metadata explicitly reports
a three-hour statistical interval.

* `temperature_2m` (TMP at 2 m), `relative_humidity_2m` (RH at 2 m),
  `cloud_cover` (TCDC entire atmosphere), and 10 m U/V wind components are
  instantaneous. Wind speed is derived as `sqrt(U^2 + V^2)`. These values are
  linearly time-interpolated only between adjacent three-hour values from the
  same initialization cycle.
* `precipitation` (APCP) is a three-hour accumulation for `(T-3h,T]`. It is
  divided evenly among the three ending hourly intervals. It is never linearly
  interpolated.
* `shortwave_radiation` (DSWRF) is a three-hour mean flux. That mean is held
  constant over its three covered hourly intervals. It is never treated as an
  instantaneous observation.
* `cdd_65f` and `hdd_65f` are computed only after hourly temperature exists.
  Apparent temperature uses NOAA heat-index and wind-chill regressions in
  their defined warm/humid and cool/windy ranges, and otherwise equals air
  temperature.

Each hourly station row retains the complete set of source objects and
checksums from its one contributing initialization cycle. Station provenance
is preserved in `weather_vintages.csv` before weighted regional aggregation.

## Forecast origin and time semantics

Each sample is an actual UTC hour belonging to an operating date in
`America/Los_Angeles`.  `time_utc` is the canonical join key.  The builder also
stores the local interval label (including numeric offset), local hour, offset
minutes, operating day, cutoff, and fractional lead hours.  Thus spring-forward
days have 23 rows and fall-back days have 25; no 24-row assumption is made.

### December 2025 CAISO cutoff

The procedure applicable to the month is CAISO Day-Ahead Market procedure
**1210, version 19.8, effective 2025-10-01**.  It describes the CAISO demand
forecast as a morning publication before the Day-Ahead Market runs.  OASIS
documentation identifies `SYS_FCST_DA_MW` as that DAM system forecast.  Neither
the procedure nor the historical OASIS result establishes a row-level immutable
publication timestamp for the December 2025 values. It therefore does not
support treating 09:00 as a conservative cutoff.

The default is therefore **07:00 Pacific prevailing time on D-1**, preceding
CAISO's documented morning/pre-market publication window. The existing
repository OASIS collector queries `SLD_FCST` with `market_run_id=DAM`, but does
not retain publication/retrieval time and does not establish CAISO's internal
forecast information cutoff.  Thus 07:00 is a deliberately conservative and
reproducible comparison origin, not a claim about CAISO's internal model cutoff
or exact publication instant.  December uses PST, so this is 15:00 UTC.  The CLI
exposes `--forecast-cutoff-local` so authoritative evidence can replace it.

## Local acquisition and build workflow

Prerequisites are Python dependencies from `requirements.txt`, internet access
to CAISO and NCEI, sufficient disk space for Grid 004 GRIB2 objects, and
`wgrib2` on `PATH`. Nothing downloaded is committed to this repository.

Run all acquisition and assembly stages with:

```bash
python scripts/acquire_strict_backtest_inputs.py --month 2025-12 \
  --caiso-source-document /immutable/docs/caiso-1210-v19.8.pdf \
  --ncei-source-document /immutable/docs/ncei-dsi-6182.pdf
```

Or run the auditable stages separately:

```bash
python scripts/acquire_december_2025_caiso.py --month 2025-12 \
  --source-document /immutable/docs/caiso-1210-v19.8.pdf
python scripts/acquire_ncei_gfs_grid004.py --month 2025-12 \
  --source-document /immutable/docs/ncei-dsi-6182.pdf
python build_strict_backtest_dataset.py --month 2025-12 \
  --actuals-csv data/backtesting/2025-12/processed/actuals.csv \
  --dam-csv data/backtesting/2025-12/processed/caiso_dam.csv \
  --weather-vintages-csv data/backtesting/2025-12/processed/weather_vintages.csv \
  --load-history-csv data/backtesting/2025-12/processed/load_history.csv \
  --output-dir data/backtesting/2025-12/processed/strict
```

Expected layout:

```text
data/backtesting/2025-12/
  raw/caiso/                 # unchanged OASIS ZIP response bodies
  raw/gfs/                   # unchanged NCEI Grid 004 GRIB2 objects
  processed/actuals.csv
  processed/caiso_dam.csv
  processed/load_history.csv
  processed/weather_vintages.csv
  processed/strict/{actuals,dam_forecasts,weather_vintages,strict_rows}.csv
  acquisition_manifest.caiso.json
  acquisition_manifest.gfs.json
  acquisition_manifest.json       # combined manifest written by wrapper
```

The CAISO script retrieves `SLD_FCST` daily, retains response ZIP bytes, rejects
HTTP/OASIS error documents, filters only `CA ISO-TAC / SYS_FCST_ACT_MW` and
`CA ISO-TAC / SYS_FCST_DA_MW / DAM`, and fails on missing or duplicate target
hours. It also fetches the exact previous-week actual hours and gives them the
conservative `caiso_actual_plus_24h_v1` availability bound. It never reads the
project's generic `load_mw` dataset or constructs regional sums.

Verify success by checking that both manifests validate against
`schemas/acquisition_manifest.schema.json`, every manifest checksum matches
its raw file, all raw files are nonempty, December has 744 unique actual, DAM,
and strict target timestamps, all weather rows use source
`noaa-ncei-gfs-grid004-0p5`, and every selected availability timestamp is no
later than its frozen origin. The strict builder keeps CAISO DAM outside model
features.

| Product | Required provenance |
|---|---|
| CAISO actual/DAM | Endpoint, encoded request parameters/URL, raw response filename, retrieval timestamp, SHA-256, raw identifiers, UTC interval. |
| Grid 004 weather | DSI 6182 object URL/name, initialization, lead, validity, +10h policy and availability, retrieval mechanism/time, SHA-256, station coordinates and weight. |
| Previous-week load | Exact CAISO actual observation time, explicit conservative availability time/policy, and unchanged MW value. |

## Satisfaction matrix and limitations

Fully implemented and tested: exact UTC joins; `CA ISO-TAC` and exact data-item
enforcement; unique actual/DAM keys; latest eligible weather vintage selection;
weather and load availability assertions; one origin per operating day; unique
targets; explicit DST cardinality; separate DAM output; deterministic calendar
features; and fail-closed rejection of realized weather.

Remaining certification conditions: run the workflow outside Codex; retain
and checksum-pin every December source response; validate both manifests;
confirm 744 unique target intervals in all final products; and retain the
applicable source documentation with the artifact. An authoritative historical
timestamp for each CAISO actual and confirmation of CAISO's internal forecast
cutoff remain unavailable, so the documented conservative bounds must not be
weakened. OASIS may revise records, making raw response retention mandatory.
No benchmark metrics are implemented or run by this path.

## Source documentation reviewed

The relevant references are CAISO Day-Ahead Market procedure 1210 v19.8
(effective 2025-10-01), CAISO OASIS Interface Specification/report definition
for `SLD_FCST`, Open-Meteo Archive and Historical Forecast API documentation,
NOAA ARL READY's `gfs0p25` archive README, the decision record declining the
permanent 0.25-degree PGRB2 archive, and NCEI DSI 6182/Grid 004 documentation
and inventory. The local run must save immutable copies or references to these
documents beside its manifests before the December artifact is certified.
