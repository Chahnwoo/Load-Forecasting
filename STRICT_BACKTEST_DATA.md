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

Use **NOAA NCEI's archived 0.25-degree GFS forecast products**. NCEI's historical
GFS inventory contains complete, cycle-and-forecast-hour-addressable products
for December 2025, rather than only a reconstruction of near-analysis hours.
An acquisition must request/list the December objects before building data and
pin the returned NCEI object/file identifier, byte size and checksum in a local
manifest. This object inventory is the final verification that every required
cycle and lead actually exists; missing objects fail the build.

Two tempting sources are explicitly unsuitable:

* The operational `noaa-gfs-bdp-pds`/`noaa-gfs-pds` AWS store is rolling
  (approximately four weeks), so it is not a durable December 2025 archive.
* NOAA ARL/READY's historical `gfs0p25` daily files are a **pseudo-analysis**:
  READY constructs them from successive runs' +0 and +3 hour forecasts, with
  newer inputs overwriting parts of the archive. They do not preserve complete,
  independently identifiable forecast cycles and must never be supplied to the
  strict builder as forecast vintages.

The three weather times have deliberately different meanings:

* `model_init_time_utc` is the GFS cycle (`00`, `06`, `12`, or `18 UTC`).
* `forecast_lead_hours` is the forecast-hour encoded by the NCEI product.
* `valid_time_utc` is initialization plus `forecast_lead_hours`.
* `available_at_utc` is when that forecast file is conservatively considered
  usable. The NCEI historical product inventory does not expose an authoritative
  historical publication timestamp for every archived object.
* `source_object` is the exact NCEI object/file and `checksum` pins its bytes.

READY documents a typical 2--4 hour delay for its *current* forecast files, but
that statement neither timestamps NCEI's historical objects nor establishes a
hard upper bound for them. It therefore does not justify weakening strict
availability. Consequently the fallback acquisition policy remains
`available_at_utc = model_init_time_utc + 10 hours`, record policy identifier
`gfs_init_plus_10h_v1`, and choose the latest cycle whose derived availability
is no later than the cutoff.  Ten hours is intentionally longer than a normal
GFS product delay and, with the 07:00 Pacific cutoff (15:00 UTC in December),
normally excludes the 06 UTC cycle in favor of 00 UTC.  This is a conservative
policy, not a claim that the file was published at exactly that instant.  A
future NCEI manifest may replace the derived time with an authoritative object
publication timestamp, but must retain the policy/source column.  Repository
retrieval time is never substituted for historical availability.

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

## Products and provenance

Run (once the four source extracts exist):

```bash
python build_strict_backtest_dataset.py \
  --month 2025-12 --entity caiso --forecast-cutoff-local 07:00 \
  --actuals-csv ACTUAL.csv --dam-csv DAM.csv \
  --weather-vintages-csv GFS_VINTAGES.csv \
  --load-history-csv LOAD_HISTORY.csv --output-dir data/strict_backtest/2025-12
```

The builder writes `actuals.csv`, `dam_forecasts.csv`,
`weather_vintages.csv`, and `strict_rows.csv`.  CAISO DAM stays out of model
features and exists only in its evaluation product.

| Field/product | Exact source and meaning |
|---|---|
| Actual | CAISO OASIS row with `TAC_AREA_NAME = CA ISO-TAC` and `DATA_ITEM = SYS_FCST_ACT_MW`; exact interval start converted to aware UTC. |
| CAISO benchmark | CAISO OASIS `CA ISO-TAC`, `SYS_FCST_DA_MW`; stored separately. The input should retain its raw OASIS download manifest. |
| Weather | NOAA NCEI 0.25° GFS full-cycle forecast product. Initialization, lead, validity and availability are distinct; exact object and checksum are retained. If no historical publication timestamp exists, availability uses `gfs_init_plus_10h_v1`. READY pseudo-analysis files are forbidden. |
| Previous-week load | The same CAISO actual system series at exactly `time_utc - 7 days`; both observation and `available_time_utc` are retained. Availability must be supplied by the source manifest and precede the cutoff. |
| Calendar | Deterministically derived from UTC converted to `America/Los_Angeles`; never downloaded. |

Input CAISO CSV columns are `interval_start_utc,tac_area_name,data_item,mw`.
Weather requires `station,model_init_time_utc,forecast_lead_hours,available_at_utc,availability_policy,valid_time_utc,source,model,source_object,checksum`
plus numeric weather variables. The builder proves
`valid_time_utc = model_init_time_utc + forecast_lead_hours`, rejects
availability before initialization, and selects one newest eligible cycle per
station/valid time using availability—not initialization. Thus each selected
value remains traceable to one exact NCEI object and checksum. Load history requires
`observation_time_utc,available_time_utc,load_mw`.

## Satisfaction matrix and limitations

Fully implemented and tested: exact UTC joins; `CA ISO-TAC` and exact data-item
enforcement; unique actual/DAM keys; latest eligible weather vintage selection;
weather and load availability assertions; one origin per operating day; unique
targets; explicit DST cardinality; separate DAM output; deterministic calendar
features; and fail-closed rejection of realized weather.

Still blocked: downloading and checksum-pinning the December 2025 NCEI objects;
an authoritative historical timestamp for when each CAISO actual became
available; and confirmation of CAISO's internal forecast cutoff.  OASIS may
revise records, so raw response archives and retrieval timestamps should be
preserved.  No benchmark metrics are implemented or run by this path.

## Source documentation reviewed

The relevant references are CAISO Day-Ahead Market procedure 1210 v19.8
(effective 2025-10-01), CAISO OASIS Interface Specification/report definition
for `SLD_FCST`, Open-Meteo Archive and Historical Forecast API documentation,
NOAA ARL READY's `gfs0p25` archive README, and NCEI historical GFS product
documentation/inventory. Automated network access was blocked by the implementation
environment's proxy, so the version/effective-date and archive inventory must be
captured with their source documents in the acquisition manifest before the
December artifact is certified. This limitation is why the implementation uses
earlier fixed times and explicit lag policy rather than inferred timestamps.
