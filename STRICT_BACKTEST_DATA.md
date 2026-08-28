# Strict CAISO day-ahead backtest data contract

This document is the implementation-level companion to the canonical [README](README.md). It describes the provenance and fail-closed rules behind the validated October–November 2025 training and December 2025 held-out evaluation.

## Frozen origin and legal information

Each Pacific operating day has one origin at D-1 07:00 `America/Los_Angeles`. A value is eligible only when its explicit availability time is no later than that origin. GFS availability uses the repository's conservative `gfs_init_plus_10h_v1` policy; it is not claimed as an authoritative publication timestamp or as CAISO's internal cutoff.

The builder rejects realized/reanalysis weather, incomplete station cycles, weather after cutoff, previous-week load after cutoff, missing intervals, duplicate UTC keys, and mixed cutoffs within an operating day. It retains CAISO DAM only as a paired benchmark and never places it in the model feature allowlist.

## CAISO products

OASIS acquisitions request `SLD_FCST` separately for `DAM` and `ACTUAL`. Normalization requires:

| Product | TAC area | Data item | Market run |
|---|---|---|---|
| Day-ahead forecast | `CA ISO-TAC` | `SYS_FCST_DA_MW` | `DAM` |
| Actual load | `CA ISO-TAC` | `SYS_FCST_ACT_MW` | `ACTUAL` |

Raw ZIP response bytes and `.zip.provenance.json` sidecars are retained. Checks validate SHA-256, ZIP readability, a CSV result rather than an OASIS error member, identifiers, and exact target keys. UTC interval starts are authoritative join keys; Pacific conversion defines operating-day membership and DST hour counts.

## GDEX GFS source and subsets

Strict weather uses NSF NCAR GDEX dataset `d084001`, source `nsf-ncar-gdex-gfs-0p25`, model `gfs-0p25`. Canonical backing names are `gfs.0p25.YYYYMMDDHH.fFFF.grib2`; name parsing validates initialization cycle, lead, and valid time.

Fresh acquisition discovers variables from each backing object's **OPeNDAP DAS metadata**. It then uses NCSS Grid with `accept=netCDF` to obtain only a small bounding box around the CAISO stations, not the roughly 500 MB global GRIB object. Responses must have a classic or 64-bit-offset **NetCDF-3** signature. Local reads use `xarray.open_dataset(..., engine="scipy")`.

Required fields are 2 m temperature and relative humidity, 10 m U/V wind, total cloud cover, total precipitation, and downward shortwave radiation. Variable names and GRIB semantics are validated from attributes rather than guessed. CF time bounds define precipitation accumulation and shortwave averaging periods; instantaneous variables interpolate only between adjacent products from the same cycle.

## Provenance and resume behavior

Every subset's exact bytes are retained as `.nc`, alongside `.nc.provenance.json` containing the deterministic NCSS URL, retrieval time, and SHA-256. The acquisition manifest additionally records the backing object/fileServer URL, initialization, lead, valid time, request parameters, discovered variables, availability policy/time, and source-document references.

For an existing pair, the script verifies sidecar structure, URL-to-object identity, checksum, NetCDF contents, locally discovered variables, and request-variable agreement. Valid pairs require **zero metadata or network calls** during resume. Missing objects are downloaded; incomplete pairs or mismatches fail closed. From valid cached subsets the script can regenerate the acquisition manifest and `processed/weather_vintages.csv` offline.

## Station and temporal semantics

The five repository California stations retain population weights through vintage selection. Selection chooses the newest eligible initialization complete across all stations for a target hour; it cannot splice stations from different cycles. Weather is aggregated only afterward.

Exact per-product statistical periods are expanded to hourly values using their metadata. Precipitation accumulation is divided across its represented hours; an average shortwave value is held across its interval. The strict selected station table keeps source objects and checksums for every contributing row.

## Certified reference state

| Month | Strict hours | Selected station-weather rows | Stations | Role |
|---|---:|---:|---:|---|
| October 2025 | 744 | 3,720 | 5 | training / chronological CV |
| November 2025 | 721 | 3,605 | 5 | training / chronological CV |
| December 2025 | 744 | 3,720 | 5 | held-out test only |

November includes the 25-hour Nov. 2 Pacific fall-back day. Training therefore contains 1,465 hours. December is not used for preprocessing, hyperparameter choice, calibration fitting, or CV.

Completed December raw-data certification validated 310/310 NetCDF subset/sidecar pairs offline and 78/78 OASIS ZIP/sidecar acquisitions for checksum, ZIP, and identifiers. It also confirmed 744 DAM targets, 744 ACTUAL targets, and 744 previous-week history rows. October and November raw counts must be derived from their own caches/manifests rather than copied from December.

Run the permanent read-only checker rather than an ad-hoc heredoc:

```bash
python -m scripts.certify_strict_backtest --months 2025-10 2025-11 2025-12 \
  --validate-raw-gdex --validate-raw-caiso
```

See the README for complete acquisition/build/evaluation commands and limitations.
