"""Contracts and joins for a vintage-aware CAISO day-ahead backtest.

This module intentionally does not download "historical" weather.  Strict mode
accepts only a forecast-vintage table carrying an authoritative issue/model
initialization timestamp.  In particular, Open-Meteo Archive API responses are
not accepted because they are realized/reanalysis weather, not old forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

MARKET_TZ = "America/Los_Angeles"
ENTITY = "CA ISO-TAC"
DAM_ITEM = "SYS_FCST_DA_MW"
ACTUAL_ITEM = "SYS_FCST_ACT_MW"
REALIZED_WEATHER_SOURCES = {
    "open-meteo-archive", "open_meteo_archive", "realized", "reanalysis",
    "noaa-arl-ready", "noaa-ready-gfs0p25",
}
OBSOLETE_WEATHER_SOURCES = {"noaa-ncei-gfs-grid004-0p5", "noaa-ncei-gfs-0p25"}
STRICT_WEATHER_SOURCE = "nsf-ncar-gdex-gfs-0p25"


class StrictDataError(ValueError):
    """Raised when input provenance cannot satisfy the frozen-origin contract."""


def _utc(series: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().any():
        raise StrictDataError(f"{name} contains missing or invalid timestamps")
    return parsed


def _require(frame: pd.DataFrame, names: Iterable[str], product: str) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise StrictDataError(f"{product} is missing required columns: {missing}")


def operating_intervals(month: str, cutoff_local: str = "07:00") -> pd.DataFrame:
    """Return real UTC hours in local operating days and one D-1 origin per day."""
    try:
        period = pd.Period(month, freq="M")
        cutoff = time.fromisoformat(cutoff_local)
    except ValueError as exc:
        raise StrictDataError("month must be YYYY-MM and cutoff must be HH:MM[:SS]") from exc
    first = period.start_time.tz_localize(MARKET_TZ)
    after = (period.end_time.normalize() + pd.Timedelta(days=1)).tz_localize(MARKET_TZ)
    utc_hours = pd.date_range(first.tz_convert("UTC"), after.tz_convert("UTC"), freq="h", inclusive="left")
    local = utc_hours.tz_convert(MARKET_TZ)
    operating_day = pd.Series(local.date, dtype="object")
    origin_local = pd.DatetimeIndex(
        [pd.Timestamp.combine(day - pd.Timedelta(days=1), cutoff).tz_localize(MARKET_TZ) for day in operating_day]
    )
    frame = pd.DataFrame({
        "operating_day": operating_day.astype(str),
        "time_utc": utc_hours,
        "forecast_cutoff_utc": origin_local.tz_convert("UTC"),
        "local_target_hour": local.hour,
        "local_target_interval": local.strftime("%Y-%m-%d %H:%M %z"),
        "utc_offset_minutes": [int(x.utcoffset().total_seconds() // 60) for x in local],
    })
    frame["forecast_lead_hours"] = (
        (frame["time_utc"] - frame["forecast_cutoff_utc"]).dt.total_seconds() / 3600
    )
    return frame


def normalize_caiso(frame: pd.DataFrame, data_item: str) -> pd.DataFrame:
    """Validate and normalize an unaggregated OASIS-style CAISO product."""
    _require(frame, ["interval_start_utc", "tac_area_name", "data_item", "mw"], "CAISO")
    out = frame.copy()
    out["interval_start_utc"] = _utc(out["interval_start_utc"], "interval_start_utc")
    if set(out["tac_area_name"].dropna().unique()) != {ENTITY}:
        raise StrictDataError(f"CAISO product must contain only tac_area_name={ENTITY!r}")
    if set(out["data_item"].dropna().unique()) != {data_item}:
        raise StrictDataError(f"CAISO product must contain only data_item={data_item!r}")
    out["mw"] = pd.to_numeric(out["mw"], errors="coerce")
    if out["mw"].isna().any():
        raise StrictDataError("CAISO product contains non-numeric MW")
    if out["interval_start_utc"].duplicated().any():
        raise StrictDataError(f"duplicate {data_item} interval_start_utc keys")
    return out.sort_values("interval_start_utc").reset_index(drop=True)


def select_weather_vintages(weather: pd.DataFrame, origins: pd.DataFrame) -> pd.DataFrame:
    """Select the latest station forecast *available* by each row's origin.

    Model initialization is scientific provenance, not proof of publication.
    Callers must therefore provide a distinct ``available_at_utc`` derived from
    an archive manifest or from a documented conservative lag policy.
    """
    _require(weather, ["station", "model_init_time_utc", "forecast_lead_hours",
                       "available_at_utc", "availability_policy", "valid_time_utc",
                       "source", "model", "source_object", "checksum"], "weather")
    out = weather.copy()
    if out["source"].astype(str).str.lower().isin(REALIZED_WEATHER_SOURCES).any():
        raise StrictDataError("strict mode forbids realized/archive/reanalysis weather fallback")
    if out["source"].astype(str).str.lower().isin(OBSOLETE_WEATHER_SOURCES).any():
        raise StrictDataError("strict mode rejects obsolete NCEI forecast sources")
    if not out["source"].astype(str).str.lower().eq(STRICT_WEATHER_SOURCE).all():
        raise StrictDataError(f"strict weather source must be {STRICT_WEATHER_SOURCE}")
    out["model_init_time_utc"] = _utc(out["model_init_time_utc"], "weather model_init_time_utc")
    out["available_at_utc"] = _utc(out["available_at_utc"], "weather available_at_utc")
    out["valid_time_utc"] = _utc(out["valid_time_utc"], "weather valid_time_utc")
    out["forecast_lead_hours"] = pd.to_numeric(out["forecast_lead_hours"], errors="coerce")
    if out["forecast_lead_hours"].isna().any() or (out["forecast_lead_hours"] < 0).any():
        raise StrictDataError("weather forecast_lead_hours must be non-negative numbers")
    expected_valid = out["model_init_time_utc"] + pd.to_timedelta(out["forecast_lead_hours"], unit="h")
    if not expected_valid.eq(out["valid_time_utc"]).all():
        raise StrictDataError("weather valid time must equal model initialization plus forecast lead")
    if (out["available_at_utc"] < out["model_init_time_utc"]).any():
        raise StrictDataError("weather availability cannot precede model initialization")
    if out["availability_policy"].isna().any() or out["availability_policy"].astype(str).str.strip().eq("").any():
        raise StrictDataError("weather availability_policy must identify timestamp provenance")
    for column in ("source_object", "checksum"):
        if out[column].isna().any() or out[column].astype(str).str.strip().eq("").any():
            raise StrictDataError(f"weather {column} is required for cycle-level provenance")
    keys = origins[["time_utc", "forecast_cutoff_utc"]].rename(columns={"time_utc": "valid_time_utc"})
    candidates = keys.merge(out, on="valid_time_utc", how="left")
    candidates = candidates[candidates["available_at_utc"] <= candidates["forecast_cutoff_utc"]]
    if candidates.empty or candidates["station"].isna().any():
        raise StrictDataError("no eligible archived weather forecast vintages at one or more origins")
    # Select a cycle, not an independently newest row for each station.  The
    # latter can silently mix cycles when one archive object/station is absent.
    expected_stations = set(out["station"].dropna().unique())
    counts = candidates.groupby(["valid_time_utc", "model_init_time_utc"])["station"].nunique()
    complete = counts[counts == len(expected_stations)].reset_index()
    newest = complete.groupby("valid_time_utc", as_index=False)["model_init_time_utc"].max()
    selected = candidates.merge(newest, on=["valid_time_utc", "model_init_time_utc"], how="inner")
    selected = selected.sort_values("available_at_utc").drop_duplicates(
        ["valid_time_utc", "station"], keep="last"
    )
    if selected["valid_time_utc"].nunique() != len(keys):
        raise StrictDataError("no complete single-cycle archived weather forecast at one or more origins")
    return selected.drop(columns="forecast_cutoff_utc").reset_index(drop=True)


def add_previous_week(origins: pd.DataFrame, load_history: pd.DataFrame) -> pd.DataFrame:
    """Attach the exact UTC hour from seven days earlier with availability proof."""
    _require(load_history, ["observation_time_utc", "available_time_utc", "load_mw"], "load history")
    history = load_history.copy()
    history["observation_time_utc"] = _utc(history["observation_time_utc"], "observation_time_utc")
    history["available_time_utc"] = _utc(history["available_time_utc"], "available_time_utc")
    if history["observation_time_utc"].duplicated().any():
        raise StrictDataError("duplicate historical load observation timestamps")
    out = origins.copy()
    out["load_previous_week_source_time_utc"] = out["time_utc"] - pd.Timedelta(days=7)
    history = history.rename(columns={"observation_time_utc": "load_previous_week_source_time_utc",
                                      "available_time_utc": "load_previous_week_available_time_utc",
                                      "load_mw": "load_previous_week"})
    out = out.merge(history, on="load_previous_week_source_time_utc", how="left")
    if out[["load_previous_week", "load_previous_week_available_time_utc"]].isna().any().any():
        raise StrictDataError("missing previous-week load or its availability timestamp")
    if (out["load_previous_week_available_time_utc"] > out["forecast_cutoff_utc"]).any():
        raise StrictDataError("load_previous_week was not available at forecast cutoff")
    return out


@dataclass
class StrictBacktestBuilder:
    month: str
    cutoff_local: str = "07:00"

    def build(self, actuals: pd.DataFrame, dam: pd.DataFrame, weather: pd.DataFrame,
              load_history: pd.DataFrame) -> dict[str, pd.DataFrame]:
        origins = operating_intervals(self.month, self.cutoff_local)
        actual = normalize_caiso(actuals, ACTUAL_ITEM)
        forecast = normalize_caiso(dam, DAM_ITEM)
        selected = select_weather_vintages(weather, origins)
        rows = add_previous_week(origins, load_history)

        actual_join = actual[["interval_start_utc", "mw"]].rename(
            columns={"interval_start_utc": "time_utc", "mw": "actual_load_mw"})
        rows = rows.merge(actual_join, on="time_utc", how="left", validate="one_to_one")
        if rows["actual_load_mw"].isna().any():
            raise StrictDataError("actual SYS_FCST_ACT_MW is incomplete for requested month")

        weather_vars = [c for c in selected.columns if c not in {
            "station", "model_init_time_utc", "forecast_lead_hours", "available_at_utc",
            "availability_policy", "valid_time_utc", "source", "model", "source_object",
            "checksum"}]
        # Preserve station-level weather separately.  Use repository station
        # weights when supplied; an unweighted mean is only valid when no
        # weights are present (principally for backwards-compatible fixtures).
        numeric_weather = [c for c in weather_vars
                           if c != "population_weight" and pd.api.types.is_numeric_dtype(selected[c])]
        if "population_weight" in selected:
            weights = pd.to_numeric(selected["population_weight"], errors="coerce")
            if weights.isna().any() or (weights < 0).any():
                raise StrictDataError("weather population_weight must be non-negative numeric values")
            totals = selected.assign(_weight=weights).groupby("valid_time_utc")["_weight"].transform("sum")
            if (totals <= 0).any():
                raise StrictDataError("weather station weights must have a positive hourly total")
            aggregation_frame = selected.copy()
            aggregation_frame["_normalized_weight"] = weights / totals
            for column in numeric_weather:
                aggregation_frame[column] = aggregation_frame[column] * aggregation_frame["_normalized_weight"]
            agg = {c: "sum" for c in numeric_weather}
            agg["population_weight"] = "sum"
        else:
            aggregation_frame = selected
            agg = {c: "mean" for c in numeric_weather}
        agg.update({"model_init_time_utc": "first", "forecast_lead_hours": "first",
                    "available_at_utc": "max",
                    "source": lambda x: "|".join(sorted(set(x))),
                    "model": lambda x: "|".join(sorted(set(x))),
                    "availability_policy": lambda x: "|".join(sorted(set(x))),
                    "source_object": lambda x: "|".join(sorted(set(x))),
                    "checksum": lambda x: "|".join(sorted(set(x)))})
        weather_hourly = aggregation_frame.groupby("valid_time_utc", as_index=False).agg(agg).rename(
            columns={"valid_time_utc": "time_utc",
                     "model_init_time_utc": "weather_model_init_time_utc",
                     "forecast_lead_hours": "weather_forecast_lead_hours",
                     "available_at_utc": "weather_available_at_utc",
                     "source": "weather_source", "model": "weather_model",
                     "availability_policy": "weather_availability_policy",
                     "source_object": "weather_source_object",
                     "checksum": "weather_checksum"})
        rows = rows.merge(weather_hourly, on="time_utc", how="left", validate="one_to_one")
        if rows["weather_available_at_utc"].isna().any():
            raise StrictDataError("weather vintages are incomplete for requested month")
        if (rows["weather_available_at_utc"] > rows["forecast_cutoff_utc"]).any():
            raise StrictDataError("weather forecast was not available by cutoff")

        local = rows["time_utc"].dt.tz_convert(MARKET_TZ)
        rows.insert(0, "region", "caiso")
        rows.insert(1, "entity", ENTITY)
        rows["day_of_week"] = local.dt.dayofweek
        rows["is_weekend"] = (rows["day_of_week"] >= 5).astype(int)
        rows["hour_sin"] = np.sin(2 * np.pi * rows["local_target_hour"] / 24)
        rows["hour_cos"] = np.cos(2 * np.pi * rows["local_target_hour"] / 24)

        if rows.duplicated("time_utc").any() or rows.groupby("operating_day")["forecast_cutoff_utc"].nunique().ne(1).any():
            raise StrictDataError("origin or target interval uniqueness invariant failed")
        dam_out = forecast.rename(columns={"interval_start_utc": "time_utc", "mw": "caiso_dam_forecast_mw"})
        expected = set(origins["time_utc"])
        present = set(dam_out["time_utc"])
        if expected - present:
            raise StrictDataError("DAM SYS_FCST_DA_MW is incomplete for requested month")
        dam_out = dam_out[dam_out["time_utc"].isin(expected)].reset_index(drop=True)
        return {"actuals": actual, "dam_forecasts": dam_out, "weather_vintages": selected,
                "strict_rows": rows}


def write_products(products: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for name, frame in products.items():
        frame.to_csv(path / f"{name}.csv", index=False)
