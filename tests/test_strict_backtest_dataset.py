import unittest

import pandas as pd

from src.backtesting.strict_dataset import (
    ACTUAL_ITEM,
    DAM_ITEM,
    ENTITY,
    StrictBacktestBuilder,
    StrictDataError,
    normalize_caiso,
    operating_intervals,
    select_weather_vintages,
)


class StrictBacktestTests(unittest.TestCase):
    def test_dst_operating_days_are_explicit(self):
        spring = operating_intervals("2025-03")
        fall = operating_intervals("2025-11")
        december = operating_intervals("2025-12").head(1)
        self.assertEqual(23, (spring.operating_day == "2025-03-09").sum())
        self.assertEqual(25, (fall.operating_day == "2025-11-02").sum())
        self.assertTrue(spring.groupby("operating_day").forecast_cutoff_utc.nunique().eq(1).all())
        self.assertFalse(spring.time_utc.duplicated().any())
        self.assertEqual(15, december.forecast_cutoff_utc.iloc[0].hour)  # 07:00 PST

    def test_latest_legal_weather_vintage_is_selected(self):
        origins = operating_intervals("2025-12").head(1)
        valid = origins.time_utc.iloc[0]
        cutoff = origins.forecast_cutoff_utc.iloc[0]
        weather = pd.DataFrame({
            "station": ["LAX"] * 3,
            "model_init_time_utc": [cutoff - pd.Timedelta(hours=8), cutoff - pd.Timedelta(hours=2),
                                    cutoff - pd.Timedelta(hours=1)],
            "available_at_utc": [cutoff - pd.Timedelta(hours=6), cutoff, cutoff + pd.Timedelta(hours=1)],
            "availability_policy": ["gfs_init_plus_10h_v1"] * 3,
            "valid_time_utc": [valid] * 3,
            "source": ["noaa-ncei-gfs-0p25"] * 3,
            "model": ["gfs"] * 3,
            "source_object": ["ncei/gfs/cycle-a/f027.grb2", "ncei/gfs/cycle-b/f021.grb2",
                              "ncei/gfs/cycle-c/f020.grb2"],
            "checksum": ["sha256:a", "sha256:b", "sha256:c"],
            "temperature_2m": [1.0, 2.0, 99.0],
        })
        weather["forecast_lead_hours"] = (
            (weather["valid_time_utc"] - weather["model_init_time_utc"]).dt.total_seconds() / 3600
        )
        selected = select_weather_vintages(weather, origins)
        self.assertEqual(1, len(selected))
        self.assertEqual(2.0, selected.temperature_2m.iloc[0])
        self.assertLessEqual(selected.available_at_utc.iloc[0], cutoff)
        self.assertEqual("ncei/gfs/cycle-b/f021.grb2", selected.source_object.iloc[0])
        self.assertEqual("sha256:b", selected.checksum.iloc[0])

    def test_realized_weather_is_never_fallback(self):
        origins = operating_intervals("2025-12").head(1)
        weather = pd.DataFrame({"station": ["LAX"],
                                "model_init_time_utc": [origins.forecast_cutoff_utc.iloc[0] - pd.Timedelta(hours=1)],
                                "available_at_utc": [origins.forecast_cutoff_utc.iloc[0]],
                                "availability_policy": ["gfs_init_plus_10h_v1"],
                                "valid_time_utc": [origins.time_utc.iloc[0]],
                                "forecast_lead_hours": [18.0], "source_object": ["archive.json"],
                                "checksum": ["sha256:a"],
                                "source": ["open-meteo-archive"], "model": ["best_match"]})
        with self.assertRaisesRegex(StrictDataError, "forbids realized"):
            select_weather_vintages(weather, origins)
        ready = weather.assign(source="noaa-arl-ready")
        with self.assertRaisesRegex(StrictDataError, "forbids realized"):
            select_weather_vintages(ready, origins)

    def test_caiso_entity_item_and_unique_key_contract(self):
        base = pd.DataFrame({"interval_start_utc": ["2025-12-01T08:00:00Z"],
                             "tac_area_name": [ENTITY], "data_item": [DAM_ITEM], "mw": [20000]})
        self.assertEqual(DAM_ITEM, normalize_caiso(base, DAM_ITEM).data_item.iloc[0])
        with self.assertRaises(StrictDataError):
            normalize_caiso(pd.concat([base, base]), DAM_ITEM)
        wrong = base.assign(data_item=ACTUAL_ITEM)
        with self.assertRaises(StrictDataError):
            normalize_caiso(wrong, DAM_ITEM)

    def test_end_to_end_asserts_load_availability_and_keeps_dam_separate(self):
        origins = operating_intervals("2025-12")
        times = origins.time_utc
        actual = pd.DataFrame({"interval_start_utc": times, "tac_area_name": ENTITY,
                               "data_item": ACTUAL_ITEM, "mw": 20000.0})
        dam = actual.assign(data_item=DAM_ITEM, mw=19900.0)
        weather = pd.DataFrame({"station": "LAX", "valid_time_utc": times,
                                "model_init_time_utc": origins.forecast_cutoff_utc - pd.Timedelta(hours=8),
                                "available_at_utc": origins.forecast_cutoff_utc - pd.Timedelta(hours=1),
                                "availability_policy": "gfs_init_plus_10h_v1",
                                "source": "noaa-ncei-gfs-0p25", "model": "gfs-0p25",
                                "source_object": "ncei/gfs/full-cycle/f024.grb2",
                                "checksum": "sha256:fixture",
                                "temperature_2m": 18.0})
        weather["forecast_lead_hours"] = (
            (weather["valid_time_utc"] - weather["model_init_time_utc"]).dt.total_seconds() / 3600
        )
        load = pd.DataFrame({"observation_time_utc": times - pd.Timedelta(days=7),
                             "available_time_utc": origins.forecast_cutoff_utc - pd.Timedelta(hours=1),
                             "load_mw": 19000.0})
        products = StrictBacktestBuilder("2025-12").build(actual, dam, weather, load)
        rows = products["strict_rows"]
        self.assertNotIn("caiso_dam_forecast_mw", rows.columns)
        self.assertTrue((rows.weather_available_at_utc <= rows.forecast_cutoff_utc).all())
        self.assertTrue((rows.load_previous_week_available_time_utc <= rows.forecast_cutoff_utc).all())
        self.assertEqual({ENTITY}, set(rows.entity))

        late = load.assign(available_time_utc=origins.forecast_cutoff_utc + pd.Timedelta(minutes=1))
        with self.assertRaisesRegex(StrictDataError, "not available"):
            StrictBacktestBuilder("2025-12").build(actual, dam, weather, late)

    def test_model_initialization_is_not_availability(self):
        origins = operating_intervals("2025-12").head(1)
        cutoff = origins.forecast_cutoff_utc.iloc[0]
        weather = pd.DataFrame({"station": ["LAX"], "valid_time_utc": [origins.time_utc.iloc[0]],
                                "model_init_time_utc": [cutoff - pd.Timedelta(hours=6)],
                                "available_at_utc": [cutoff + pd.Timedelta(minutes=1)],
                                "availability_policy": ["gfs_init_plus_10h_v1"],
                                "source": ["noaa-ncei-gfs-0p25"], "model": ["gfs-0p25"],
                                "forecast_lead_hours": [23.0],
                                "source_object": ["ncei/gfs/cycle/f023.grb2"],
                                "checksum": ["sha256:a"]})
        with self.assertRaisesRegex(StrictDataError, "no eligible"):
            select_weather_vintages(weather, origins)
        impossible = weather.assign(available_at_utc=cutoff - pd.Timedelta(hours=7))
        with self.assertRaisesRegex(StrictDataError, "cannot precede"):
            select_weather_vintages(impossible, origins)


if __name__ == "__main__":
    unittest.main()
