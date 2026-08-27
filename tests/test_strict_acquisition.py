import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.backtesting.acquisition import (GDEX_SOURCE, caiso_request,
                                         gdex_object_url, gdex_ncss_url, hourly_from_gdex,
                                         parse_gdex_name, sha256_file, write_manifest)
from scripts.acquire_december_2025_caiso import normalize, raw_identifiers
from src.backtesting.strict_dataset import ACTUAL_ITEM, DAM_ITEM


class StrictAcquisitionTests(unittest.TestCase):
    def test_caiso_request_is_exact_and_encoded(self):
        url, params = caiso_request(pd.Timestamp("2025-12-01T08:00Z"),
                                    pd.Timestamp("2025-12-02T08:00Z"),
                                    market_run_id="DAM")
        self.assertEqual("SLD_FCST", params["queryname"])
        self.assertEqual("DAM", params["market_run_id"])
        self.assertEqual("20251201T08:00-0000", params["startdatetime"])
        self.assertIn("oasis.caiso.com/oasisapi/SingleZip?", url)
        self.assertNotIn("load_mw", url)
        actual_url, actual_params = caiso_request(
            pd.Timestamp("2025-12-01T08:00Z"), pd.Timestamp("2025-12-02T08:00Z"),
            market_run_id="ACTUAL")
        self.assertEqual("ACTUAL", actual_params["market_run_id"])
        self.assertIn("market_run_id=ACTUAL", actual_url)
        with self.assertRaisesRegex(ValueError, "ACTUAL, DAM"):
            caiso_request(pd.Timestamp("2025-12-01T08:00Z"),
                          pd.Timestamp("2025-12-02T08:00Z"), market_run_id="RTM")

    def test_gdex_object_parsing(self):
        name = "gfs.0p25.2025120100.f057.grib2"
        parsed = parse_gdex_name(name)
        self.assertEqual(57, parsed["forecast_lead_hours"])
        self.assertEqual(pd.Timestamp("2025-12-03T09:00Z"), parsed["valid_time_utc"])
        self.assertTrue(gdex_object_url(name).endswith("/2025/20251201/" + name))
        with self.assertRaises(ValueError):
            parse_gdex_name("gfs.0p25.2025120101.f058.grib2")
        with self.assertRaises(ValueError):
            parse_gdex_name("gfs.0p50.2025120100.f057.grib2")

    def test_ncss_request_is_canonical_and_subsetted(self):
        name = "gfs.0p25.2025113000.f030.grib2"
        url, params = gdex_ncss_url(name, ["Temperature_height_above_ground"],
                                    north=42, south=32, east=-114, west=-125)
        self.assertIn("/ncss/grid/files/g/d084001/2025/20251130/" + name, url)
        self.assertEqual("netcdf4", params["accept"])
        self.assertEqual("Temperature_height_above_ground", params["var"])
        self.assertNotIn("fileServer", url)

    def test_real_oasis_contract_filtering_and_provenance(self):
        fixture = pd.DataFrame({"INTERVALSTARTTIME_GMT": ["2025-12-01T08:00Z"] * 6,
                                "INTERVALENDTIME_GMT": ["2025-12-01T09:00Z"] * 6,
                                "TAC_AREA_NAME": ["CA ISO-TAC", "PGE-TAC", "CA ISO-TAC",
                                                  "CA ISO-TAC", "CA ISO-TAC", "CA ISO-TAC"],
                                "XML_DATA_ITEM": [ACTUAL_ITEM, ACTUAL_ITEM, DAM_ITEM,
                                                  ACTUAL_ITEM, DAM_ITEM, DAM_ITEM],
                                "MARKET_RUN_ID": ["ACTUAL", "ACTUAL", "DAM",
                                                  "DAM", "ACTUAL", "RTM"],
                                "LABEL": ["fixture"] * 6, "MW": [1, 2, 3, 4, 5, 6]})
        actual = normalize(fixture, ACTUAL_ITEM)
        dam = normalize(fixture, DAM_ITEM)
        self.assertEqual([1], actual.mw.tolist())
        self.assertEqual([3], dam.mw.tolist())
        self.assertEqual({"CA ISO-TAC"}, set(actual.tac_area_name))
        identifiers = raw_identifiers(fixture)
        self.assertIn({"TAC_AREA_NAME": "CA ISO-TAC", "XML_DATA_ITEM": ACTUAL_ITEM,
                       "MARKET_RUN_ID": "ACTUAL"}, identifiers)
        self.assertTrue(all(set(value) == {"TAC_AREA_NAME", "XML_DATA_ITEM", "MARKET_RUN_ID"}
                            for value in identifiers))

    def test_historical_data_item_fallback(self):
        fixture = pd.DataFrame({"INTERVALSTARTTIME_GMT": ["2025-12-01T08:00Z"],
                                "TAC_AREA_NAME": ["CA ISO-TAC"], "DATA_ITEM": [DAM_ITEM],
                                "MARKET_RUN_ID": ["DAM"], "MW": [3]})
        self.assertEqual([3], normalize(fixture, DAM_ITEM).mw.tolist())
        self.assertEqual(DAM_ITEM, raw_identifiers(fixture)[0]["XML_DATA_ITEM"])

    def test_checksum_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.nc"
            source.write_bytes(b"ncss fixture\n")
            self.assertEqual("sha256:96c77686dc6b854b85a5b2cb8248f5f6325a66c92ff1f7a4f44582c471bdeda0",
                             sha256_file(source))
            manifest = Path(directory) / "manifest.json"
            gdex_provenance = {"dataset": "d084001", "dataset_reference": "doi:fixture",
                               "backing_source_object": "fixture.grib2", "backing_fileserver_url": "https://fixture/file",
                               "model_init_time_utc": "2025-11-30T00:00:00Z", "forecast_lead_hours": 30,
                               "valid_time_utc": "2025-12-01T06:00:00Z", "ncss_request_url": "https://fixture/ncss",
                               "request_parameters": {"var": "fixture"}, "raw_subset_local_filename": str(source),
                               "availability_policy": "gfs_init_plus_10h_v1", "available_at_utc": "2025-11-30T10:00:00Z"}
            write_manifest(manifest, [{"source": GDEX_SOURCE, "source_object": "fixture", **gdex_provenance,
                                       "checksum": sha256_file(source),
                                       "retrieved_at_utc": "2026-01-01T00:00:00Z"}], month="2025-12",
                           source_documents=["/immutable/gdex-d084001.html"])
            payload = json.loads(manifest.read_text())
            self.assertEqual("strict-acquisition-manifest-v1", payload["schema_version"])
            self.assertEqual(GDEX_SOURCE, payload["acquisitions"][0]["source"])
            self.assertEqual(["/immutable/gdex-d084001.html"], payload["source_documents"])
            with self.assertRaisesRegex(ValueError, "missing provenance"):
                write_manifest(manifest, [{"source": GDEX_SOURCE,
                                           "checksum": sha256_file(source)}], month="2025-12",
                               source_documents=["/immutable/gdex-d084001.html"])

    def grid_frame(self):
        times = pd.date_range("2025-12-01T06:00Z", periods=3, freq="3h")
        return pd.DataFrame({"station": "LAX", "model_init_time_utc": pd.Timestamp("2025-12-01T00:00Z"),
                             "valid_time_utc": times, "temperature_2m": [10., 13., 16.],
                             "relative_humidity_2m": [40., 70., 100.], "cloud_cover": [0., 30., 60.],
                             "u_wind_10m": [1., 4., 7.], "v_wind_10m": [0., 0., 0.], "precipitation": [3., 6., 9.],
                             "precipitation_period_hours": 3, "shortwave_radiation": [100., 400., 700.],
                             "shortwave_period_hours": 3})

    def test_three_hour_instantaneous_and_interval_semantics(self):
        hourly = hourly_from_gdex(self.grid_frame()).set_index("valid_time_utc")
        self.assertEqual(14., hourly.loc[pd.Timestamp("2025-12-01T10:00Z"), "temperature_2m"])
        # The 09Z APCP amount is distributed over 07, 08, 09Z, not interpolated.
        self.assertEqual(2., hourly.loc[pd.Timestamp("2025-12-01T08:00Z"), "precipitation"])
        self.assertEqual(400., hourly.loc[pd.Timestamp("2025-12-01T08:00Z"), "shortwave_radiation"])
        # The first interval covers 04, 05, 06Z, but only its in-range 06Z hour is
        # assigned. Its amount is still divided by the original three-hour period.
        self.assertEqual(1., hourly.loc[pd.Timestamp("2025-12-01T06:00Z"), "precipitation"])
        self.assertEqual(100., hourly.loc[pd.Timestamp("2025-12-01T06:00Z"), "shortwave_radiation"])
        self.assertEqual(pd.Timestamp("2025-12-01T06:00Z"), hourly.index.min())
        self.assertEqual(pd.Timestamp("2025-12-01T12:00Z"), hourly.index.max())
        self.assertNotIn(pd.Timestamp("2025-12-01T05:00Z"), hourly.index)

    def test_incomplete_cycle_and_bad_accumulation_fail(self):
        incomplete = self.grid_frame().drop(index=1)
        with self.assertRaisesRegex(ValueError, "complete and contiguous"):
            hourly_from_gdex(incomplete)
        bad = self.grid_frame().assign(precipitation_period_hours=0)
        with self.assertRaisesRegex(ValueError, "periods must be positive"):
            hourly_from_gdex(bad)

    def test_missing_extraction_provenance_fails_strict_contract(self):
        from src.backtesting.strict_dataset import StrictDataError, operating_intervals, select_weather_vintages
        origin = operating_intervals("2025-12").head(1)
        weather = pd.DataFrame({"station": ["LAX"], "model_init_time_utc": ["2025-11-30T00:00Z"],
                                "forecast_lead_hours": [32], "valid_time_utc": [origin.time_utc.iloc[0]],
                                "available_at_utc": ["2025-11-30T10:00Z"],
                                "availability_policy": ["gfs_init_plus_10h_v1"],
                                "source": [GDEX_SOURCE], "model": ["gfs-0p25"],
                                "source_object": [""], "checksum": [""]})
        with self.assertRaisesRegex(StrictDataError, "source_object"):
            select_weather_vintages(weather, origin)


if __name__ == "__main__":
    unittest.main()
