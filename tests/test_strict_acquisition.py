import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.backtesting.acquisition import (NCEI_GRID004_SOURCE, caiso_request,
                                         grid004_object_url, hourly_from_grid004,
                                         parse_grid004_name, sha256_file, write_manifest)
from scripts.acquire_december_2025_caiso import normalize
from src.backtesting.strict_dataset import ACTUAL_ITEM, DAM_ITEM


class StrictAcquisitionTests(unittest.TestCase):
    def test_caiso_request_is_exact_and_encoded(self):
        url, params = caiso_request(pd.Timestamp("2025-12-01T08:00Z"),
                                    pd.Timestamp("2025-12-02T08:00Z"))
        self.assertEqual("SLD_FCST", params["queryname"])
        self.assertEqual("DAM", params["market_run_id"])
        self.assertEqual("20251201T08:00-0000", params["startdatetime"])
        self.assertIn("oasis.caiso.com/oasisapi/SingleZip?", url)
        self.assertNotIn("load_mw", url)

    def test_grid004_object_parsing(self):
        name = "gfs_4_20251201_0000_057.grb2"
        parsed = parse_grid004_name(name)
        self.assertEqual(57, parsed["forecast_lead_hours"])
        self.assertEqual(pd.Timestamp("2025-12-03T09:00Z"), parsed["valid_time_utc"])
        self.assertTrue(grid004_object_url(name).endswith("/202512/20251201/" + name))
        with self.assertRaises(ValueError):
            parse_grid004_name("gfs_4_20251201_0000_058.grb2")
        with self.assertRaises(ValueError):
            parse_grid004_name("gfs_3_20251201_0000_057.grb2")

    def test_exact_oasis_contract_filtering(self):
        fixture = pd.DataFrame({"INTERVALSTARTTIME_GMT": ["2025-12-01T08:00Z"] * 3,
                                "TAC_AREA_NAME": ["CA ISO-TAC", "PGE-TAC", "CA ISO-TAC"],
                                "DATA_ITEM": [ACTUAL_ITEM, ACTUAL_ITEM, DAM_ITEM],
                                "MARKET_RUN_ID": ["RTM", "RTM", "DAM"], "MW": [1, 2, 3]})
        actual = normalize(fixture, ACTUAL_ITEM)
        dam = normalize(fixture, DAM_ITEM)
        self.assertEqual([1], actual.mw.tolist())
        self.assertEqual([3], dam.mw.tolist())
        self.assertEqual({"CA ISO-TAC"}, set(actual.tac_area_name))

    def test_checksum_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.grb2"
            source.write_bytes(b"grid004 fixture\n")
            self.assertEqual("sha256:cdaf6670e00e868795623ac96c4df1d690231bc03f60e0a3fa26e875588508d9",
                             sha256_file(source))
            manifest = Path(directory) / "manifest.json"
            write_manifest(manifest, [{"source": NCEI_GRID004_SOURCE, "source_object": "fixture",
                                       "checksum": sha256_file(source),
                                       "retrieved_at_utc": "2026-01-01T00:00:00Z"}], month="2025-12",
                           source_documents=["/immutable/ncei-dsi-6182.pdf"])
            payload = json.loads(manifest.read_text())
            self.assertEqual("strict-acquisition-manifest-v1", payload["schema_version"])
            self.assertEqual(NCEI_GRID004_SOURCE, payload["acquisitions"][0]["source"])
            self.assertEqual(["/immutable/ncei-dsi-6182.pdf"], payload["source_documents"])
            with self.assertRaisesRegex(ValueError, "missing provenance"):
                write_manifest(manifest, [{"source": NCEI_GRID004_SOURCE,
                                           "checksum": sha256_file(source)}], month="2025-12",
                               source_documents=["/immutable/ncei-dsi-6182.pdf"])

    def grid_frame(self):
        times = pd.date_range("2025-12-01T06:00Z", periods=3, freq="3h")
        return pd.DataFrame({"station": "LAX", "model_init_time_utc": pd.Timestamp("2025-12-01T00:00Z"),
                             "valid_time_utc": times, "temperature_2m": [10., 13., 16.],
                             "relative_humidity_2m": [40., 70., 100.], "cloud_cover": [0., 30., 60.],
                             "wind_speed_10m": [1., 4., 7.], "precipitation": [3., 6., 9.],
                             "precipitation_period_hours": 3, "shortwave_radiation": [100., 400., 700.],
                             "shortwave_period_hours": 3})

    def test_three_hour_instantaneous_and_interval_semantics(self):
        hourly = hourly_from_grid004(self.grid_frame()).set_index("valid_time_utc")
        self.assertEqual(14., hourly.loc[pd.Timestamp("2025-12-01T10:00Z"), "temperature_2m"])
        # The 09Z APCP amount is distributed over 07, 08, 09Z, not interpolated.
        self.assertEqual(2., hourly.loc[pd.Timestamp("2025-12-01T08:00Z"), "precipitation"])
        self.assertEqual(400., hourly.loc[pd.Timestamp("2025-12-01T08:00Z"), "shortwave_radiation"])
        self.assertEqual(0., hourly.loc[pd.Timestamp("2025-12-01T06:00Z"), "precipitation"])

    def test_incomplete_cycle_and_bad_accumulation_fail(self):
        incomplete = self.grid_frame().drop(index=1)
        with self.assertRaisesRegex(ValueError, "complete and contiguous"):
            hourly_from_grid004(incomplete)
        bad = self.grid_frame().assign(precipitation_period_hours=1)
        with self.assertRaisesRegex(ValueError, "accumulation semantics"):
            hourly_from_grid004(bad)

    def test_missing_extraction_provenance_fails_strict_contract(self):
        from src.backtesting.strict_dataset import StrictDataError, operating_intervals, select_weather_vintages
        origin = operating_intervals("2025-12").head(1)
        weather = pd.DataFrame({"station": ["LAX"], "model_init_time_utc": ["2025-11-30T00:00Z"],
                                "forecast_lead_hours": [32], "valid_time_utc": [origin.time_utc.iloc[0]],
                                "available_at_utc": ["2025-11-30T10:00Z"],
                                "availability_policy": ["gfs_init_plus_10h_v1"],
                                "source": [NCEI_GRID004_SOURCE], "model": ["gfs-grid004-0p5"],
                                "source_object": [""], "checksum": [""]})
        with self.assertRaisesRegex(StrictDataError, "source_object"):
            select_weather_vintages(weather, origin)


if __name__ == "__main__":
    unittest.main()
