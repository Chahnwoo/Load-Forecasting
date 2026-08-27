import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np
import xarray as xr

from src.backtesting.acquisition import (GDEX_SOURCE, caiso_request,
                                         gdex_object_url, gdex_ncss_url, hourly_from_gdex,
                                         parse_gdex_name, sha256_file, write_manifest)
from scripts.acquire_december_2025_caiso import normalize, raw_identifiers
from src.backtesting.strict_dataset import ACTUAL_ITEM, DAM_ITEM
from scripts import acquire_gdex_gfs_0p25 as gdex_script


class StrictAcquisitionTests(unittest.TestCase):
    def _statistical_dataset(self, lower=24., upper=30., *, field="precipitation",
                             bounds_attr=True, include_bounds=True, bounds_values=None):
        time=xr.DataArray([27.],dims=["time1"],attrs={"units":"hours since 2025-11-30 00:00:00"})
        if bounds_attr: time.attrs["bounds"]="time1_bounds"
        data=xr.DataArray([12.],dims=["time1"],coords={"time1":time})
        variables={field:data}
        if include_bounds:
            values=np.array([[lower,upper]]) if bounds_values is None else np.asarray(bounds_values)
            first_dim="time1" if values.shape[0] == 1 else "interval"
            variables["time1_bounds"]=xr.DataArray(values,dims=[first_dim,"bounds"])
        return xr.Dataset(variables)

    def _parsed_f030(self):
        return parse_gdex_name("gfs.0p25.2025113000.f030.grib2")

    def test_ncss_cf_bounds_define_statistical_periods(self):
        for field in ("precipitation","shortwave_radiation"):
            with self.subTest(field=field):
                ds=self._statistical_dataset(field=field)
                result=gdex_script.statistical_time_bounds(ds,ds[field],self._parsed_f030(),field)
                prefix="shortwave" if field == "shortwave_radiation" else field
                self.assertEqual(24.,result[f"{prefix}_statistical_interval_start_lead_hours"])
                self.assertEqual(30.,result[f"{prefix}_statistical_interval_end_lead_hours"])
                self.assertEqual(6.,result[f"{prefix}_statistical_interval_hours"])
                self.assertEqual(pd.Timestamp("2025-12-01T00:00Z"),result[f"{prefix}_statistical_interval_start_utc"])
                self.assertEqual(pd.Timestamp("2025-12-01T06:00Z"),result[f"{prefix}_statistical_interval_end_utc"])
        ds=self._statistical_dataset(27,30)
        result=gdex_script.statistical_time_bounds(ds,ds.precipitation,self._parsed_f030(),"precipitation")
        self.assertEqual(3.,result["precipitation_statistical_interval_hours"])

    def test_ncss_cf_bounds_fail_closed(self):
        cases=[(self._statistical_dataset(24,27),"does not match backing-object"),
               (self._statistical_dataset(bounds_attr=False),"bounds attribute"),
               (self._statistical_dataset(include_bounds=False),"is missing"),
               (self._statistical_dataset(bounds_values=[[24,27],[27,30]]),"exactly one lower/upper pair"),
               (self._statistical_dataset(30,30),"must be positive"),
               (self._statistical_dataset(31,30),"must be positive")]
        for ds,message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError,message):
                gdex_script.statistical_time_bounds(ds,ds.precipitation,self._parsed_f030(),"precipitation")

    def test_gdex_required_objects_bracket_three_hour_products(self):
        fixture = pd.DataFrame({
            "operating_day": ["fixture", "fixture"],
            "forecast_cutoff_utc": [pd.Timestamp("2025-11-30T11:00Z")] * 2,
            "time_utc": [pd.Timestamp("2025-12-01T07:00Z"), pd.Timestamp("2025-12-01T08:00Z")],
        })
        original = gdex_script.operating_intervals
        gdex_script.operating_intervals = lambda month: fixture
        try:
            objects = gdex_script.required_objects("fixture")
        finally:
            gdex_script.operating_intervals = original
        self.assertEqual([
            "gfs.0p25.2025113000.f030.grib2",
            "gfs.0p25.2025113000.f033.grib2",
        ], objects)
        self.assertFalse(any("f031" in item or "f032" in item for item in objects))

    def _gdex_metadata_session(self, content):
        class Response:
            def __init__(self, body): self.content = body
            def raise_for_status(self): pass
        class Session:
            def get(self, *args, **kwargs): return Response(content)
        return Session()

    def _gdex_das(self):
        return (Path(__file__).parent / "fixtures/gdex_gfs_opendap.das").read_bytes()

    def test_opendap_discovery_uses_exact_instantaneous_machine_names(self):
        found, endpoint = gdex_script.discover_variables(
            self._gdex_metadata_session(self._gdex_das()),
            "gfs.0p25.2025113000.f030.grib2")
        self.assertEqual("Temperature_height_above_ground", found["temperature_2m"])
        self.assertEqual("Relative_humidity_height_above_ground", found["relative_humidity_2m"])
        self.assertEqual("u-component_of_wind_height_above_ground", found["u_wind_10m"])
        self.assertEqual("v-component_of_wind_height_above_ground", found["v_wind_10m"])
        self.assertTrue(endpoint.endswith("gfs.0p25.2025113000.f030.grib2.das"))

    def test_grib_metadata_discovers_three_and_six_hour_shortwave(self):
        three, _ = gdex_script.discover_variables(
            self._gdex_metadata_session(self._gdex_das()),
            "gfs.0p25.2025113000.f030.grib2")
        self.assertEqual("Downward_Short-Wave_Radiation_Flux_surface_3_Hour_Average",
                         three["shortwave_radiation"])
        six_das = self._gdex_das().replace(b"_3_Hour_Average", b"_6_Hour_Average").replace(
            b'"3 hour"', b'"6 hour"')
        six, _ = gdex_script.discover_variables(
            self._gdex_metadata_session(six_das), "gfs.0p25.2025113000.f033.grib2")
        self.assertEqual("Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average",
                         six["shortwave_radiation"])

    def test_grib_metadata_discovers_mixed_precipitation_and_entire_atmosphere_cloud(self):
        found, _ = gdex_script.discover_variables(
            self._gdex_metadata_session(self._gdex_das()),
            "gfs.0p25.2025113000.f030.grib2")
        self.assertEqual("Total_precipitation_surface_Mixed_intervals_Accumulation",
                         found["precipitation"])
        self.assertEqual("Total_cloud_cover_entire_atmosphere", found["cloud_cover"])

    def test_averaged_only_cloud_cover_fails_closed(self):
        das = self._gdex_das().replace(
            b'String Grib2_Statistical_Process_Type "UnknownStatType--1";',
            b'String Grib2_Statistical_Process_Type "Average";',
            1)
        with self.assertRaisesRegex(RuntimeError, r"expected one cloud_cover, found \[\]"):
            gdex_script.discover_variables(
                self._gdex_metadata_session(das), "gfs.0p25.2025113000.f030.grib2")

    def test_two_instantaneous_cloud_candidates_fail_closed(self):
        duplicate = b"""
    Other_instantaneous_cloud_cover {
        String Grib2_Parameter_Name "Total cloud cover";
        String Grib2_Level_Desc "Entire atmosphere";
        String Grib2_Statistical_Process_Type "UnknownStatType--1";
        String units "%";
    }
"""
        das = self._gdex_das().replace(b"    NC_GLOBAL {", duplicate + b"    NC_GLOBAL {")
        with self.assertRaisesRegex(RuntimeError, "expected one cloud_cover.*Other_instantaneous"):
            gdex_script.discover_variables(
                self._gdex_metadata_session(das), "gfs.0p25.2025113000.f030.grib2")

    def test_ambiguous_physical_metadata_matches_fail_closed_with_diagnostics(self):
        duplicate = b"""
    Other_shortwave_surface_3_Hour_Average {
        String Grib2_Parameter_Name "Downward Short-Wave Radiation Flux";
        String Grib2_Level_Desc "Ground or water surface";
        String Grib2_Statistical_Process_Type "Average";
        String units "W.m-2";
    }
"""
        das = self._gdex_das().replace(b"    NC_GLOBAL {", duplicate + b"    NC_GLOBAL {")
        with self.assertRaisesRegex(RuntimeError, "expected one shortwave_radiation.*parameter"):
            gdex_script.discover_variables(
                self._gdex_metadata_session(das), "gfs.0p25.2025113000.f030.grib2")

    def test_vertical_levels_are_selected_per_physical_field(self):
        heights = xr.DataArray([2., 10.], dims="vertical", attrs={"units": "m", "standard_name": "height"})
        array = xr.DataArray([280., 290.], dims="vertical", coords={"vertical": heights}, attrs={"units": "K"})
        self.assertEqual(280., float(gdex_script._select_height(array, 2, "temperature_2m")))
        self.assertEqual(290., float(gdex_script._select_height(array, 10, "u_wind_10m")))
        with self.assertRaisesRegex(RuntimeError, "requires 5 m"):
            gdex_script._select_height(array, 5, "temperature_2m")

    def test_extract_selects_two_metre_temperature_rh_and_ten_metre_wind(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "levels.nc"
            coords = {"height": xr.DataArray([2., 10.], dims="height", attrs={"units": "m", "standard_name": "height"}),
                      "lat": [35.], "lon": [-120.]}
            def field(values, units):
                return xr.DataArray(np.asarray(values).reshape(2, 1, 1), dims=("height", "lat", "lon"), coords=coords, attrs={"units": units})
            ds=xr.Dataset({"t": field([280., 999.], "K"), "rh": field([45., 999.], "%"),
                           "u": field([999., 3.], "m/s"), "v": field([999., 4.], "m/s"),
                           "cloud": (("lat","lon"), [[50.]]),
                           "precip": (("lat","lon"), [[3.]]), "solar": (("lat","lon"), [[100.]])})
            ds["precip"].attrs["description"]="0-3 hour accumulation"
            ds["solar"].attrs["description"]="0-3 hour average"
            ds.to_netcdf(path, engine="scipy")
            variables={"temperature_2m":"t", "relative_humidity_2m":"rh", "u_wind_10m":"u", "v_wind_10m":"v",
                       "cloud_cover":"cloud", "precipitation":"precip", "shortwave_radiation":"solar"}
            stations=pd.DataFrame({"station_name":["X"], "latitude":[35.], "longitude":[-120.], "population_weight":[1.]})
            row=gdex_script.extract(path,variables,stations,{})[0]
            self.assertAlmostEqual(6.85,row["temperature_2m"])
            self.assertEqual((45.,3.,4.),(row["relative_humidity_2m"],row["u_wind_10m"],row["v_wind_10m"]))

    def test_acquisition_module_entrypoints_import_from_repository_root(self):
        repository_root = Path(__file__).resolve().parents[1]
        modules = (
            "scripts.acquire_december_2025_caiso",
            "scripts.acquire_gdex_gfs_0p25",
            "scripts.acquire_strict_backtest_inputs",
        )
        for module in modules:
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=repository_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(0, result.returncode, result.stderr)

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
        self.assertEqual("netCDF", params["accept"])
        self.assertNotIn("netCDF4", url)
        self.assertEqual("2025-12-01T06:00:00Z", params["time"])
        self.assertEqual("Temperature_height_above_ground", params["var"])
        self.assertEqual({"north": "42", "south": "32", "east": "-114", "west": "-125"},
                         {key: params[key] for key in ("north", "south", "east", "west")})
        self.assertNotIn("fileServer", url)

    def _ncss_response(self, content, content_type="application/x-netcdf", status=200,
                       url="https://fixture.example/final"):
        class Response:
            headers = {"content-type": content_type}
            status_code = status
            def __init__(self):
                self.content = content
                self.url = url
        class Session:
            def get(self, *args, **kwargs): return Response()
        return Session()

    def test_ncss_download_accepts_classic_netcdf_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subset.nc"
            gdex_script.download(self._ncss_response(b"CDF\x01fixture"),
                                 "https://fixture.example/request", path)
            self.assertEqual(b"CDF\x01fixture", path.read_bytes())

    def test_ncss_download_rejects_error_documents_and_invalid_magic(self):
        cases = ((b"<html>error</html>", "text/html", "not NetCDF-compatible"),
                 (b"plain error", "text/plain", "not NetCDF-compatible"),
                 (b"not a netcdf", "application/x-netcdf", "invalid NetCDF signature"))
        for body, content_type, message in cases:
            with self.subTest(content_type=content_type, body=body), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, message):
                    gdex_script.download(self._ncss_response(body, content_type),
                                         "https://fixture.example/request", Path(directory) / "subset.nc")

    def test_ncss_http_error_includes_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError) as raised:
                gdex_script.download(self._ncss_response(
                    b"Format netCDF4 is not supported", "text/plain", 400),
                    "https://fixture.example/request", Path(directory) / "subset.nc")
        message = str(raised.exception)
        self.assertIn("HTTP 400", message)
        self.assertIn("content-type='text/plain'", message)
        self.assertIn("Format netCDF4 is not supported", message)
        self.assertIn("final request URL=https://fixture.example/final", message)

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

    def test_six_hour_intervals_preserve_total_and_average_at_boundary(self):
        frame=self.grid_frame().assign(precipitation_period_hours=6,
                                       shortwave_period_hours=6)
        hourly=hourly_from_gdex(frame).set_index("valid_time_utc")
        # Only 06Z is in the output from the first [01Z, 06Z] convention interval,
        # but its total remains divided by all six represented hours.
        self.assertEqual(.5,hourly.loc[pd.Timestamp("2025-12-01T06:00Z"),"precipitation"])
        self.assertEqual(100.,hourly.loc[pd.Timestamp("2025-12-01T06:00Z"),"shortwave_radiation"])
        self.assertEqual(pd.Timestamp("2025-12-01T06:00Z"),hourly.index.min())
        self.assertNotIn(pd.Timestamp("2025-12-01T05:00Z"),hourly.index)

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
