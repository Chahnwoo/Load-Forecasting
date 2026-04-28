#!/usr/bin/env python3
import unittest
import pandas as pd
import numpy as np
import os
import subprocess
import tempfile
import sys
from datetime import datetime, timezone, timedelta

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib.util
spec = importlib.util.spec_from_file_location("collect_next_day_predictions", os.path.join(os.path.dirname(__file__), "collect_next_day_predictions.py"))
collect_next_day_predictions = importlib.util.module_from_spec(spec)
sys.modules["collect_next_day_predictions"] = collect_next_day_predictions
spec.loader.exec_module(collect_next_day_predictions)
apply_calendar_and_lag_features = collect_next_day_predictions.apply_calendar_and_lag_features

class TestNextDayPipeline(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test artifacts
        self.test_dir = tempfile.TemporaryDirectory()
        self.hist_csv_path = os.path.join(self.test_dir.name, "historical.csv")
        self.next_csv_path = os.path.join(self.test_dir.name, "next_day.csv")
        self.preds_csv_path = os.path.join(self.test_dir.name, "predictions.csv")
        self.metrics_csv_path = os.path.join(self.test_dir.name, "metrics.csv")
        
        # Python executable from virtual environment if running inside it, otherwise sys.executable
        self.python_exe = sys.executable

        self._generate_dummy_data()

    def tearDown(self):
        self.test_dir.cleanup()

    def _generate_dummy_data(self):
        # Generate 48 hours of historical data (2 regions)
        start_hist = datetime(2026, 4, 20, 0, tzinfo=timezone.utc)
        hist_times = [start_hist + timedelta(hours=i) for i in range(48)]
        
        hist_data = []
        for r in ["caiso", "pge"]:
            for t in hist_times:
                hist_data.append({
                    "region": r,
                    "time_key": t.strftime("%Y-%m-%d:%H"),
                    "time_utc": t.strftime("%Y-%m-%d %H:%M:%S"),
                    "temperature_2m": np.random.uniform(50, 90),
                    "apparent_temperature": np.random.uniform(50, 90),
                    "relative_humidity_2m": np.random.uniform(20, 80),
                    "precipitation": 0.0,
                    "cloud_cover": 0.0,
                    "wind_speed_10m": 5.0,
                    "shortwave_radiation": 100.0,
                    "cdd_65f": 0.0,
                    "hdd_65f": 0.0,
                    "load_mw": np.random.uniform(2000, 5000),
                    "is_weekend": 0,
                    "US_federal_holidays": 0,
                    "state_holidays": 0,
                    "load_previous_week": np.random.uniform(2000, 5000),
                })
        
        self.hist_df = pd.DataFrame(hist_data)
        self.hist_df.to_csv(self.hist_csv_path, index=False)

        # Generate 24 hours of next day data, exactly 7 days after the first 24 hours of hist_data
        start_next = start_hist + timedelta(days=7)
        next_times = [start_next + timedelta(hours=i) for i in range(24)]
        
        next_data = []
        for r in ["caiso", "pge"]:
            for t in next_times:
                next_data.append({
                    "region": r,
                    "time_key": t.strftime("%Y-%m-%d:%H"),
                    "time_utc": t.strftime("%Y-%m-%d %H:%M:%S"),
                    "temperature_2m": np.random.uniform(50, 90),
                    "apparent_temperature": np.random.uniform(50, 90),
                    "relative_humidity_2m": np.random.uniform(20, 80),
                    "precipitation": 0.0,
                    "cloud_cover": 0.0,
                    "wind_speed_10m": 5.0,
                    "shortwave_radiation": 100.0,
                    "cdd_65f": 0.0,
                    "hdd_65f": 0.0,
                    "load_pred_mw": np.random.uniform(2000, 5000), # CAISO prediction
                })
        
        self.next_df = pd.DataFrame(next_data)
        self.next_df.to_csv(self.next_csv_path, index=False)

    # ==========================================
    # Requirement 1: Data Collection Edge Cases
    # ==========================================
    def test_collect_apply_features_happy_path(self):
        # Should add is_weekend, holidays, and load_previous_week
        out_df = apply_calendar_and_lag_features(self.next_df, self.hist_csv_path)
        self.assertIn("is_weekend", out_df.columns)
        self.assertIn("US_federal_holidays", out_df.columns)
        self.assertIn("state_holidays", out_df.columns)
        self.assertIn("load_previous_week", out_df.columns)
        
        # load_previous_week should not be NA because the historical data 7 days ago exists
        self.assertFalse(out_df["load_previous_week"].isna().any())

    def test_collect_missing_historical_csv(self):
        # Edge case: historical CSV path is invalid
        out_df = apply_calendar_and_lag_features(self.next_df, "invalid_path.csv")
        self.assertIn("load_previous_week", out_df.columns)
        self.assertTrue(out_df["load_previous_week"].isna().all())

    def test_collect_time_shift_boundary(self):
        # Edge case: What if historical CSV is present but doesn't have the data for exactly 7 days ago?
        empty_hist_path = os.path.join(self.test_dir.name, "empty_hist.csv")
        pd.DataFrame(columns=self.hist_df.columns).to_csv(empty_hist_path, index=False)
        out_df = apply_calendar_and_lag_features(self.next_df, empty_hist_path)
        # Should cleanly merge and leave NaNs
        self.assertTrue(out_df["load_previous_week"].isna().all())

    # ==========================================
    # Requirement 2: Model Training Edge Cases
    # ==========================================
    def test_run_pipeline_ridge_happy_path(self):
        cmd = [
            self.python_exe, "src/modeling/run_next_day_pipeline.py",
            "--historical_csv", self.hist_csv_path,
            "--next_day_csv", self.next_csv_path,
            "--output_predictions", self.preds_csv_path,
            "--model", "ridge"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Ridge pipeline failed: {result.stderr}")
        self.assertTrue(os.path.exists(self.preds_csv_path))
        
        preds_df = pd.read_csv(self.preds_csv_path)
        self.assertIn("predicted_load_mw", preds_df.columns)
        self.assertIn("load_pred_mw", preds_df.columns)

    def test_run_pipeline_lstm_sequence_boundary(self):
        # Sequence models require concatenation of lookback.
        # Ensure it works when historical data has enough rows (we have 48 hours per region, lookback defaults to 24).
        cmd = [
            self.python_exe, "src/modeling/run_next_day_pipeline.py",
            "--historical_csv", self.hist_csv_path,
            "--next_day_csv", self.next_csv_path,
            "--output_predictions", self.preds_csv_path,
            "--model", "lstm",
            "--epochs", "1",      # fast training
            "--lookback", "24"    # exactly 24 hours
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"LSTM pipeline failed: {result.stderr}")
        self.assertTrue(os.path.exists(self.preds_csv_path))
        
        preds_df = pd.read_csv(self.preds_csv_path)
        # We expect exactly 24 next day predictions per region (48 total)
        self.assertEqual(len(preds_df), 48)

    def test_run_pipeline_insufficient_lookback(self):
        # Edge case: If lookback is larger than available historical data, it should fail gracefully
        # or handle it. Here, lookback=100 but we only have 48 hours of hist per region.
        cmd = [
            self.python_exe, "src/modeling/run_next_day_pipeline.py",
            "--historical_csv", self.hist_csv_path,
            "--next_day_csv", self.next_csv_path,
            "--output_predictions", self.preds_csv_path,
            "--model", "lstm",
            "--epochs", "1",
            "--lookback", "100"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Sequence creation inside TorchSequenceRegressor raises ValueError if no sequences.
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No sequences were created", result.stderr)

    def test_run_pipeline_missing_features(self):
        # Edge case: next_day_csv is missing an important feature (e.g., temperature_2m)
        corrupted_next_df = self.next_df.drop(columns=["temperature_2m"])
        corrupted_path = os.path.join(self.test_dir.name, "corrupted_next.csv")
        corrupted_next_df.to_csv(corrupted_path, index=False)
        
        cmd = [
            self.python_exe, "src/modeling/run_next_day_pipeline.py",
            "--historical_csv", self.hist_csv_path,
            "--next_day_csv", corrupted_path,
            "--output_predictions", self.preds_csv_path,
            "--model", "ridge"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        # run_next_day_pipeline.py is designed to zero-fill missing columns!
        self.assertEqual(result.returncode, 0, f"Missing feature zero-filling failed: {result.stderr}")
        self.assertTrue(os.path.exists(self.preds_csv_path))

    # ==========================================
    # Requirement 3: Evaluation Edge Cases
    # ==========================================
    def test_evaluate_happy_path(self):
        # Generate valid predictions csv
        preds_data = []
        for i in range(10):
            preds_data.append({
                "region": "caiso",
                "time_utc": f"2026-04-27 {i:02d}:00:00",
                "load_pred_mw": 2000.0 + i,
                "predicted_load_mw": 2005.0 + i
            })
        pd.DataFrame(preds_data).to_csv(self.preds_csv_path, index=False)
        
        cmd = [
            self.python_exe, "src/evaluation/evaluate_next_day.py",
            self.preds_csv_path,
            "--output_metrics", self.metrics_csv_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Eval failed: {result.stderr}")
        self.assertTrue(os.path.exists(self.metrics_csv_path))
        
        metrics_df = pd.read_csv(self.metrics_csv_path)
        self.assertIn("RMSE", metrics_df.columns)
        self.assertAlmostEqual(metrics_df["RMSE"].iloc[0], 5.0, places=1) # The error is exactly 5.0 everywhere

    def test_evaluate_missing_baseline(self):
        # Edge case: some rows are missing CAISO's prediction. The script should drop them and not crash.
        preds_data = [
            {"region": "caiso", "time_utc": "2026-04-27 00:00:00", "load_pred_mw": 2000.0, "predicted_load_mw": 2005.0},
            {"region": "caiso", "time_utc": "2026-04-27 01:00:00", "load_pred_mw": np.nan, "predicted_load_mw": 2005.0},
        ]
        pd.DataFrame(preds_data).to_csv(self.preds_csv_path, index=False)
        
        cmd = [
            self.python_exe, "src/evaluation/evaluate_next_day.py",
            self.preds_csv_path,
            "--output_metrics", self.metrics_csv_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        
        # It should compute error only for the first row (error = 5.0)
        metrics_df = pd.read_csv(self.metrics_csv_path)
        self.assertAlmostEqual(metrics_df["RMSE"].iloc[0], 5.0, places=1)

    def test_evaluate_empty_predictions(self):
        # Edge case: what if predictions are completely missing/corrupted
        empty_df = pd.DataFrame(columns=["region", "time_utc"])
        empty_df.to_csv(self.preds_csv_path, index=False)
        
        cmd = [
            self.python_exe, "src/evaluation/evaluate_next_day.py",
            self.preds_csv_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CSV must contain 'load_pred_mw' and 'predicted_load_mw' columns", result.stderr)

if __name__ == "__main__":
    unittest.main()
