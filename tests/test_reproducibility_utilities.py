import unittest

import numpy as np
import pandas as pd

from scripts.analyze_strict_predictions import METHODS, analyze


class PredictionAnalysisTests(unittest.TestCase):
    def test_analysis_keeps_operating_days_whole_and_is_deterministic(self):
        times = pd.date_range("2025-12-01T08:00Z", periods=48, freq="h")
        actual = np.full(48, 100.0)
        frame = pd.DataFrame({"time_utc": times, "actual_load_mw": actual})
        frame[METHODS["ridge"]] = actual + 1
        frame[METHODS["raw_caiso"]] = actual + 4
        frame[METHODS["linearly_calibrated_caiso"]] = actual + 2
        frame[METHODS["previous_week"]] = actual + 0.5
        first = analyze(frame, resamples=100, seed=7)
        second = analyze(frame, resamples=100, seed=7)
        self.assertEqual(2, len(first[0]))
        self.assertEqual([24, 24], first[0]["hours"].tolist())
        pd.testing.assert_frame_equal(first[1], second[1])
        raw = first[1].query("baseline == 'raw_caiso'").iloc[0]
        self.assertAlmostEqual(75.0, raw.rmse_improvement_pct)
        self.assertEqual(2, raw.operating_day_wins)

    def test_analysis_rejects_duplicate_hours(self):
        frame = pd.DataFrame({"time_utc": ["2025-12-01T08:00Z"] * 2,
                              "actual_load_mw": [1, 1]})
        for column in METHODS.values():
            frame[column] = 1
        with self.assertRaisesRegex(ValueError, "unique"):
            analyze(frame, resamples=2)


if __name__ == "__main__":
    unittest.main()
