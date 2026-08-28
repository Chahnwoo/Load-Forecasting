import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from scripts.evaluate_strict_tabular import (
    CAISO,
    EXPECTED_ROWS,
    FEATURE_ALLOWLIST,
    RIDGE_ALPHAS,
    SEED,
    TARGET,
    TEST_MONTH,
    TRAIN_MONTHS,
    XGBOOST_GRID,
    EvaluationDataError,
    _xgboost_model,
    cross_validate,
    feature_matrix,
    fit_caiso_calibration,
    final_refit,
    load_strict_month,
    make_expanding_folds,
    make_pipeline,
    score_december,
)


class FeatureEchoModel:
    def __init__(self, offset):
        self.offset = offset

    def predict(self, features):
        return features["load_previous_week"].to_numpy() + self.offset


def frame(rows=1465, start="2025-10-01T07:00:00Z"):
    index = np.arange(rows, dtype=float)
    result = pd.DataFrame({
        "time_utc": pd.date_range(start, periods=rows, freq="h"),
        TARGET: 20000 + index,
        CAISO: 20010 + index,
    })
    for number, name in enumerate(FEATURE_ALLOWLIST, 1):
        result[name] = (index + number) % 17
    result["day_of_week"] = (index.astype(int) // 24) % 7
    result["is_weekend"] = (result.day_of_week >= 5).astype(int)
    return result


class StrictTabularEvaluatorTests(unittest.TestCase):
    def test_allowlist_excludes_leaky_and_provenance_fields(self):
        forbidden = {TARGET, CAISO, "time_utc", "operating_day", "forecast_cutoff_utc",
                     "weather_model_init_time_utc", "weather_available_at_utc", "weather_source",
                     "weather_model", "weather_availability_policy", "weather_source_object",
                     "weather_checksum", "load_previous_week_source_time_utc", "latitude",
                     "longitude", "population_weight"}
        self.assertTrue(forbidden.isdisjoint(FEATURE_ALLOWLIST))
        data = frame(10).assign(arbitrary_numeric_column=999, weather_checksum=123)
        self.assertEqual(list(FEATURE_ALLOWLIST), list(feature_matrix(data).columns))

    def test_month_roles_and_counts_are_constants(self):
        self.assertEqual(("2025-10", "2025-11"), TRAIN_MONTHS)
        self.assertEqual("2025-12", TEST_MONTH)
        self.assertEqual(1465, sum(EXPECTED_ROWS[x] for x in TRAIN_MONTHS))
        self.assertEqual(744, EXPECTED_ROWS[TEST_MONTH])

    def test_existing_hyperparameter_search_space_is_unchanged(self):
        self.assertEqual((0.01, 0.1, 1.0, 10.0, 100.0, 1000.0), RIDGE_ALPHAS)
        self.assertEqual([
            (2, 0.03, 300, 8, 0.8, 0.8, 0.1, 10.0),
            (2, 0.05, 200, 12, 0.9, 0.9, 1.0, 20.0),
            (3, 0.03, 250, 12, 0.8, 0.8, 1.0, 20.0),
        ], [tuple(parameters[key] for key in (
            "max_depth", "learning_rate", "n_estimators", "min_child_weight",
            "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
        )) for parameters in XGBOOST_GRID])

    def test_folds_are_expanding_chronological_weeks(self):
        folds = make_expanding_folds(1465)
        self.assertEqual(3, len(folds))
        for fold in folds:
            self.assertEqual(168, len(fold.validation_indices))
            self.assertLess(fold.train_indices[-1], fold.validation_indices[0])
        validation = np.concatenate([x.validation_indices for x in folds])
        self.assertEqual(len(validation), len(np.unique(validation)))
        self.assertLess(validation[-1], 1465)  # all indices belong to Oct-Nov

    def test_cross_validation_refits_preprocessing_per_fold(self):
        data = frame()
        results = cross_validate("linear", make_pipeline(LinearRegression()), {}, data,
                                 make_expanding_folds(len(data)))
        self.assertEqual([961, 1129, 1297], [x["train_rows"] for x in results])
        self.assertTrue(all(x["validation_rows"] == 168 for x in results))
        self.assertTrue(all(x["train_end_time_utc"] < x["validation_start_time_utc"]
                            for x in results))

    def test_final_refit_uses_all_october_november_rows(self):
        fitted = final_refit({"linear": make_pipeline(LinearRegression())}, frame())
        scaler = fitted["linear"].named_steps["preprocess"].named_transformers_["continuous"]
        self.assertEqual(1465, scaler.n_samples_seen_)

    def test_caiso_calibrations_fit_exactly_october_november(self):
        training = frame()
        training[TARGET] = 125.0 + 0.8 * training[CAISO]
        calibration = fit_caiso_calibration(training)
        self.assertEqual(1465, calibration.training_row_count)
        self.assertEqual(TRAIN_MONTHS, calibration.training_months)
        self.assertAlmostEqual(125.0, calibration.intercept, places=7)
        self.assertAlmostEqual(0.8, calibration.slope, places=12)
        self.assertAlmostEqual(
            float((training[CAISO] - training[TARGET]).mean()),
            calibration.training_bias,
        )

    def test_calibrations_reject_december_rows(self):
        december = frame(744, "2025-12-01T08:00:00Z")
        with self.assertRaisesRegex(EvaluationDataError, "October-November"):
            fit_caiso_calibration(december)

    def test_december_uses_frozen_calibration_and_common_timestamps(self):
        training = frame()
        calibration = fit_caiso_calibration(training)
        december = frame(744, "2025-12-01T08:00:00Z")
        models = {name: FeatureEchoModel(offset) for name, offset in
                  (("linear", 1), ("ridge", 2), ("xgboost", 3))}
        predictions, metrics = score_december(december, models, calibration)

        np.testing.assert_allclose(
            december[CAISO] - calibration.training_bias,
            predictions["caiso_bias_corrected_prediction"],
        )
        np.testing.assert_allclose(
            calibration.intercept + calibration.slope * december[CAISO],
            predictions["caiso_linearly_calibrated_prediction"],
        )
        self.assertEqual(744, predictions.time_utc.nunique())
        self.assertTrue(all(predictions[column].notna().sum() == 744 for column in (
            CAISO, "previous_week_prediction", "linear_prediction", "ridge_prediction",
            "xgboost_prediction", "caiso_bias_corrected_prediction",
            "caiso_linearly_calibrated_prediction",
        )))
        self.assertTrue({
            "rmse_improvement_vs_raw_caiso_pct",
            "rmse_improvement_vs_bias_corrected_caiso_pct",
            "rmse_improvement_vs_linearly_calibrated_caiso_pct",
        }.issubset(metrics.columns))

        # Changing December labels changes scores, but cannot alter any frozen
        # calibration or model prediction.
        changed = december.copy()
        changed[TARGET] += 100_000
        changed_predictions, _ = score_december(changed, models, calibration)
        prediction_columns = [column for column in predictions if column.endswith("prediction")]
        pd.testing.assert_frame_equal(
            predictions[prediction_columns], changed_predictions[prediction_columns]
        )

    def test_duplicate_missing_and_misaligned_timestamps_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = frame(744, "2025-12-01T08:00:00Z")
            data.drop(columns=CAISO).to_csv(root / "strict_rows.csv", index=False)
            dam = data[["time_utc", CAISO]].copy()
            dam.loc[1, "time_utc"] = dam.loc[0, "time_utc"]
            dam.to_csv(root / "dam_forecasts.csv", index=False)
            with self.assertRaisesRegex(EvaluationDataError, "duplicate"):
                load_strict_month(root, TEST_MONTH, 744)
            dam = data[["time_utc", CAISO]].copy()
            dam.loc[0, "time_utc"] = pd.NaT
            dam.to_csv(root / "dam_forecasts.csv", index=False)
            with self.assertRaisesRegex(EvaluationDataError, "missing or invalid"):
                load_strict_month(root, TEST_MONTH, 744)

    def test_december_requires_744_identically_aligned_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = frame(744, "2025-12-01T08:00:00Z")
            data.drop(columns=CAISO).to_csv(root / "strict_rows.csv", index=False)
            data[["time_utc", CAISO]].to_csv(root / "dam_forecasts.csv", index=False)
            loaded = load_strict_month(root, TEST_MONTH, 744)
            self.assertEqual(744, len(loaded))
            self.assertFalse(loaded.time_utc.duplicated().any())

    def test_xgboost_seed_is_reproducible(self):
        try:
            first = make_pipeline(_xgboost_model({"n_estimators": 5, "max_depth": 2}))
        except RuntimeError:
            self.skipTest("xgboost is not installed")
        data = frame(80)
        x, y = feature_matrix(data), data[TARGET]
        p1 = first.fit(x, y).predict(x)
        second = make_pipeline(_xgboost_model({"n_estimators": 5, "max_depth": 2}))
        p2 = second.fit(x, y).predict(x)
        self.assertEqual(202512, SEED)
        np.testing.assert_array_equal(p1, p2)


if __name__ == "__main__":
    unittest.main()
