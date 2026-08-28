#!/usr/bin/env python3
"""Leakage-resistant tabular evaluation for the certified CAISO strict backtest.

This entry point deliberately stands alone from the repository's retrospective
training code.  October and November are loaded, validated, and used for model
selection before the December files are read.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 202512
TRAIN_MONTHS = ("2025-10", "2025-11")
TEST_MONTH = "2025-12"
EXPECTED_ROWS = {"2025-10": 744, "2025-11": 721, "2025-12": 744}
TARGET = "actual_load_mw"
CAISO = "caiso_dam_forecast_mw"
TIME = "time_utc"

# This is intentionally an allowlist, rather than a denylist or a dtype-based
# selection.  New columns in strict_rows.csv can therefore never become model
# inputs without a reviewed source change.
FEATURE_ALLOWLIST = (
    "load_previous_week",
    "temperature_2m",
    "relative_humidity_2m",
    "u_wind_10m",
    "v_wind_10m",
    "cloud_cover",
    "precipitation",
    "shortwave_radiation",
    "forecast_lead_hours",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
)
CATEGORICAL_FEATURES = ("day_of_week",)
BINARY_FEATURES = ("is_weekend",)
CONTINUOUS_FEATURES = tuple(
    x for x in FEATURE_ALLOWLIST if x not in CATEGORICAL_FEATURES + BINARY_FEATURES
)
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
XGBOOST_GRID = (
    {"max_depth": 2, "learning_rate": 0.03, "n_estimators": 300,
     "min_child_weight": 8, "subsample": 0.8, "colsample_bytree": 0.8,
     "reg_alpha": 0.1, "reg_lambda": 10.0},
    {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 200,
     "min_child_weight": 12, "subsample": 0.9, "colsample_bytree": 0.9,
     "reg_alpha": 1.0, "reg_lambda": 20.0},
    {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 250,
     "min_child_weight": 12, "subsample": 0.8, "colsample_bytree": 0.8,
     "reg_alpha": 1.0, "reg_lambda": 20.0},
)


class EvaluationDataError(ValueError):
    """Raised when strict evaluator inputs fail closed validation."""


@dataclass(frozen=True)
class TimeFold:
    train_indices: np.ndarray
    validation_indices: np.ndarray


@dataclass(frozen=True)
class CaisoCalibration:
    """Frozen CAISO-only parameters estimated from the training period."""

    training_bias: float
    intercept: float
    slope: float
    training_row_count: int
    training_months: tuple[str, ...]


def fit_caiso_calibration(training: pd.DataFrame) -> CaisoCalibration:
    """Fit both CAISO calibrations using exactly October-November labels."""
    required = {TIME, TARGET, CAISO}
    if not required.issubset(training.columns):
        raise EvaluationDataError(
            f"calibration training data is missing columns: {sorted(required - set(training.columns))}"
        )
    times = pd.to_datetime(training[TIME], utc=True, errors="coerce")
    pacific_months = tuple(sorted({
        t.tz_convert("America/Los_Angeles").strftime("%Y-%m") for t in times
        if not pd.isna(t)
    }))
    expected_count = sum(EXPECTED_ROWS[month] for month in TRAIN_MONTHS)
    if (times.isna().any() or times.duplicated().any() or len(training) != expected_count
            or pacific_months != TRAIN_MONTHS):
        raise EvaluationDataError(
            "CAISO calibration requires exactly the 1465 unique October-November rows"
        )
    actual = training[TARGET].to_numpy(dtype=float)
    caiso = training[CAISO].to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(caiso).all():
        raise EvaluationDataError("CAISO calibration inputs must be finite")
    fitted = LinearRegression().fit(caiso.reshape(-1, 1), actual)
    return CaisoCalibration(
        training_bias=float(np.mean(caiso - actual)),
        intercept=float(fitted.intercept_),
        slope=float(fitted.coef_[0]),
        training_row_count=len(training),
        training_months=TRAIN_MONTHS,
    )


def _read_csv(path: Path, product: str) -> pd.DataFrame:
    if not path.is_file():
        raise EvaluationDataError(f"missing {product}: {path}")
    return pd.read_csv(path)


def _validated_times(frame: pd.DataFrame, product: str) -> pd.Series:
    if TIME not in frame:
        raise EvaluationDataError(f"{product} is missing required column {TIME!r}")
    parsed = pd.to_datetime(frame[TIME], utc=True, errors="coerce")
    if parsed.isna().any():
        raise EvaluationDataError(f"{product} contains missing or invalid {TIME}")
    if parsed.duplicated().any():
        raise EvaluationDataError(f"{product} contains duplicate {TIME} keys")
    return parsed


def load_strict_month(root: Path, month: str, expected_rows: int) -> pd.DataFrame:
    """Read, align, and validate one strict model/benchmark month."""
    rows = _read_csv(root / "strict_rows.csv", "strict rows")
    dam = _read_csv(root / "dam_forecasts.csv", "CAISO DAM forecasts")
    missing_features = sorted(set(FEATURE_ALLOWLIST) - set(rows.columns))
    if missing_features:
        raise EvaluationDataError(
            f"strict_rows.csv is missing required allowlisted features: {missing_features}"
        )
    if TARGET not in rows:
        raise EvaluationDataError(f"strict_rows.csv is missing target {TARGET!r}")
    if CAISO not in dam:
        raise EvaluationDataError(f"dam_forecasts.csv is missing benchmark {CAISO!r}")
    rows = rows.copy()
    dam = dam.copy()
    rows[TIME] = _validated_times(rows, "strict_rows.csv")
    dam[TIME] = _validated_times(dam, "dam_forecasts.csv")
    if len(rows) != expected_rows or len(dam) != expected_rows:
        raise EvaluationDataError(
            f"{month} must contain exactly {expected_rows} model and CAISO rows; "
            f"found {len(rows)} and {len(dam)}"
        )
    if set(rows[TIME]) != set(dam[TIME]):
        raise EvaluationDataError(f"{month} model and CAISO {TIME} keys are not identical")
    result = rows[[TIME, TARGET, *FEATURE_ALLOWLIST]].merge(
        dam[[TIME, CAISO]], on=TIME, how="inner", validate="one_to_one"
    ).sort_values(TIME).reset_index(drop=True)
    expected_period = pd.Period(month, freq="M")
    if not all(pd.Period(t.tz_convert("America/Los_Angeles").tz_localize(None), freq="M") == expected_period
               for t in result[TIME]):
        raise EvaluationDataError(f"{month} contains target timestamps outside its Pacific operating month")
    numeric = [TARGET, CAISO, *FEATURE_ALLOWLIST]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    if result[[TIME, *numeric]].isna().any().any():
        raise EvaluationDataError(f"{month} contains missing or non-numeric required values")
    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
        raise EvaluationDataError(f"{month} contains non-finite required values")
    return result


def make_expanding_folds(row_count: int, windows: int = 3, window_hours: int = 168) -> list[TimeFold]:
    """Create adjacent validation weeks at the end of chronological training data."""
    first_validation = row_count - windows * window_hours
    if first_validation <= 0:
        raise EvaluationDataError("not enough training rows for expanding-window validation")
    folds = []
    for start in range(first_validation, row_count, window_hours):
        folds.append(TimeFold(np.arange(start), np.arange(start, start + window_hours)))
    return folds


def make_preprocessor() -> ColumnTransformer:
    try:
        categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        categorical = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [("continuous", StandardScaler(), list(CONTINUOUS_FEATURES)),
         ("categorical", categorical, list(CATEGORICAL_FEATURES)),
         ("binary", "passthrough", list(BINARY_FEATURES))],
        remainder="drop",
    )


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Select only reviewed features, regardless of arbitrary extra columns."""
    missing = sorted(set(FEATURE_ALLOWLIST) - set(frame.columns))
    if missing:
        raise EvaluationDataError(f"missing allowlisted model features: {missing}")
    return frame.loc[:, list(FEATURE_ALLOWLIST)].copy()


def make_pipeline(model: Any) -> Pipeline:
    return Pipeline([("preprocess", make_preprocessor()), ("model", model)])


def _xgboost_model(parameters: dict[str, Any]):
    try:
        import xgboost
    except ImportError as exc:
        raise RuntimeError("xgboost is required; install the project requirements") from exc
    return xgboost.XGBRegressor(
        objective="reg:squarederror", random_state=SEED, n_jobs=1,
        tree_method="hist", verbosity=0, **parameters
    )


def cross_validate(name: str, estimator: Pipeline, parameters: dict[str, Any],
                   training: pd.DataFrame, folds: Iterable[TimeFold]) -> list[dict[str, Any]]:
    """Fit a fresh preprocessing pipeline on each fold's training prefix."""
    x, y = feature_matrix(training), training[TARGET].to_numpy()
    results = []
    for fold_number, fold in enumerate(folds, start=1):
        fitted = clone(estimator).fit(x.iloc[fold.train_indices], y[fold.train_indices])
        prediction = fitted.predict(x.iloc[fold.validation_indices])
        results.append({
            "model": name, "parameters": json.dumps(parameters, sort_keys=True),
            "fold": fold_number, "train_rows": len(fold.train_indices),
            "validation_rows": len(fold.validation_indices),
            "train_end_time_utc": training[TIME].iloc[fold.train_indices[-1]],
            "validation_start_time_utc": training[TIME].iloc[fold.validation_indices[0]],
            "validation_end_time_utc": training[TIME].iloc[fold.validation_indices[-1]],
            "validation_rmse": math.sqrt(mean_squared_error(y[fold.validation_indices], prediction)),
        })
    mean_rmse = float(np.mean([r["validation_rmse"] for r in results]))
    for result in results:
        result["mean_validation_rmse"] = mean_rmse
    return results


def select_models(training: pd.DataFrame) -> tuple[dict[str, Pipeline], dict[str, Any], pd.DataFrame]:
    folds = make_expanding_folds(len(training))
    candidates = [("linear", {}, make_pipeline(LinearRegression()))]
    candidates.extend(("ridge", {"alpha": alpha}, make_pipeline(Ridge(alpha=alpha)))
                      for alpha in RIDGE_ALPHAS)
    candidates.extend(("xgboost", params, make_pipeline(_xgboost_model(params)))
                      for params in XGBOOST_GRID)
    all_results: list[dict[str, Any]] = []
    candidate_scores: dict[str, list[tuple[float, dict[str, Any], Pipeline]]] = {}
    for name, params, pipeline in candidates:
        rows = cross_validate(name, pipeline, params, training, folds)
        all_results.extend(rows)
        candidate_scores.setdefault(name, []).append((rows[0]["mean_validation_rmse"], params, pipeline))
    selected, hyperparameters = {}, {}
    for name, choices in candidate_scores.items():
        _, params, pipeline = min(choices, key=lambda item: item[0])
        selected[name] = pipeline
        hyperparameters[name] = params
    return final_refit(selected, training), hyperparameters, pd.DataFrame(all_results)


def final_refit(selected: dict[str, Pipeline], training: pd.DataFrame) -> dict[str, Pipeline]:
    """Refit fresh selected pipelines on every October-November observation."""
    x, y = feature_matrix(training), training[TARGET]
    return {name: clone(pipeline).fit(x, y) for name, pipeline in selected.items()}


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if len(actual) != len(prediction) or not np.isfinite(prediction).all():
        raise EvaluationDataError("prediction counts must match actuals and all predictions must be finite")
    if np.any(actual == 0):
        raise EvaluationDataError("MAPE is undefined because actual_load_mw contains zero")
    error = prediction - actual
    return {"rmse": math.sqrt(float(np.mean(error ** 2))),
            "mae": float(mean_absolute_error(actual, prediction)),
            "mape": float(np.mean(np.abs(error / actual)) * 100),
            "bias": float(np.mean(error))}


def score_december(test: pd.DataFrame, models: dict[str, Pipeline],
                   calibration: CaisoCalibration) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(test) != EXPECTED_ROWS[TEST_MONTH]:
        raise EvaluationDataError("December scoring requires exactly 744 aligned target rows")
    predictions = test[[TIME, TARGET, CAISO]].copy()
    predictions["previous_week_prediction"] = test["load_previous_week"].to_numpy()
    # Only the already-frozen parameters are used here; December actuals are
    # retained solely for scoring below.
    predictions["caiso_bias_corrected_prediction"] = (
        predictions[CAISO] - calibration.training_bias
    )
    predictions["caiso_linearly_calibrated_prediction"] = (
        calibration.intercept + calibration.slope * predictions[CAISO]
    )
    for name in ("linear", "ridge", "xgboost"):
        values = np.asarray(models[name].predict(feature_matrix(test)), dtype=float)
        if len(values) != 744 or not np.isfinite(values).all():
            raise EvaluationDataError(f"{name} did not produce 744 finite predictions")
        predictions[f"{name}_prediction"] = values
    actual = predictions[TARGET].to_numpy(dtype=float)
    method_columns = {
        "previous_week": "previous_week_prediction", "linear": "linear_prediction",
        "ridge": "ridge_prediction", "xgboost": "xgboost_prediction", "caiso_dam": CAISO,
        "caiso_bias_corrected": "caiso_bias_corrected_prediction",
        "caiso_linearly_calibrated": "caiso_linearly_calibrated_prediction",
    }
    metrics = [{"method": method, **_metrics(actual, predictions[column].to_numpy(dtype=float))}
               for method, column in method_columns.items()]
    baseline_names = {
        "caiso_dam": "raw_caiso", "caiso_bias_corrected": "bias_corrected_caiso",
        "caiso_linearly_calibrated": "linearly_calibrated_caiso",
    }
    baseline_rmse = {name: next(row["rmse"] for row in metrics if row["method"] == method)
                     for method, name in baseline_names.items()}
    for row in metrics:
        for name, rmse in baseline_rmse.items():
            row[f"rmse_improvement_vs_{name}_pct"] = (
                (rmse - row["rmse"]) / rmse * 100 if rmse else 0.0
            )
        # Retain the original output field for downstream consumers.
        row["rmse_improvement_vs_caiso_pct"] = row["rmse_improvement_vs_raw_caiso_pct"]
    return predictions, pd.DataFrame(metrics)


def evaluate(train_roots: list[Path], test_root: Path, output: Path) -> None:
    if len(train_roots) != 2:
        raise EvaluationDataError("exactly October and November training roots are required")
    training_parts = [load_strict_month(root, month, EXPECTED_ROWS[month])
                      for root, month in zip(train_roots, TRAIN_MONTHS)]
    training = pd.concat(training_parts, ignore_index=True).sort_values(TIME).reset_index(drop=True)
    if len(training) != 1465 or training[TIME].duplicated().any():
        raise EvaluationDataError("October-November training data must have 1465 unique rows")
    models, selected, cv_results = select_models(training)
    calibration = fit_caiso_calibration(training)

    # This ordering is part of the leakage barrier: December is not read until
    # selection and the all-training-row refit above have completed.
    test = load_strict_month(test_root, TEST_MONTH, EXPECTED_ROWS[TEST_MONTH])
    predictions, metrics = score_december(test, models, calibration)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    cv_results.to_csv(output / "cv_results.csv", index=False)
    (output / "feature_manifest.json").write_text(json.dumps({
        "feature_allowlist": list(FEATURE_ALLOWLIST), "continuous": list(CONTINUOUS_FEATURES),
        "categorical": list(CATEGORICAL_FEATURES), "binary": list(BINARY_FEATURES),
    }, indent=2) + "\n")
    import xgboost
    manifest = {
        "training_months": list(TRAIN_MONTHS), "test_month": TEST_MONTH,
        "training_row_count": len(training), "test_row_count": len(test),
        "training_caiso_bias": calibration.training_bias,
        "calibration_intercept": calibration.intercept,
        "calibration_slope": calibration.slope,
        "calibration_training_months": list(calibration.training_months),
        "calibration_training_row_count": calibration.training_row_count,
        "selected_hyperparameters": selected, "random_seed": SEED,
        "library_versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                             "xgboost": xgboost.__version__},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("data/backtesting")
    parser.add_argument("--october-root", type=Path, default=base / "2025-10/processed/strict")
    parser.add_argument("--november-root", type=Path, default=base / "2025-11/processed/strict")
    parser.add_argument("--december-root", type=Path, default=base / "2025-12/processed/strict")
    parser.add_argument("--output", type=Path,
                        default=base / "evaluation/strict_oct_nov_to_dec")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate([args.october_root, args.november_root], args.december_root, args.output)


if __name__ == "__main__":
    main()
