"""
Robustness check for the trained model.

A model that performs worse than predicting the mean
is fundamentally broken. Comparing against this baseline 
is the simplest and most defensible robustness check.

The trained model must outperform a naive baseline (predicting the mean
quantity for every row). We measure this using R² score on a held-out test
segment. If the model's R² is not meaningfully higher than the baseline's,
the model has not learned anything from the input features and we flag it
as failing the robustness check.

We require the model to achieve R² ≥ 0.0 on test data. The baseline
by definition has R² = 0 on test data. So our requirement is
"the model must do at least as well as predicting the mean". This is the
absolute minimum bar — anything lower means the model is actively harmful.
"""

import logging

import numpy as np
import pandas as pd

from src.data_loader import resolve_latest
from src.train import (
    resolve_latest_model,
    load_model,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    _load_manifest,
)

import joblib

logger = logging.getLogger(__name__)


class RobustnessCheckFailed(Exception):
    """Raised when the model fails the robustness check."""
    pass


def validate_model() -> dict:
    """
    Load the latest model, evaluate it on a held-out test segment, and
    compare against a baseline mean predictor.

    Returns a dictionary with the test results. Raises RobustnessCheckFailed
    if the model does not meet the robustness expectation.
    """
    logger.info("Loading latest model ...")
    model_path = resolve_latest_model()
    model = joblib.load(model_path)

    logger.info("Loading validation data ...")
    df = pd.read_parquet(resolve_latest())
    regular = df[~df["invoice_no"].str.startswith("C", na=False)]
    test_df = regular.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    # Use the last 20% of rows as held-out test data
    # (the data is roughly time-ordered so this is also a temporal split)
    split = int(len(test_df) * 0.8)
    holdout = test_df.iloc[split:]
    logger.info("Validating on %d held-out rows.", len(holdout))

    X_test = holdout[FEATURE_COLUMNS]
    y_test = holdout[TARGET_COLUMN].values

    # Model predictions and score
    y_pred = model.predict(X_test)
    model_r2 = _r2_score(y_test, y_pred)

    # Baseline: predict the mean of training quantities for every row
    train_mean = test_df[TARGET_COLUMN].iloc[:split].mean()
    baseline_pred = np.full_like(y_test, train_mean, dtype=float)
    baseline_r2 = _r2_score(y_test, baseline_pred)

    results = {
        "model_path":  str(model_path),
        "test_rows":   len(holdout),
        "model_r2":    float(model_r2),
        "baseline_r2": float(baseline_r2),
        "passed":      model_r2 >= baseline_r2,
    }

    logger.info("Model R²: %.4f", results["model_r2"])
    logger.info("Baseline R² (mean predictor): %.4f", results["baseline_r2"])

    if not results["passed"]:
        raise RobustnessCheckFailed(
            f"Model R² ({model_r2:.4f}) did not beat baseline ({baseline_r2:.4f}). "
            f"The model has not learned anything useful from the features."
        )

    logger.info("Robustness check PASSED.")
    return results


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² score manually.
    R² = 1 - SS_res/SS_tot.
    """
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# Entry point for direct execution (used by Docker step 3)

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    try:
        validate_model()
        sys.exit(0)
    except RobustnessCheckFailed as e:
        logger.error("Robustness check failed: %s", e)
        sys.exit(1)    
