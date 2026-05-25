"""
Robustness check for the trained model.

The trained model must outperform a naive baseline (predicting the mean
quantity for every row). We measure this using R² score on a held-out test
segment. If the model's R² is not meaningfully higher than the baseline's,
the model has not learned anything from the input features and we flag it
as failing the robustness check. A model that performs worse than predicting
the mean is fundamentally broken — this is the simplest and most defensible
robustness check.

For Task 3: if the model was trained with a temporal cutoff
(train_end_month in the manifest), the held-out segment is the data
AFTER the cutoff - a proper unseen split. Otherwise the function falls
back to the Task 2 behavior of slicing the last 20% of training data
by row order.

We require model R^2 >= baseline R^2 on the holdout. The baseline by
definition has R^2 = 0 on test data, so this requires "the model must
do at least as well as predicting the mean". Anything lower means the
model is actively harmful.
"""

import logging

import joblib
import numpy as np
import pandas as pd

from src.data_loader import resolve_latest
from src.train import (
    resolve_latest_model,
    get_latest_model_hash,
    get_model_metadata,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


class RobustnessCheckFailed(Exception):
    """Raised when the model fails the robustness check."""
    pass


def validate_model() -> dict:
    """
    Load the latest model, evaluate it on a held-out segment, and compare
    against a baseline mean predictor.
    """
    logger.info("Loading latest model ...")
    model_path = resolve_latest_model()
    model = joblib.load(model_path)

    metadata        = get_model_metadata(get_latest_model_hash())
    features        = metadata["features"]
    train_end_month = metadata.get("train_end_month")

    logger.info("Loading validation data ...")
    df = pd.read_parquet(resolve_latest())
    regular = df[~df["invoice_no"].str.startswith("C", na=False)]

    if train_end_month is not None:
        # Proper temporal holdout: months strictly after the cutoff
        train_seg = regular[regular["year_month"] <= train_end_month]
        holdout   = regular[regular["year_month"] >  train_end_month]
        holdout   = holdout.dropna(subset=features + [TARGET_COLUMN])
        train_mean = float(train_seg[TARGET_COLUMN].dropna().mean())
        strategy = "temporal"
        logger.info("Temporal holdout (months > %s): %d rows.",
                    train_end_month, len(holdout))
    else:
        # Fallback: Task 2 behavior - last 20% of training data by row order
        regular = regular.dropna(subset=features + [TARGET_COLUMN])
        split = int(len(regular) * 0.8)
        holdout = regular.iloc[split:]
        train_mean = float(regular[TARGET_COLUMN].iloc[:split].mean())
        strategy = "iloc"
        logger.info("Iloc holdout (last 20%%): %d rows.", len(holdout))

    if len(holdout) == 0:
        raise RobustnessCheckFailed("No holdout data available.")

    X_test = holdout[features]
    y_test = holdout[TARGET_COLUMN].values.astype(float)

    y_pred = model.predict(X_test)
    model_r2 = _r2_score(y_test, y_pred)

    baseline_pred = np.full_like(y_test, train_mean, dtype=float)
    baseline_r2 = _r2_score(y_test, baseline_pred)

    results = {
        "model_path":       str(model_path),
        "holdout_strategy": strategy,
        "test_rows":        len(holdout),
        "model_r2":         float(model_r2),
        "baseline_r2":      float(baseline_r2),
        "passed":           model_r2 >= baseline_r2,
    }

    logger.info("Model R^2:                    %.4f", results["model_r2"])
    logger.info("Baseline R^2 (mean predictor): %.4f", results["baseline_r2"])

    if not results["passed"]:
        raise RobustnessCheckFailed(
            f"Model R^2 ({model_r2:.4f}) did not beat baseline ({baseline_r2:.4f}). "
            f"The model has not learned anything useful from the features."
        )

    logger.info("Robustness check PASSED.")
    return results


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R^2 = 1 - SS_res / SS_tot."""
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# Entry point for direct execution (used by Docker step 3)

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    try:
        validate_model()
        sys.exit(0)
    except RobustnessCheckFailed as e:
        logger.error("Robustness check failed: %s", e)
        sys.exit(1)
