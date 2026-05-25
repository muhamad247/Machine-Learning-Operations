"""
Self-written orchestrator for Task 2 + Task 3 that runs the full ML pipeline as 3 sequential steps:

  Step 1 - Run all data quality tests from Task 1
  Step 2 - Train and version the model (with induced error handling + parametrized for Task 3)
  Step 3 - Validate model robustness against a baseline

Each step is isolated as its own function. Step 1 or 2 failing stops the
pipeline entirely. Step 3 failing is logged and recorded in the run status
but does not lose the trained model - the run is still registered with its
model_hash so downstream flows (drift, A/B) can resolve it. This separation
matters for Task 3: a model that didn't beat baseline is still a valid
artifact to compare against another version.

For Task 3, the flow accepts CLI arguments that change how training behaves
(temporal cutoff, model type, feature set). Each run is registered in the
flow_registry with its full config and the resulting model id, so the A/B
flow can later resolve a flow run to its actual model.

Example v1 vs v2:

  py -m src.flow --train-end 2011-06 \
                 --features price,country \
                 --model-type linear \
                 --run-name v1-baseline

  py -m src.flow --train-end 2011-06 \
                 --features price,country,stock_code \
                 --model-type ridge \
                 --run-name v2-enriched
"""

import argparse
import logging
import subprocess
import sys

from src.train import (
    train_model,
    get_latest_model_hash,
    InsufficientDataError,
)
from src.validate import validate_model, RobustnessCheckFailed
from src.flow_registry import register_run

logger = logging.getLogger(__name__)


# Step 1 - run pytest on all data quality tests

def step_1_data_tests() -> None:
    logger.info("=" * 70)
    logger.info("STEP 1: Running pre-training data quality tests")
    logger.info("=" * 70)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Data quality tests failed. Stopping the pipeline - we do not "
            "train a model on data that does not meet our quality standards."
        )
    logger.info("Step 1 complete: all data tests passed.\n")


# Step 2 - train and version the model

def step_2_train(*, simulate_error, train_end_month, model_type, features) -> str:
    logger.info("=" * 70)
    logger.info("STEP 2: Training and versioning the model")
    logger.info("=" * 70)

    try:
        model_path = train_model(
            simulate_small_dataset=simulate_error,
            train_end_month=train_end_month,
            model_type=model_type,
            features=features,
        )
        model_hash = get_latest_model_hash()
        logger.info("Step 2 complete: model saved to %s (id %s)\n",
                    model_path, model_hash)
        return model_hash
    except InsufficientDataError as e:
        logger.error("Training failed due to insufficient data: %s", e)
        logger.error(
            "The pipeline stops here. To recover, either (a) provide more "
            "data, or (b) lower the MIN_ROWS threshold in src/train.py."
        )
        raise


# Step 3 - robustness validation

def step_3_validate() -> None:
    logger.info("=" * 70)
    logger.info("STEP 3: Validating model robustness")
    logger.info("=" * 70)
    results = validate_model()
    logger.info("Step 3 complete: robustness check passed.")
    logger.info("Results: %s\n", results)


# Entry point

def main():
    parser = argparse.ArgumentParser(description="ML Ops training pipeline")
    parser.add_argument(
        "--simulate-error", action="store_true",
        help="Simulate the induced training error (small dataset).",
    )
    parser.add_argument(
        "--train-end", default=None, metavar="YYYY-MM",
        help="Temporal cutoff for training data. Months after this are "
             "reserved as the unseen segment for Task 3 drift and A/B tests.",
    )
    parser.add_argument(
        "--model-type", default="linear", choices=["linear", "ridge"],
        help="Regressor choice. Ridge adds L2 regularization, helpful when "
             "more features are included.",
    )
    parser.add_argument(
        "--features", default="price,country",
        help="Comma-separated feature columns. Supported: price, country, "
             "stock_code, year_month.",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Optional human-readable name for this flow run.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    features = [f.strip() for f in args.features.split(",") if f.strip()]

    config = {
        "run_name":        args.run_name,
        "train_end_month": args.train_end,
        "model_type":      args.model_type,
        "features":        features,
        "simulate_error":  args.simulate_error,
    }

    # Step 1 and 2 failing means we have no model - register as failed and exit.
    try:
        step_1_data_tests()
        model_hash = step_2_train(
            simulate_error=args.simulate_error,
            train_end_month=args.train_end,
            model_type=args.model_type,
            features=features,
        )
    except Exception as e:
        logger.error("Pipeline stopped before model was trained: %s: %s",
                     type(e).__name__, e)
        register_run(config=config, model_hash=None,
                     status="failed", error=str(e))
        sys.exit(1)

    # Step 2 succeeded - the model exists. Run step 3 but keep the model_hash
    # in the run record regardless of the outcome.
    status = "passed"
    error_msg = None
    try:
        step_3_validate()
    except RobustnessCheckFailed as e:
        logger.warning(
            "Robustness check failed but the model was trained successfully. "
            "Registering the run with status 'robustness_failed' - the model "
            "is still available for downstream comparison (drift, A/B)."
        )
        status = "robustness_failed"
        error_msg = str(e)

    run_id = register_run(config=config, model_hash=model_hash,
                          status=status, error=error_msg)

    logger.info("Pipeline complete.")
    logger.info("Flow run id: %s", run_id)
    logger.info("Model id:    %s", model_hash)
    logger.info("Status:      %s", status)

    # Exit non-zero on robustness failure so CI / scripts can still detect it
    if status != "passed":
        sys.exit(2)


if __name__ == "__main__":
    main()