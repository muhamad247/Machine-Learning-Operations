"""
Self-written orchestrator that runs the full ML pipeline as 3 sequential steps:

  Step 1 — Run all data quality tests from Task 1
  Step 2 — Train and version the model (with induced error handling)
  Step 3 — Validate model robustness against a baseline

Each step isnisolated as its own function — if any step fails the pipeline stops.
"""

import argparse
import logging
import subprocess
import sys

from src.train import train_model, InsufficientDataError
from src.validate import validate_model, RobustnessCheckFailed

logger = logging.getLogger(__name__)


# Step 1 — run pytest on all data quality tests

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
            "Data quality tests failed. Stopping the pipeline — we do not "
            "train a model on data that does not meet our quality standards."
        )
    logger.info("Step 1 complete: all data tests passed.\n")


# Step 2 — train and version the model

def step_2_train(simulate_error: bool = False) -> None:
    logger.info("=" * 70)
    logger.info("STEP 2: Training and versioning the model")
    logger.info("=" * 70)

    try:
        model_path = train_model(simulate_small_dataset=simulate_error)
        logger.info("Step 2 complete: model saved to %s\n", model_path)
    except InsufficientDataError as e:
        logger.error("Training failed due to insufficient data: %s", e)
        logger.error(
            "The pipeline stops here. To recover, either (a) provide more "
            "data, or (b) lower the MIN_ROWS threshold in src/train.py."
        )
        raise


# Step 3 — robustness validation

def step_3_validate() -> None:
    logger.info("=" * 70)
    logger.info("STEP 3: Validating model robustness")
    logger.info("=" * 70)

    try:
        results = validate_model()
        logger.info("Step 3 complete: robustness check passed.")
        logger.info("Results: %s\n", results)
    except RobustnessCheckFailed as e:
        logger.error("Robustness check failed: %s", e)
        raise


# Entry point

def main():
    parser = argparse.ArgumentParser(description="ML Ops Task 2 pipeline")
    parser.add_argument(
        "--simulate-error",
        action="store_true",
        help="Simulate the induced training error (small dataset).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    try:
        step_1_data_tests()
        step_2_train(simulate_error=args.simulate_error)
        step_3_validate()
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error("Pipeline stopped: %s", type(e).__name__)
        sys.exit(1)


if __name__ == "__main__":
    main()
