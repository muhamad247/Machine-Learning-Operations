"""
A/B test flow - Task 3 part 3.

Takes two flow run ids, resolves each to its trained model via the flow
registry, and runs predictions on the unseen segment. The unseen segment
is split "evenly, randomly, reproducibly" by hashing each row's
customer_id with a salt that includes the A/B test_id, then taking mod 2:

    bucket 0 -> variant A (model from run-a)
    bucket 1 -> variant B (model from run-b)

Run with:

    py -m src.ab_test --run-a <run_id_v1> \
                      --run-b <run_id_v2> \
                      --test-id ab-001


Why hash(customer_id + test_id) % 2:
  * Reproducible without storing an assignment table: same (customer, test)
    -> same bucket forever.
  * Roughly 50/50 by the uniformity of MD5 over enough customers.
  * The test_id salt isolates concurrent A/B tests from each other - the
    same customer can land in bucket A for ab-001 and bucket B for ab-002.
  * Done at the customer level (not the row level), so the same customer
    is never sent to both models in the same test. Customers usually have
    multiple invoice lines and that consistency matters for real A/B logic.

Rows with missing customer_id cannot be deterministically assigned and
are dropped. The drop count is logged and saved in the result.


Performance metric:
  Mean Absolute Error (MAE) on the bucket predictions. We chose MAE over
  R^2 because:
    * It is directly interpretable in the units of the target (units per
      order line).
    * It is more robust than R^2 to the heavy-tailed distribution of
      quantity in this dataset.
  R^2 is also reported for context.


Multiple concurrent or subsequent A/B tests - how to handle them:

Each test gets a test_id (CLI argument). The test_id is used as the salt
in the customer hash, so different tests assign the same customer base to
different buckets independently of each other. Results from each test go
to a separate JSON file under ab_tests/<test_id>.json, so concurrent
tests don't overwrite each other.

In a real production setting we would also:
  * Persist a customer-to-bucket assignment table per test_id (so that
    later analysis can reconstruct who was in which bucket even if the
    hash function changes).
  * Maintain a registry of active tests, with status (running / stopped),
    start and end times, and a target sample size per bucket, so multiple
    tests can be inspected together and old tests retired cleanly.

For an offline reproducible setup, the salted-hash approach is sufficient
and stateless.


Both flow runs must share a train_end_month so they are evaluated on the
same unseen segment. If they don't, the flow stops with a clear error.
"""

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.flow_registry import get_run, resolve_model_hash
from src.segments import load_segments
from src.train import load_model, get_model_metadata, TARGET_COLUMN

logger = logging.getLogger(__name__)

AB_DIR = Path("ab_tests")


# Step 1: resolve flow runs to models

def step_1_load_models(run_id_a: str, run_id_b: str) -> dict:
    logger.info("=" * 70)
    logger.info("A/B STEP 1: Resolve flow runs and load models")
    logger.info("=" * 70)

    run_a = get_run(run_id_a)
    run_b = get_run(run_id_b)

    end_a = run_a["config"].get("train_end_month")
    end_b = run_b["config"].get("train_end_month")
    if end_a != end_b:
        raise ValueError(
            f"Flow runs use different train_end_month "
            f"({end_a!r} vs {end_b!r}). Pick two runs that share a cutoff "
            f"so they are evaluated on the same unseen segment."
        )
    if end_a is None:
        raise ValueError(
            "Both runs must have a train_end_month defined. Re-run the "
            "training flow with --train-end YYYY-MM to create a proper "
            "unseen segment for the A/B test."
        )

    model_id_a = resolve_model_hash(run_id_a)
    model_id_b = resolve_model_hash(run_id_b)

    logger.info("Variant A: run %s -> model %s", run_id_a, model_id_a)
    logger.info("Variant B: run %s -> model %s", run_id_b, model_id_b)

    return {
        "run_a":           run_a,
        "run_b":           run_b,
        "model_a":         load_model(model_id_a),
        "model_b":         load_model(model_id_b),
        "meta_a":          get_model_metadata(model_id_a),
        "meta_b":          get_model_metadata(model_id_b),
        "train_end_month": end_a,
    }


# Step 2: split unseen data, predict per branch, score

def step_2_predict(test_id: str, ctx: dict) -> dict:
    logger.info("=" * 70)
    logger.info("A/B STEP 2: Allocate buckets and predict")
    logger.info("=" * 70)

    _, unseen = load_segments(ctx["train_end_month"])
    dropped = int(unseen["customer_id"].isna().sum())
    unseen  = unseen.dropna(subset=["customer_id"])

    buckets = unseen["customer_id"].apply(
        lambda c: _bucket(c, test_id, n_buckets=2)
    )
    bucket_a = unseen[buckets == 0]
    bucket_b = unseen[buckets == 1]
    logger.info(
        "Bucket A: %d rows  |  Bucket B: %d rows  |  Dropped (no customer_id): %d",
        len(bucket_a), len(bucket_b), dropped,
    )

    metrics_a = _predict_and_score(ctx["model_a"], ctx["meta_a"], bucket_a)
    metrics_b = _predict_and_score(ctx["model_b"], ctx["meta_b"], bucket_b)

    return {
        "test_id":         test_id,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "train_end_month": ctx["train_end_month"],
        "dropped_no_customer_id": dropped,
        "branch_a": {
            "run_id":     ctx["run_a"]["run_id"],
            "run_name":   ctx["run_a"]["config"].get("run_name"),
            "model_type": ctx["meta_a"].get("model_type"),
            "features":   ctx["meta_a"].get("features"),
            "rows":       len(bucket_a),
            **metrics_a,
        },
        "branch_b": {
            "run_id":     ctx["run_b"]["run_id"],
            "run_name":   ctx["run_b"]["config"].get("run_name"),
            "model_type": ctx["meta_b"].get("model_type"),
            "features":   ctx["meta_b"].get("features"),
            "rows":       len(bucket_b),
            **metrics_b,
        },
        "winner": _pick_winner(metrics_a, metrics_b),
    }


# Step 3: persist result

def step_3_persist(results: dict) -> Path:
    logger.info("=" * 70)
    logger.info("A/B STEP 3: Persist results")
    logger.info("=" * 70)

    AB_DIR.mkdir(parents=True, exist_ok=True)
    out = AB_DIR / f"{results['test_id']}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    logger.info("A/B test result saved -> %s", out)
    logger.info("Winner: %s", results["winner"])
    return out


# Helpers

def _bucket(customer_id, test_id: str, n_buckets: int) -> int:
    """
    Deterministic bucket assignment from customer_id, salted with test_id.

    Same (customer_id, test_id) -> same bucket, every time.
    Different test_id -> independent allocation for the same customer.
    """
    key = f"{test_id}::{customer_id}".encode()
    h = hashlib.md5(key).hexdigest()
    return int(h, 16) % n_buckets


def _predict_and_score(model, meta: dict, df: pd.DataFrame) -> dict:
    features = meta["features"]
    df = df.dropna(subset=features + [TARGET_COLUMN])
    if df.empty:
        return {"mae": None, "r2": None, "note": "empty bucket"}

    y_true = df[TARGET_COLUMN].values.astype(float)
    y_pred = model.predict(df[features])

    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2  = _r2(y_true, y_pred)
    return {"mae": mae, "r2": r2}


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _pick_winner(a: dict, b: dict) -> str:
    """
    Lightweight winner pick by MAE - not a statistical significance test
    (the assignment explicitly does not require one).
    """
    if a["mae"] is None and b["mae"] is None:
        return "inconclusive"
    if a["mae"] is None:
        return "B"
    if b["mae"] is None:
        return "A"
    rel_gap = abs(a["mae"] - b["mae"]) / max(a["mae"], b["mae"])
    if rel_gap < 0.01:
        return "tie"
    return "A" if a["mae"] < b["mae"] else "B"


# Entry point

def main():
    parser = argparse.ArgumentParser(description="Offline A/B test flow")
    parser.add_argument("--run-a",   required=True,
                        help="Flow run id for variant A.")
    parser.add_argument("--run-b",   required=True,
                        help="Flow run id for variant B.")
    parser.add_argument("--test-id", default="ab-001",
                        help="Unique id for this A/B test, used as the "
                             "salt in the customer_id hash. Use a "
                             "different test-id for each concurrent A/B "
                             "test you run.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    try:
        ctx = step_1_load_models(args.run_a, args.run_b)
        results = step_2_predict(args.test_id, ctx)
        step_3_persist(results)
        logger.info("A/B test complete.")
    except Exception as e:
        logger.error("A/B flow failed: %s: %s", type(e).__name__, e)
        raise


if __name__ == "__main__":
    main()
