"""
Trains a linear regression model to predict quantity from price and country.

We predict quantity because it's the natural business question for a wholesale
retailer: at a given price point and country, how many units does a customer
typically buy? It's a regression task with a continuous target.

The assignment recommends a sensible serialization format better than pickle.
Joblib is the standard for sklearn models. It's faster than pickle for
numpy arrays and is the official recommendation from scikit-learn.

We enforce a minimum training data size of 1000 rows. If the segment is
smaller than this we raise a clear error and stop. This prevents silently
training on too little data which would produce an unreliable model.
The minimum is configurable via the MIN_ROWS constant so different scenarios
can be tested without changing the function signature.
"""

import hashlib
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.data_loader import resolve_latest

logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = Path("models")
MODEL_MANIFEST = MODELS_DIR / "model_manifest.json"

# Training configuration
MIN_ROWS = 1000          # induced error threshold
TARGET_COLUMN = "quantity"
FEATURE_COLUMNS = ["price", "country"]

# Schema for downstream validation
INPUT_SCHEMA = {
    "price":   "float",
    "country": "string",
}
OUTPUT_SCHEMA = {
    "quantity": "float",
}


# Custom error class for training failures

class InsufficientDataError(Exception):
    """Raised when the training segment has fewer rows than MIN_ROWS."""
    pass


# Main training function

def train_model(simulate_small_dataset: bool = False) -> Path:
    """
    Load data, train a linear regression model, save it as a versioned artifact.

    simulate_small_dataset : bool
        If True, artificially shrink the training set to test the error
        handling. Used to demonstrate the induced error scenario from the
        assignment.

    Returns path to the saved model file.
    Otherwise InsufficientDataError if the training segment has fewer than MIN_ROWS rows.
    """
    logger.info("Loading versioned dataset ...")
    df = pd.read_parquet(resolve_latest())

    # Filter to non-cancelled orders only — same segment as our quality tests
    regular = df[~df["invoice_no"].str.startswith("C", na=False)]

    # Drop any rows with missing values in our features or target
    train_df = regular.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    # Optionally simulate a small dataset to demonstrate error handling
    if simulate_small_dataset:
        logger.warning("Simulating small dataset (taking only 500 rows).")
        train_df = train_df.head(500)

    # Induced error check — fail clearly if data is too small
    if len(train_df) < MIN_ROWS:
        raise InsufficientDataError(
            f"Training data has only {len(train_df)} rows, "
            f"minimum required is {MIN_ROWS}. "
            f"Cannot train a reliable model on so little data."
        )

    logger.info("Training on %d rows.", len(train_df))

    X = train_df[FEATURE_COLUMNS]
    y = train_df[TARGET_COLUMN]

    # Pipeline: one-hot encode country, leave price as is, then linear regression
    preprocessor = ColumnTransformer(
        transformers=[
            ("country_ohe", OneHotEncoder(handle_unknown="ignore"), ["country"]),
        ],
        remainder="passthrough",
    )
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    model.fit(X, y)
    logger.info("Model trained. R² on training data: %.4f", model.score(X, y))

    return _save_versioned_model(model, train_df)


# Versioning helpers

def _save_versioned_model(model: Pipeline, train_df: pd.DataFrame) -> Path:
    """
    Save the model with a content hash filename and update the manifest.

    Same versioning approach as the dataset: deterministic filename, all
    versions preserved, latest tracked in a manifest.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Hash the serialized model bytes for a deterministic version id
    buf = io.BytesIO()
    joblib.dump(model, buf)
    model_hash = hashlib.sha256(buf.getvalue()).hexdigest()[:12]
    model_path = MODELS_DIR / f"model_{model_hash}.joblib"

    if not model_path.exists():
        with open(model_path, "wb") as f:
            f.write(buf.getvalue())
        logger.info("Saved model → %s", model_path)
    else:
        logger.info("Model hash already exists — no duplicate written.")

    # Update the model manifest with metadata
    manifest = _load_manifest()
    manifest["versions"][model_hash] = {
        "path":            str(model_path),
        "created_at":      datetime.now(timezone.utc).isoformat(),
        "training_rows":   len(train_df),
        "input_schema":    INPUT_SCHEMA,
        "output_schema":   OUTPUT_SCHEMA,
        "training_score":  float(model.score(train_df[FEATURE_COLUMNS],
                                             train_df[TARGET_COLUMN])),
        "dependencies":    ["scikit-learn", "pandas", "joblib"],
        "model_type":      "LinearRegression with OneHotEncoder for country",
    }
    manifest["latest"] = model_hash
    _save_manifest(manifest)

    return model_path


def resolve_latest_model() -> Path:
    """Return the path to the most recently trained model."""
    manifest = _load_manifest()
    if not manifest.get("versions"):
        raise FileNotFoundError(
            "No trained model found. Run `py -m src.flow` first."
        )
    return Path(manifest["versions"][manifest["latest"]]["path"])


def list_models() -> list:
    """List all available models with their metadata."""
    manifest = _load_manifest()
    return [
        {"id": h, **meta}
        for h, meta in manifest.get("versions", {}).items()
    ]


def load_model(model_id: str) -> Pipeline:
    """Load a specific model by its id."""
    manifest = _load_manifest()
    if model_id not in manifest.get("versions", {}):
        raise KeyError(f"No model with id '{model_id}'.")
    return joblib.load(manifest["versions"][model_id]["path"])


def _load_manifest() -> dict:
    if MODEL_MANIFEST.exists():
        return json.loads(MODEL_MANIFEST.read_text())
    return {"latest": None, "versions": {}}


def _save_manifest(manifest: dict) -> None:
    MODEL_MANIFEST.write_text(json.dumps(manifest, indent=2))


# Entry point for direct execution (used by Docker step 2)

if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    # SIMULATE_ERROR=1 environment variable triggers the small dataset scenario
    simulate = os.environ.get("SIMULATE_ERROR", "0") == "1"

    try:
        train_model(simulate_small_dataset=simulate)
        sys.exit(0)
    except InsufficientDataError as e:
        logger.error("Training failed: %s", e)
        sys.exit(1)