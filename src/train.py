"""
Trains a regression model to predict quantity from price and country by default. 
(See the Task 3 section below for the parametrized version with additional 
features and model types.)

We predict quantity because it's the natural business question for a wholesale
retailer: at a given price point and country, how many units does a customer
typically buy? It's a regression task with a continuous target.

The assignment recommends a sensible serialization format better than pickle.
Joblib is the standard for sklearn models. It's faster than pickle for numpy 
arrays and is the official recommendation from scikit-learn.

We enforce a minimum training data size of 1000 rows. If the segment is
smaller than this we raise a clear error and stop. This prevents silently
training on too little data which would produce an unreliable model.
The minimum is configurable via the MIN_ROWS constant so different scenarios
can be tested without changing the function signature.

Parametrized for Task 3 flow versioning:
  - train_end_month  : temporal cutoff. Rows after this month are reserved
                       as the unseen segment for drift and A/B tests. If
                       None, all data is used for training (Task 2 behavior).
  - model_type       : "linear" or "ridge".
  - features         : list of feature columns. Supported: price, country,
                       stock_code, year_month. Categorical columns are
                       one-hot encoded with min_frequency grouping so that
                       rare values (and unknowns at predict time) end up in
                       a single "infrequent" bucket.

The "v2 improvement path" used in Task 3 adds stock_code as a feature with
Ridge regularization. stock_code is the strongest signal for typical order
size in this dataset — wholesale buyers cluster around certain product
codes — so including it should beat the v1 (price + country) baseline.

Models are versioned by content hash with full metadata. The flow registry
links each flow run to the model id it produced, so downstream flows
(drift, A/B) can resolve a flow version to its actual model deterministically.
"""

import hashlib
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.data_loader import resolve_latest

logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = Path("models")
MODEL_MANIFEST = MODELS_DIR / "model_manifest.json"

# Training configuration
MIN_ROWS = 1000
TARGET_COLUMN = "quantity"
DEFAULT_FEATURES = ["price", "country"]
CATEGORICAL_COLUMNS = {"country", "stock_code", "year_month"}

# Backwards-compat for any code still importing the Task 2 constant
FEATURE_COLUMNS = list(DEFAULT_FEATURES)


class InsufficientDataError(Exception):
    """Raised when the training segment has fewer rows than MIN_ROWS."""
    pass


def train_model(
    simulate_small_dataset: bool = False,
    train_end_month: Optional[str] = None,
    model_type: str = "linear",
    features: Optional[List[str]] = None,
    min_category_frequency: int = 50,
    random_state: int = 42,
) -> Path:
    """
    Train a model with the given configuration and save it as a versioned artifact.

    Returns the path to the saved model file. Raises InsufficientDataError
    if the training segment has fewer than MIN_ROWS rows.
    """
    features = list(features) if features else list(DEFAULT_FEATURES)

    logger.info("Loading versioned dataset ...")
    df = pd.read_parquet(resolve_latest())

    # Same population as Task 2: non-cancelled orders only.
    regular = df[~df["invoice_no"].str.startswith("C", na=False)]

    # Apply the temporal cutoff. Anything after it is held out for Task 3.
    if train_end_month is not None:
        before = len(regular)
        regular = regular[regular["year_month"] <= train_end_month]
        logger.info(
            "Temporal cutoff '%s' applied: %d -> %d rows.",
            train_end_month, before, len(regular),
        )

    train_df = regular.dropna(subset=features + [TARGET_COLUMN])

    if simulate_small_dataset:
        logger.warning("Simulating small dataset (taking only 500 rows).")
        train_df = train_df.head(500)

    if len(train_df) < MIN_ROWS:
        raise InsufficientDataError(
            f"Training data has only {len(train_df)} rows, "
            f"minimum required is {MIN_ROWS}. "
            f"Cannot train a reliable model on so little data."
        )

    logger.info("Training on %d rows with features %s (model_type=%s).",
                len(train_df), features, model_type)

    X = train_df[features]
    y = train_df[TARGET_COLUMN]

    # Categorical columns get one-hot encoding; rare values are grouped into
    # one "infrequent" bucket which also catches unknowns at predict time.
    categorical_features = [c for c in features if c in CATEGORICAL_COLUMNS]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=min_category_frequency,
            ), categorical_features),
        ],
        remainder="passthrough",
    )

    if model_type == "ridge":
        regressor = Ridge(alpha=1.0, random_state=random_state)
    elif model_type == "linear":
        regressor = LinearRegression()
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])

    model.fit(X, y)
    train_r2 = float(model.score(X, y))
    logger.info("Model trained. R^2 on training data: %.4f", train_r2)

    return _save_versioned_model(
        model,
        train_df,
        features=features,
        model_type=model_type,
        train_end_month=train_end_month,
        training_score=train_r2,
        min_category_frequency=min_category_frequency,
    )


# Versioning helpers

def _save_versioned_model(
    model: Pipeline,
    train_df: pd.DataFrame,
    *,
    features: List[str],
    model_type: str,
    train_end_month: Optional[str],
    training_score: float,
    min_category_frequency: int,
) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    buf = io.BytesIO()
    joblib.dump(model, buf)
    model_hash = hashlib.sha256(buf.getvalue()).hexdigest()[:12]
    model_path = MODELS_DIR / f"model_{model_hash}.joblib"

    if not model_path.exists():
        with open(model_path, "wb") as f:
            f.write(buf.getvalue())
        logger.info("Saved model -> %s", model_path)
    else:
        logger.info("Model hash already exists - no duplicate written.")

    manifest = _load_manifest()
    manifest["versions"][model_hash] = {
        "path":            str(model_path),
        "created_at":      datetime.now(timezone.utc).isoformat(),
        "training_rows":   len(train_df),
        "features":        features,
        "target":          TARGET_COLUMN,
        "model_type":      model_type,
        "train_end_month": train_end_month,
        "training_score":  training_score,
        "min_category_frequency": min_category_frequency,
        "input_schema":    _infer_input_schema(features),
        "output_schema":   {TARGET_COLUMN: "float"},
        "dependencies":    ["scikit-learn>=1.1", "pandas", "joblib"],
    }
    manifest["latest"] = model_hash
    _save_manifest(manifest)

    return model_path


def _infer_input_schema(features: List[str]) -> dict:
    schema = {}
    for f in features:
        schema[f] = "float" if f == "price" else "string"
    return schema


def get_latest_model_hash() -> str:
    """Return the id of the most recently trained model."""
    manifest = _load_manifest()
    if not manifest.get("latest"):
        raise FileNotFoundError("No trained model found. Run `py -m src.flow` first.")
    return manifest["latest"]


def resolve_latest_model() -> Path:
    """Return the path to the most recently trained model."""
    manifest = _load_manifest()
    if not manifest.get("latest"):
        raise FileNotFoundError("No trained model found. Run `py -m src.flow` first.")
    return Path(manifest["versions"][manifest["latest"]]["path"])


def list_models() -> list:
    """List all available models with their metadata."""
    manifest = _load_manifest()
    return [{"id": h, **meta} for h, meta in manifest.get("versions", {}).items()]


def load_model(model_id: str) -> Pipeline:
    """Load a specific model by its id."""
    manifest = _load_manifest()
    if model_id not in manifest.get("versions", {}):
        raise KeyError(f"No model with id '{model_id}'.")
    return joblib.load(manifest["versions"][model_id]["path"])


def get_model_metadata(model_id: str) -> dict:
    """Return the manifest entry for a model id."""
    manifest = _load_manifest()
    if model_id not in manifest.get("versions", {}):
        raise KeyError(f"No model with id '{model_id}'.")
    return manifest["versions"][model_id]


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

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    simulate = os.environ.get("SIMULATE_ERROR", "0") == "1"

    try:
        train_model(simulate_small_dataset=simulate)
        sys.exit(0)
    except InsufficientDataError as e:
        logger.error("Training failed: %s", e)
        sys.exit(1)
