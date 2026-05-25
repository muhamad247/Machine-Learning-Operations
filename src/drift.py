"""
Drift monitoring flow - Task 3 part 1.

Independent flow from the training pipeline. Loads the reference segment
(data the model was trained on) and the unseen segment (data after the
training cutoff) and tests whether feature distributions have shifted.

Run with:
    py -m src.drift --run-id <flow_run_id>

The run_id is a flow registry id whose config contains a train_end_month.
The drift test then compares months <= cutoff (reference) against months
> cutoff (unseen).


Choice of test:
  * PSI (Population Stability Index) for the numeric `price` feature.
  * Chi-square goodness-of-fit for the categorical `country` feature.

Why PSI for price:
  PSI is the standard tabular-drift metric in industry. It is symmetric to
  KL divergence in spirit but easier to interpret and act on. The widely
  accepted thresholds are:
       PSI < 0.10            ->  no significant shift
       0.10 <= PSI < 0.25    ->  moderate shift, monitor
       PSI >= 0.25           ->  significant shift, investigate / retrain
  We use these thresholds. The reference distribution is binned into
  deciles (equal-frequency by quantile), and we compare the population
  share that falls into each bin in the unseen segment. A small epsilon
  (1e-6) is added to avoid log(0) when a bin is empty in one segment.

Why Chi-square for country:
  Country is categorical with a long tail. Chi-square tests whether the
  observed counts in the unseen segment match the expected counts from
  the reference distribution. We use significance level alpha = 0.05.
  Categories with expected count < 5 in the unseen segment get folded
  into an "OTHER" bucket so the chi-square approximation stays valid.

What is the expected behavior:
  * On price we expect at most moderate drift. Some seasonal and
    inflation-related movement is normal in a retail dataset, but a
    significant shift (PSI >= 0.25) would suggest pricing strategy
    changed, a new product mix, or a data-quality issue upstream.
  * On country we expect a stable distribution. The customer base of an
    established UK retailer should not change radically month-to-month.
    A flagged drift here typically means a new market was opened, a
    geo-lookup upstream broke, or the dataset got mixed with another
    source.

Why this segment:
  The unseen segment is the rows with year_month > train_end_month from
  the flow run's config. By construction the model never saw these rows
  during training. Using them for drift testing measures "did the world
  change after we trained?" which is exactly the question this flow is
  meant to answer.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.flow_registry import get_run
from src.segments import load_segments

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("drift_reports")

# Thresholds
PSI_MODERATE    = 0.10
PSI_SIGNIFICANT = 0.25
CHI2_ALPHA      = 0.05


# Step 1: compute drift metrics

def step_1_run_drift_tests(run_id: str) -> dict:
    logger.info("=" * 70)
    logger.info("DRIFT STEP 1: Compute drift on reference vs unseen segment")
    logger.info("=" * 70)

    run = get_run(run_id)
    train_end_month = run["config"].get("train_end_month")
    if not train_end_month:
        raise ValueError(
            f"Flow run '{run_id}' has no train_end_month - drift test "
            f"needs a temporal cutoff to define the unseen segment."
        )

    reference, unseen = load_segments(train_end_month)
    logger.info("Reference rows: %d  |  Unseen rows: %d",
                len(reference), len(unseen))

    if len(unseen) == 0:
        raise ValueError(
            f"Unseen segment is empty - cutoff '{train_end_month}' is at or "
            f"after the last month in the dataset."
        )

    price_psi   = _compute_psi(reference["price"].dropna(),
                               unseen["price"].dropna())
    country_chi = _compute_chi_square(reference["country"], unseen["country"])

    results = {
        "run_id":          run_id,
        "train_end_month": train_end_month,
        "reference_rows":  len(reference),
        "unseen_rows":     len(unseen),
        "price_psi": {
            "value":          price_psi,
            "moderate_threshold":    PSI_MODERATE,
            "significant_threshold": PSI_SIGNIFICANT,
            "verdict":        _psi_verdict(price_psi),
        },
        "country_chi_square": country_chi,
    }

    logger.info("price PSI = %.4f  (%s)",
                price_psi, results["price_psi"]["verdict"])
    logger.info("country chi2 p-value = %.4f  (%s)",
                country_chi["p_value"], country_chi["verdict"])

    return results


# Step 2: persist the report

def step_2_persist_report(results: dict) -> Path:
    logger.info("=" * 70)
    logger.info("DRIFT STEP 2: Persist drift report")
    logger.info("=" * 70)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = REPORTS_DIR / f"drift_{results['run_id']}_{timestamp}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Drift report saved -> %s", out)
    return out


# Drift metric implementations

def _compute_psi(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    """
    Population Stability Index between two numeric distributions.

    Bin edges come from the reference quantiles (equal-frequency). We then
    compare the share of each bin in the current vs the reference and sum
    the log contributions. eps avoids log(0) on empty bins.
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        return 0.0
    edges = edges.astype(float)
    edges[0]  = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    eps = 1e-6
    ref_pct = ref_counts / max(len(reference), 1) + eps
    cur_pct = cur_counts / max(len(current), 1)  + eps

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _psi_verdict(psi: float) -> str:
    if psi < PSI_MODERATE:
        return "no_significant_drift"
    if psi < PSI_SIGNIFICANT:
        return "moderate_drift"
    return "significant_drift"


def _compute_chi_square(reference: pd.Series, current: pd.Series) -> dict:
    """
    Chi-square goodness-of-fit on the country distribution.

    Categories with fewer than 5 expected observations in the unseen
    segment get folded into 'OTHER' so the chi-square approximation
    remains valid.
    """
    all_cats = sorted(set(reference.dropna().unique()) | set(current.dropna().unique()))
    ref_counts = reference.value_counts().reindex(all_cats, fill_value=0)
    cur_counts = current.value_counts().reindex(all_cats, fill_value=0)

    n_ref = ref_counts.sum()
    n_cur = cur_counts.sum()
    if n_ref == 0 or n_cur == 0:
        return {
            "chi2_stat":    None,
            "p_value":      None,
            "n_categories": 0,
            "alpha":        CHI2_ALPHA,
            "verdict":      "inconclusive",
        }

    expected = (ref_counts / n_ref) * n_cur

    # Fold rare expected categories into one bucket
    rare = expected < 5
    if rare.any():
        rare_obs = float(cur_counts[rare].sum())
        rare_exp = float(expected[rare].sum())
        observed_kept = cur_counts[~rare].astype(float).copy()
        expected_kept = expected[~rare].astype(float).copy()
        if rare_exp >= 5:
            observed_kept["OTHER"] = rare_obs
            expected_kept["OTHER"] = rare_exp
    else:
        observed_kept = cur_counts.astype(float).copy()
        expected_kept = expected.astype(float).copy()

    # Rescale expected so it has the same total as observed (chisquare needs this)
    total_exp = expected_kept.sum()
    if total_exp > 0:
        expected_kept = expected_kept * (observed_kept.sum() / total_exp)

    chi2, p_value = stats.chisquare(
        f_obs=observed_kept.values,
        f_exp=expected_kept.values,
    )

    return {
        "chi2_stat":    float(chi2),
        "p_value":      float(p_value),
        "n_categories": int(len(observed_kept)),
        "alpha":        CHI2_ALPHA,
        "verdict":      "drift_detected" if p_value < CHI2_ALPHA else "no_drift",
    }


# Entry point

def main():
    parser = argparse.ArgumentParser(description="Drift monitoring flow")
    parser.add_argument("--run-id", required=True,
                        help="Flow run id to drift-test against.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    try:
        results = step_1_run_drift_tests(args.run_id)
        step_2_persist_report(results)
        logger.info("Drift monitoring complete.")
    except Exception as e:
        logger.error("Drift flow failed: %s: %s", type(e).__name__, e)
        raise


if __name__ == "__main__":
    main()
