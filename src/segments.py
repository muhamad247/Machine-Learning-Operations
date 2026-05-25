"""
Load reference (training) and unseen segments for Task 3 flows.

Both the drift monitor and the A/B test work on data that was NOT seen
during training. We split on year_month using the same cutoff that
train_model used internally, so the unseen segment is exactly the rows
the model never saw.

Why year_month and not a random hold-out:
  A random hold-out leaks future signal into the training segment, which
  defeats the purpose of a drift test. A temporal split mirrors how this
  pipeline would actually be used in production - train on history,
  monitor on what comes next.
"""

from typing import Tuple

import pandas as pd

from src.data_loader import resolve_latest


def load_segments(train_end_month: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (reference_segment, unseen_segment) split by year_month.

    Both segments restrict to non-cancelled orders to match the training
    population. Rows with year_month <= train_end_month go into reference,
    everything after goes into unseen.
    """
    df = pd.read_parquet(resolve_latest())
    regular = df[~df["invoice_no"].str.startswith("C", na=False)]
    reference = regular[regular["year_month"] <= train_end_month]
    unseen    = regular[regular["year_month"] >  train_end_month]
    return reference, unseen
