"""
Tests for cancelled orders as a separate segment.

Cancelled orders (invoice_no starting with 'C') are structurally different
from regular orders. We test them separately to verify they follow 
the expected cancellation pattern.

A cancellation record is a reversal of an original order, so:
  - quantities must be negative (reversing the original positive quantity)
  - invoice_no must always start with 'C'
  - price must be positive (same unit price as the original order)

If any of these fail it means the cancellation data is corrupted or
something other than cancellations ended up in this segment.
"""

import pandas as pd
import pytest

from src.data_loader import resolve_latest


# Fixture

@pytest.fixture(scope="module")
def cancelled_orders() -> pd.DataFrame:
    df = pd.read_parquet(resolve_latest())
    return df[df["invoice_no"].str.startswith("C", na=False)].reset_index(drop=True)


class TestCancelledOrders:

    def test_cancelled_quantities_are_negative(self, cancelled_orders):
        """
        At least 99% of cancelled order quantities must be negative.

        Reasoning: a cancellation reverses the original order, so the quantity
        should always be negative. We allow 1% tolerance for edge cases like
        manual corrections that may have been logged as cancellations.
        """
        negative_rate = (cancelled_orders["quantity"] < 0).mean()
        assert negative_rate >= 0.99, (
            f"Only {negative_rate:.2%} of cancelled quantities are negative, "
            f"expected ≥ 99%."
        )

    def test_cancelled_prices_are_positive(self, cancelled_orders):
        """
        At least 99% of cancelled order prices must be positive.

        Reasoning: the price in a cancellation row reflects the unit price of
        the original order, so it should be the same positive value. A negative
        or zero price on a cancellation would mean the record is malformed.
        """
        positive_rate = (cancelled_orders["price"] > 0).mean()
        assert positive_rate >= 0.99, (
            f"Only {positive_rate:.2%} of cancelled prices are positive, "
            f"expected ≥ 99%."
        )

    def test_cancelled_invoice_nos_start_with_c(self, cancelled_orders):
        """
        All cancelled order invoice numbers must start with 'C'.

        Reasoning: this is the definition of a cancellation record in this
        dataset. If any row in this segment does not start with 'C' it means
        our segmentation logic is broken.
        """
        c_rate = cancelled_orders["invoice_no"].str.startswith("C", na=False).mean()
        assert c_rate == 1.0, (
            f"Only {c_rate:.2%} of cancelled invoices start with 'C', "
            f"expected 100%."
        )
