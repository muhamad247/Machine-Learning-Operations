"""
Tests for missing / null values.

Cancelled orders (invoice_no starting with 'C') are not included,
because they are structurally different — they often lack a description,
because the cancellation record only references the original invoice. 
Including them would inflate the null rate and make the threshold meaningless.

Every sold item must have:
  description
  quantity    
  price       
  invoice_date

Thresholds were derived in two steps:

1. We computed the actual null rate for each column
   on the full dataset before writing any tests:
     description  : 0.41% null  → threshold set to 1%
     quantity     : 0.00% null  → threshold set to 0% 
     price        : 0.00% null  → threshold set to 0% 
     invoice_date : 0.00% null  → threshold set to 0% 

2. Quantity, price and invoice_date get zero tolerance
   regardless of observed rate because they are recorded automatically
   by the warehouse system. Any null means a system failure, not natural
   variation. Description gets a small buffer because it is entered
   manually and minor gaps are expected.
"""

import pandas as pd
import pytest

from src.data_loader import resolve_latest


# Fixture — loaded once per test file for performance

@pytest.fixture(scope="module")
def regular_orders() -> pd.DataFrame:
    """
    Load the latest versioned Parquet and return only non-cancelled orders.

    scope="module" means the ~1M row file is read once, not once per test.
    """
    df = pd.read_parquet(resolve_latest())
    return df[~df["invoice_no"].str.startswith("C", na=False)].reset_index(drop=True)


# Tests

class TestNullValues:

    def test_description_null_rate(self, regular_orders):
        """
        At most 1% of description values may be null.

        Reasoning: description feeds warehouse picking lists and customer
        invoices. Missing descriptions signal upstream data-entry failures.
        """
        null_rate = regular_orders["description"].isna().mean()
        assert null_rate <= 0.01, (
            f"description null rate is {null_rate:.2%}, expected ≤ 1%."
        )

    def test_quantity_no_nulls(self, regular_orders):
        """
        quantity must have zero null values.

        Reasoning: quantity is recorded by the warehouse management system
        at the moment of dispatch. A null here means a shipment was logged
        without a unit count.
        """
        null_count = regular_orders["quantity"].isna().sum()
        assert null_count == 0, (
            f"quantity has {null_count} null values — critical integrity gap."
        )

    def test_price_no_nulls(self, regular_orders):
        """
        price must have zero null values.

        Reasoning: unit price is captured from the product catalogue at invoice
        creation. A null price makes it impossible to calculate invoice totals
        and corrupts all revenue metrics downstream.
        """
        null_count = regular_orders["price"].isna().sum()
        assert null_count == 0, (
            f"price has {null_count} null values — invoice totals would be wrong."
        )

    def test_invoice_date_no_nulls(self, regular_orders):
        """
        invoice_date must have zero null values.

        Reasoning: invoice_date is the primary time-series index. A null
        timestamp means the row cannot be placed on the timeline and breaks
        all temporal operations including trend analysis and time-window tests.
        """
        null_count = regular_orders["invoice_date"].isna().sum()
        assert null_count == 0, (
            f"invoice_date has {null_count} null values — time series broken."
        )
