"""
Tests for attribute value distributions (price and country)

These two columns represent the most business-critical dimensions:
  price   → directly affects revenue calculations
  country → defines the geographic scope of the business

For price we check the lower end, upper end and the overall shape.
For country we check the dominant value, if hte known markets exist and the unknown countries

Choosing ID-like columns (invoice_no, stock_code) would be 
pointless since their distribution carries no business meaning.

Like in the test_null_values, cancelled orders are not included for the same reasons.

Thresholds were derived looking at the actual data distribution 
before writing any tests and using business knowledge.
"""

import pandas as pd
import pytest

from src.data_loader import resolve_latest


# Fixture

@pytest.fixture(scope="module")
def regular_orders() -> pd.DataFrame:
    df = pd.read_parquet(resolve_latest())
    return df[~df["invoice_no"].str.startswith("C", na=False)].reset_index(drop=True)


# Attribute 1: price

class TestPriceDistribution:
    """
    Distribution checks for unit price (GBP).

      5th percentile  = £0.42
      median          = £2.10
      95th percentile = £9.95
    
    Values above £5000 do exist in the data but they turned out to be internal
    accounting entries (Amazon fees, bad debt adjustments, manual corrections)
    and not real product prices.
    """

    def test_price_mostly_positive(self, regular_orders):
        """
        At least 98% of price values must be greater than £0.

        Reasoning: a unit price of £0 on a non-cancelled order means an item
        was shipped for free. This can happen for samples but should be rare.
        More than 2% would indicate a systematic pricing error corrupting
        revenue calculations.
        """
        positive_rate = (regular_orders["price"] > 0).mean()
        assert positive_rate >= 0.98, (
            f"Only {positive_rate:.2%} of prices are > 0, expected ≥ 98%."
        )

    def test_price_median_in_range(self, regular_orders):
        """
        The median price must lie between £0.50 and £20.

        Reasoning: the actual median is £2.10 which matches what you would
        expect from a gift shop. We widen the band to £0.50–£20 to tolerate
        seasonal shifts (e.g. Christmas items) while still catching a scenario
        where the wrong column got loaded — customer_id for example averages
        in the thousands and would immediately fail this check.
        """
        median = regular_orders["price"].median()
        assert 0.50 <= median <= 20.00, (
            f"Median price is £{median:.2f}, expected between £0.50 and £20. "
            f"Column may be mis-mapped."
        )    

    def test_price_upper_bound(self, regular_orders):
        """
        At least 99% of price values must be ≤ £5000.

        Reasoning: real product prices in this dataset are all well below
        £5000. The values that exceed this are internal accounting entries
        like Amazon fees and bad debt adjustments — not actual products.
        A 1% tolerance covers any edge cases.
        """
        within_bound = (regular_orders["price"] <= 5000).mean()
        assert within_bound >= 0.99, (
            f"Only {within_bound:.2%} of prices are ≤ £5000, expected ≥ 99%."
        )


# Attribute 2: country

class TestCountryDistribution:
    """
    Distribution checks for the country column.

    The UCI documentation states this is a UK-based retailer.
    When you look at the actual data:
      United Kingdom → 92.1%
      Ireland        → 1.7%
      Germany        → 1.6%
      France         → 1.3%
      Netherlands    → 0.5%
    """

    def test_uk_is_majority(self, regular_orders):
        """
        United Kingdom must account for at least 80% of rows.

        Reasoning: empirically 92% of rows are UK. We set the floor at 80%
        so genuine growth in international orders does not trigger a false
        alarm, but a corrupted or shuffled country column would drop UK share
        much lower and get caught immediately.
        """
        uk_rate = regular_orders["country"].str.strip().eq("United Kingdom").mean()
        assert uk_rate >= 0.80, (
            f"United Kingdom is only {uk_rate:.2%} of rows, expected ≥ 80%."
        )

    def test_top_markets_present(self, regular_orders):
        """
        United Kingdom, EIRE, Germany and France must all appear in the data.

        Reasoning: these are the top 4 markets across both fiscal years.
        If any are missing the data extract is truncated or the column
        has been mis-mapped.
        """
        known_markets = {"United Kingdom", "EIRE", "Germany", "France"}
        actual = set(regular_orders["country"].unique())
        missing = known_markets - actual
        assert not missing, (
            f"Expected countries {known_markets} to be present. "
            f"Missing: {missing}."
        )

    def test_unspecified_rate(self, regular_orders):
        """
        'Unspecified' must account for at most 2% of rows.

        Reasoning: 'Unspecified' appears when the customer country is unknown
        (e.g. guest checkouts with no registered address). This should be a
        small minority. A spike above 2% would mean something broke upstream
        in the geo-lookup for a large batch of customers.
        """
        unspecified_rate = regular_orders["country"].str.strip().eq("Unspecified").mean()
        assert unspecified_rate <= 0.02, (
            f"'Unspecified' is {unspecified_rate:.2%} of rows, expected ≤ 2%."
        )