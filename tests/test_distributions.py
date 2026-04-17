"""
Tests for attribute value distributions (price, country and time series)

These columns represent the most business-critical dimensions:
  price   → directly affects revenue calculations
  country → defines the geographic scope of the business
  time    → ensures the data is consistent across the full time series


For price we check the lower end, upper end and the overall shape.
For country we check the dominant value, if hte known markets exist and the unknown countries
For time series we use January 2010 as a reference month and compare other months against it.


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


@pytest.fixture(scope="module")
def reference_month(regular_orders) -> pd.DataFrame:
    """
    January 2010 — used as the reference month for time series tests.

    We chose January 2010 because it is a stable, non-seasonal month in the
    middle of the dataset with no Christmas or holiday effects that would
    distort the baseline.
    """
    return regular_orders[regular_orders["year_month"] == "2010-01"]


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


# Attribute 3: time series consistency

class TestTimeSeriesDistribution:
    """
    Distribution checks over time using January 2010 as a reference month.

    January 2010 was chosen because it is a stable, non-seasonal month in the
    middle of the dataset. We compare other months against it to detect
    anomalies that only show up in specific time periods.
    """

    def test_all_months_have_data(self, regular_orders):
        """
        Every month between Dec 2009 and Dec 2011 must have at least one order.

        Reasoning: this is a continuous transaction log. A month with zero
        orders would indicate missing data in the extract, not a real business
        gap — the retailer operated year-round.
        """
        months = pd.period_range("2009-12", "2011-12", freq="M").astype(str)
        actual_months = set(regular_orders["year_month"].unique())
        missing = [m for m in months if m not in actual_months]
        assert not missing, (
            f"Missing data for months: {missing}. Dataset may be incomplete."
        )

    def test_monthly_uk_share_consistent(self, regular_orders, reference_month):
        """
        UK share per month must stay within 15 percentage points of January 2010.

        Reasoning: the UK share in January 2010 reflects the baseline business
        geography. We allow ±15% to tolerate genuine seasonal variation
        (e.g. more international orders around Christmas) without missing
        a scenario where the country column breaks for an entire month.
        """
        ref_uk_share = reference_month["country"].eq("United Kingdom").mean()

        monthly_uk = (
            regular_orders.groupby("year_month")["country"]
            .apply(lambda x: x.eq("United Kingdom").mean())
        )

        outliers = monthly_uk[abs(monthly_uk - ref_uk_share) > 0.15]
        assert outliers.empty, (
            f"These months have UK share deviating >15% from reference "
            f"({ref_uk_share:.1%}):\n{outliers}"
        )

    def test_monthly_median_price_consistent(self, regular_orders, reference_month):
        """
        Median price per month must stay within £5 of January 2010.

        Reasoning: the median price reflects the typical product mix. We allow
        ±£5 to tolerate seasonal shifts like Christmas gift items being more
        expensive. A larger deviation would suggest a data issue like prices
        being recorded in a different currency for certain months.
        """
        ref_median = reference_month["price"].median()

        monthly_median = regular_orders.groupby("year_month")["price"].median()

        outliers = monthly_median[abs(monthly_median - ref_median) > 5.0]
        assert outliers.empty, (
            f"These months have median price deviating >£5 from reference "
            f"(£{ref_median:.2f}):\n{outliers}"
        )

    def test_monthly_null_rate_consistent(self, regular_orders):
        """
        Description null rate per month must not exceed 5%.

        Reasoning: the overall null rate is 0.41%. A month with more than 5%
        null descriptions would indicate a data ingestion failure for that
        specific time period rather than normal variation.
        """
        monthly_null = (
            regular_orders.groupby("year_month")["description"]
            .apply(lambda x: x.isna().mean())
        )

        outliers = monthly_null[monthly_null > 0.05]
        assert outliers.empty, (
            f"These months have description null rate above 5%:\n{outliers}"
        )    