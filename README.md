# ML Systems and Operations

## Dataset

**Online Retail II** (UCI ML Repository · https://archive.ics.uci.edu/dataset/502/online+retail+ii)

A real transaction log from a UK-based online gift-ware retailer covering December 2009 to December 2011. It has around 1 million rows. It has real time series (InvoiceDate), numeric columns (Quantity, Price), categorical columns (Country, StockCode) and a text column (Description).

## How to Run

**Step 1 — Install dependencies**
```
py -m pip install -r requirements.txt
```

**Step 2 — Download the dataset and save it as Parquet**
```
py -m src.data_loader
```
This fetches the data from UCI, cleans it up and converts it to Parquet and saves it to `data/versioned/`. If you already ran this once, it skips the download and goes straight to saving.

**Step 3 — Run the tests**
```
py -m pytest tests/ -v
```

If you already have the Parquet file, you can skip Step 2 and go straight to Step 3.

## Versioning

Each time the dataset is saved, the code computes a hash of the file content and uses it as the filename:

```
data/versioned/online_retail_<hash>.parquet
```

This way the same data always produces the same filename, and different versions never overwrite each other. A `manifest.json` keeps track of all versions and which one is latest. The tests pick up the latest version automatically.


## Tests

### test_null_values.py

| Column | Threshold | Why |
|---|---|---|
| description | ≤ 1% null | 1% gives a small buffer for noise |
| quantity | 0% null | Always recorded at dispatch — a null here is a serious data gap |
| price | 0% null | Can't calculate invoice totals without it |
| invoice_date | 0% null | It's the time series index — a null row has no place on the timeline |

### test_distributions.py

**price:** We check the lower end, upper end and the overall shape

| Check | Threshold | Why |
|---|---|---|
| Values > £0 | ≥ 98% | Free items exist (samples etc.) but should be rare |
| Values ≤ £5000 | ≥ 99% | Anything higher is almost always a data entry mistake |
| Median | £0.50 – £20 | Empirical median is around £2.10; a wildly different value would suggest the wrong column got loaded |

**country:** We check the dominant value, whether known markets exist and the unknown/unresolved entries:

| Check | Threshold | Why |
|---|---|---|
| UK share | ≥ 80% | Leaves room for genuine growth in international orders |
| UK, Ireland, Germany, France present | always | These are consistently the top 4 markets — if they're missing, something is wrong with the extract |
| Unspecified | ≤ 2% | This is the placeholder for unknown countries — a spike means geo-lookup failed upstream |

**time series:** We use January 2010 as a reference month and compare other months against it:

| Check | Threshold | Why |
|---|---|---|
| All months have data | always | A missing month means the extract is incomplete |
| Monthly UK share | within ±15% of reference | Tolerates seasonal variation but catches column corruption |
| Monthly median price | within ±£5 of reference | Tolerates seasonal shifts but catches currency or mapping errors |
| Monthly description null rate | ≤ 5% | Overall rate is 0.41%; above 5% means ingestion failed for that month |

### test_cancelled_orders.py

Tests the cancelled orders segment separately to verify they follow the expected pattern:

| Check | Threshold | Why |
|---|---|---|
| Quantities are negative | ≥ 99% | Cancellations reverse the original order so quantities must be negative |
| Prices are positive | ≥ 99% | Price reflects the original unit price, so should always be positive |
| Invoice numbers start with 'C' | 100% | This is the definition of a cancellation record in this dataset |

