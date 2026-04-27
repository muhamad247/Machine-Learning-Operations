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

**Step 4 — Run the full Task 2 pipeline (data tests → train → validate)**
```
py -m src.flow
```

To demo the induced training error (small dataset scenario):
```
py -m src.flow --simulate-error
```

## Versioning

Each time the dataset is saved, the code computes a hash of the file content and uses it as the filename:

```
data/versioned/online_retail_<hash>.parquet
```

This way the same data always produces the same filename, and different versions never overwrite each other. A `manifest.json` keeps track of all versions and which one is latest. The tests pick up the latest version automatically.

The same approach is used for trained models — they are saved as `models/model_<hash>.joblib` and tracked in `models/model_manifest.json` with their input/output schemas, dependencies and training metadata.

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


## Task 2 — Model Training Pipeline

### Pipeline Steps (orchestrated in `src/flow.py`)

The flow runs three steps in sequence. If any step fails, the pipeline stops.

| Step | What it does |
|---|---|
| 1. Data tests | Runs all the tests from Task 1 with pytest. The pipeline only proceeds if data quality is acceptable. |
| 2. Train and version model | Trains a linear regression and saves it with a content-hash filename in `models/`. |
| 3. Validate robustness | Loads the trained model and checks it beats a baseline (predicting the mean). |

### Model

Predicts `quantity` from `price` and `country` on regular (non-cancelled) orders. Linear regression with one-hot encoded country.

| Property | Value |
|---|---|
| Target | quantity |
| Features | price, country |
| Model type | LinearRegression with OneHotEncoder for country |
| Serialization | joblib |

### Robustness Check

The model must do at least as well as a baseline that predicts the mean quantity for every row. The baseline by definition has R² = 0 on test data, so we require model R² ≥ baseline R². If the model doesn't beat this, it has not learned anything useful from the features and the check fails.

### Induced Error Handling

The training step requires at least 1000 rows of training data. If less is available it raises `InsufficientDataError` and stops the pipeline with a clear message. To demonstrate this scenario:

```
py -m src.flow --simulate-error
```

This artificially shrinks the dataset to 500 rows and shows the error handling working.

## Task 2 — Results

When running `py -m src.flow` end-to-end:

- **Step 1 (data tests):** all 17 tests pass
- **Step 2 (training):** model trained successfully on 1,047,877 rows, R² on training data ≈ 0.005, saved to `models/model_<hash>.joblib`
- **Step 3 (robustness check):** model R² on the held-out test set is approximately -0.0008 vs baseline R² of 0.0000. The model fails the robustness check.

The linear regression with only `price` and `country` as features cannot meaningfully predict `quantity` — customers buy 1 candle for £5 or 100 candles for £5 depending on whether they are retail or wholesale buyers and our two features cannot distinguish those cases. Our robustness check correctly identified this as a weak model and stopped the pipeline.

When running `py -m src.flow --simulate-error`:

- **Step 1 (data tests):** all 17 tests pass
- **Step 2 (training):** the dataset is artificially shrunk to 500 rows, which is below the `MIN_ROWS = 1000` threshold. Training raises `InsufficientDataError` with a clear message and the pipeline stops cleanly.

This demonstrates the induced error handling working as intended.