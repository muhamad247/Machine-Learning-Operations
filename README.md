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

**When running with Docker (`docker-compose up --build`):**
- Step 1 runs inside its own container with only pandas, pyarrow and pytest installed
- Step 2 runs inside its own container with scikit-learn and joblib added
- Step 3 runs inside its own container with numpy added
- Results are identical to the local run — all 17 tests pass, model trains successfully, robustness check catches the weak model

**When running with Docker and simulated error (`$env:SIMULATE_ERROR=1; docker-compose up --build`):**
- Step 1 passes all 17 tests
- Step 2 stops with InsufficientDataError — 500 rows is below the 1000 minimum
- Step 3 never runs since Step 2 failed


## Task 3 — Post-deployment Tests

Three new pieces on top of the Task 2 pipeline:

1. The training flow is now **parametrized** so different configurations can be tracked and compared.
2. A **drift monitoring flow** checks whether feature distributions have shifted on data the model didn't see.
3. An **A/B prediction flow** evaluates two versions of the training flow side-by-side on the same unseen segment.

Plus two small support modules: `src/flow_registry.py` (tracks flow runs and links them to model ids) and `src/segments.py` (loads reference and unseen segments based on the temporal cutoff).

### Setup

One new dependency for the chi-square test:

```
py -m pip install scipy
```

### The unseen segment

For Task 3, training accepts a `--train-end YYYY-MM` parameter. Rows with `year_month` after this month are reserved as the *unseen segment* — used by both the drift flow and the A/B flow. Without a cutoff there is no truly unseen data (Task 2 training fits on every non-cancelled row), so all Task 3 commands assume a cutoff is set.

`src/validate.py` was updated to use this temporal holdout when a cutoff is present in the model metadata, falling back to the Task 2 iloc-split behavior otherwise. Existing Task 2 results are unchanged.

### Workflow

**Step 1 — Run v1 of the flow**

```
py -m src.flow --train-end 2011-06 --features price,country --model-type linear --run-name v1-baseline
```

This is Task 2 with a temporal cutoff added. At the end the flow prints the run id and model id, both stored under `flow_runs/<id>.json` together with the full config.

**Step 2 — Run v2 of the flow (changed hyperparameters)**

```
py -m src.flow --train-end 2011-06 --features price,country,stock_code --model-type ridge --run-name v2-enriched
```

Two things change from v1: the model type (Ridge instead of plain linear) and the feature set (adds `stock_code`). Both are tracked in the flow run config and produce a different run id and model id. `stock_code` is one-hot encoded with `min_frequency=50` so rare codes get folded into a single "infrequent" bucket — keeps dimensionality manageable and gives unseen codes at predict time a defined home.

The reason `stock_code` is the v2 addition: it's the strongest signal for typical order size in this dataset. A small candle and a 100-pack of mini lights live at the same price points but have very different typical quantities, so adding it should beat the v1 baseline that only sees price and country.

**Step 3 — Run the drift flow**

```
py -m src.drift --run-id <run_id_v1>
```

Loads the reference segment (months ≤ 2011-06) and the unseen segment (months > 2011-06) and runs two drift tests, then writes a JSON report to `drift_reports/`.

| Test | Feature | Why this test |
|---|---|---|
| PSI (Population Stability Index) | price | Standard tabular-drift metric. Bounded, interpretable, has well-known thresholds |
| Chi-square goodness-of-fit | country | Categorical with a long tail; chi-square compares observed vs expected counts directly |

**PSI thresholds:** `<0.10` no significant shift, `0.10–0.25` moderate, `≥0.25` significant. These are industry-standard cutoffs.

**Chi-square:** alpha = 0.05. Categories with expected count `<5` are folded into an `OTHER` bucket so the chi-square approximation stays valid.

**Note on chi-square at large N:** with ~500k rows in the unseen segment, chi-square is very sensitive — even small natural variation in country shares can show up as "drift detected". In a real production setting we'd pair the p-value with an effect-size measure. For this exercise the p-value alone is sufficient since the focus is the test setup, not statistical interpretation.

**Expected behavior**

| Feature | Expectation | Reasoning |
|---|---|---|
| price | At most moderate drift | Some seasonal and inflation-related movement is normal in retail. Significant drift would suggest a pricing strategy change, new product mix, or upstream data issue |
| country | Stable | An established UK retailer's customer base shouldn't shift radically month-to-month. Significant drift usually means a new market opened, geo-lookup broke, or the dataset got mixed with another source |

**Step 4 — Run the A/B test**

```
py -m src.ab_test --run-a <run_id_v1> --run-b <run_id_v2> --test-id ab-001
```

Resolves each run id to its trained model via `flow_registry`, loads the same unseen segment, and splits it by `hash(test_id + customer_id) % 2`:

- bucket 0 → variant A (v1 model)
- bucket 1 → variant B (v2 model)

Results are saved to `ab_tests/ab-001.json` with MAE and R² per branch, plus a simple winner pick by MAE.

**Why MAE:** Interpretable in units of the target (units per order line) and more robust than R² to the heavy-tailed `quantity` distribution. R² is also reported for context.

**Why hash on `customer_id`:** Same `(customer, test)` pair always lands in the same bucket — reproducible without storing an assignment table. Splitting at the customer level (not the row level) means the same customer never sees both models within a single test. Rows with missing `customer_id` get dropped and the drop count is reported.

### Handling multiple concurrent A/B tests

Each test gets a unique `--test-id` which is used as the salt in the customer hash. Different test ids give independent allocations of the same customer base, so the same customer can be in bucket A for `ab-001` and bucket B for `ab-002` with no contamination. Results live under `ab_tests/<test_id>.json` so concurrent tests never overwrite each other.

In a real production setup we would also keep a per-test customer-to-bucket assignment table (so the allocation survives a hash-function change) and a registry of active tests with status, start/end times, and target sample size per bucket. For an offline reproducible setup the salted hash is sufficient and stateless.

### Files added or changed

| File | Status | What it does |
|---|---|---|
| `src/train.py` | modified | Accepts `train_end_month`, `model_type`, `features` |
| `src/flow.py` | modified | Accepts the new CLI args, registers each run |
| `src/validate.py` | modified | Uses temporal holdout when cutoff exists |
| `src/flow_registry.py` | new | Tracks flow runs, resolves run id → model id |
| `src/segments.py` | new | Shared reference/unseen segment loader |
| `src/drift.py` | new | Drift monitoring flow (PSI + Chi-square) |
| `src/ab_test.py` | new | A/B prediction flow |

### Artifacts produced

| Location | What |
|---|---|
| `flow_runs/<run_id>.json` | Full record of each flow run (config + model id + status) |
| `flow_runs/manifest.json` | Index of all flow runs |
| `drift_reports/drift_<run_id>_<timestamp>.json` | Drift test results per run |
| `ab_tests/<test_id>.json` | A/B test results per test id |

To inspect registered flow runs at any point:

```
py -m src.flow_registry
```

## Task 3 — Results

### Flow versioning

Two versions of the training flow were executed with the same temporal cutoff (`--train-end 2011-06`, 756,664 training rows, 291,213 unseen rows):

| Run | run_id | model_hash | Config | Training R² | Holdout R² | Status |
|---|---|---|---|---|---|---|
| v1-baseline | `6fc441a4bc20` | `7740c223bad9` | linear, price + country | 0.0093 | -0.0022 | robustness_failed |
| v2-enriched | `fc7a1df7db03` | `16cde6595495` | ridge, price + country + stock_code | 0.0243 | -0.0300 | robustness_failed |

v2 has higher training R² but worse holdout R². The temporal split exposed overfitting on `stock_code` patterns that don't generalize past June 2011 — a real ML-ops finding the iloc-based holdout from Task 2 would have masked. Both models are retained for downstream comparison since the goal in Task 3 is operational A/B testing, not deployment readiness.

### Drift monitoring

Run on `6fc441a4bc20` (the drift signal is independent of which model is referenced — both share the same cutoff):

| Test | Feature | Value | Verdict |
|---|---|---|---|
| PSI | price | 0.0265 | no_significant_drift |
| Chi-square | country (42 cats) | χ² = 4359.58, p < 0.001 | drift_detected |

Price drift is well below the 0.10 moderate threshold — consistent with the expectation in the drift documentation (some seasonal movement is normal). The country chi-square flags drift, but at 291k rows the test is extremely sensitive — even tiny natural variation in country shares becomes statistically significant. In production we would pair the p-value with an effect-size measure; for this assignment the focus is the test setup, and the result is honestly reported.

### A/B test

Two runs of the A/B flow, identical configuration except for `--test-id` (the hash salt). The same v1 model and v2 model are evaluated; only the customer allocation changes.

| Metric | ab-001 (v1) | ab-001 (v2) | ab-002 (v1) | ab-002 (v2) |
|---|---|---|---|---|
| Rows | 107,291 | 117,986 | 104,592 | 120,685 |
| MAE | 12.13 | **11.86** | **11.18** | 12.64 |
| R² | -0.0022 | -0.2899 | -0.0624 | -0.0123 |
| Winner |  | **B** | **A** |  |

65,936 unseen rows were dropped from each test for missing `customer_id` — these can't be deterministically allocated. The bucket sizes are uneven (47% / 53%) because customers contribute different numbers of rows; the salted hash splits *customers* evenly, not rows.

The key observation: the two tests produce **opposite winners**. The MAE differences between models (0.27 in ab-001, 1.46 in ab-002) are small compared to the variance introduced by which customers land in which bucket, so the salt alone flips the conclusion. This is the operational reason a production A/B test needs a statistical significance check — exactly the analysis the assignment scopes out of this exercise. The R² swings are even more dramatic on v2 (-0.29 vs -0.012), driven by a small number of bulk-order customers whose squared errors dominate the metric when they happen to land in v2's bucket.

The salted-hash allocation behaved correctly in both tests: the same `(customer, test_id)` always produced the same bucket within a test, and different test_ids reshuffled allocation independently. Results were written to `ab_tests/ab-001.json` and `ab_tests/ab-002.json` without collision.