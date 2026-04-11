import hashlib
import io
import json
import logging
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR       = Path("data/raw")
VERSIONED_DIR = Path("data/versioned")
MANIFEST_PATH = VERSIONED_DIR / "manifest.json"

DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

_RENAME = {
    "Invoice":     "invoice_no",
    "StockCode":   "stock_code",
    "Description": "description",
    "Quantity":    "quantity",
    "InvoiceDate": "invoice_date",
    "Price":       "price",
    "Customer ID": "customer_id",
    "Country":     "country",
}


# Step 1: Download

def download_raw(force: bool = False) -> Path:
    """
    Download the dataset zip from UCI and extract the Excel file.
    If the file is already on disk we skip the download.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_dest  = RAW_DIR / "online_retail_II.zip"
    xlsx_dest = RAW_DIR / "online_retail_II.xlsx"

    if xlsx_dest.exists() and not force:
        logger.info("Raw file already on disk — skipping download.")
        return xlsx_dest

    logger.info("Downloading from UCI ...")
    urllib.request.urlretrieve(DATASET_URL, zip_dest)

    with zipfile.ZipFile(zip_dest, "r") as zf:
        xlsx_name = next(n for n in zf.namelist() if n.endswith(".xlsx"))
        xlsx_dest.write_bytes(zf.read(xlsx_name))

    logger.info("Saved to %s", xlsx_dest)
    return xlsx_dest


# Step 2: Load and normalise

def load_and_normalise(raw_path: Path) -> pd.DataFrame:
    """
    Read both Excel sheets (2009-2010 and 2010-2011), concatenate and normalise.
    """
    dfs = []
    for sheet in ["Year 2009-2010", "Year 2010-2011"]:
        df = pd.read_excel(
            raw_path,
            sheet_name=sheet,
            dtype={
                "Invoice":     "string",
                "StockCode":   "string",
                "Description": "string",
                "Customer ID": "string",
                "Country":     "string",
            },
        )
        df["source_sheet"] = sheet
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined.rename(columns=_RENAME, inplace=True)
    combined["invoice_date"] = pd.to_datetime(combined["invoice_date"], utc=True)
    """year_month makes it easy to slice data by month for time-series segmentation without re-parsing strings each time."""
    combined["year_month"]   = (
        combined["invoice_date"].dt.tz_localize(None).dt.to_period("M").astype(str)
    )

    return combined


# Step 3: Save as Parquet

def save_versioned(df: pd.DataFrame) -> Path:
    """
    Write the DataFrame to a content-addressed Parquet file.

    The manifest records metadata (row count, columns, creation time) for
    every version and tracks which is 'latest'. Tests always load the correct
    version.
    """
    VERSIONED_DIR.mkdir(parents=True, exist_ok=True)

    content_hash = _compute_hash(df)
    parquet_path = VERSIONED_DIR / f"online_retail_{content_hash}.parquet"

    if not parquet_path.exists():
        df.to_parquet(parquet_path, index=False, compression="snappy")
        logger.info("Saved Parquet → %s", parquet_path)
    else:
        logger.info("Hash already exists — no duplicate written.")

    manifest = _load_manifest()
    manifest["versions"][content_hash] = {
        "path":       str(parquet_path),
        "rows":       len(df),
        "columns":    list(df.columns),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_url": DATASET_URL,
    }
    manifest["latest"] = content_hash
    _save_manifest(manifest)

    return parquet_path


def resolve_latest() -> Path:
    """Return the path of the most recently saved Parquet version."""
    manifest = _load_manifest()
    if not manifest.get("versions"):
        raise FileNotFoundError(
            "No versioned dataset found. Run `py -m src.data_loader` first."
        )
    return Path(manifest["versions"][manifest["latest"]]["path"])


# Internal helpers

def _compute_hash(df: pd.DataFrame) -> str:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()[:12]

def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"latest": None, "versions": {}}

def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


# Entry point

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    raw  = download_raw()
    df   = load_and_normalise(raw)
    path = save_versioned(df)
    logger.info("Done. %d rows saved to %s", len(df), path)
    sys.exit(0)
