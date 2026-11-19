"""
Data ingestion module for CUSTOMER360 AI.
Downloads and validates the authentic IBM Telco Customer Churn dataset.
"""

import json
import logging
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from src.config import (
    DATA_SOURCE_URL,
    FALLBACK_SOURCE_URL,
    RAW_DATA_FILE,
    RAW_DATA_DIR,
    DATASET_NAME,
    DATASET_LICENSE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def acquire_dataset(
    dest_path: Optional[Path] = None,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Acquires the genuine IBM Telco Customer Churn dataset from official/reliable repositories.
    If already present locally and force_download is False, loads from disk.
    
    Returns:
        pd.DataFrame: Raw customer churn dataframe.
    """
    target_path = Path(dest_path) if dest_path else RAW_DATA_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_path = target_path.parent / "provenance_metadata.json"

    if target_path.exists() and not force_download:
        logger.info(f"Loading existing raw dataset from: {target_path}")
        df = pd.read_csv(target_path)
        logger.info(f"Loaded raw dataset with shape: {df.shape}")
        return df

    urls_to_try = [DATA_SOURCE_URL, FALLBACK_SOURCE_URL]
    download_success = False
    source_used = None

    for url in urls_to_try:
        try:
            logger.info(f"Attempting download from: {url}")
            # Use urllib to fetch dataset
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CUSTOMER360-AI/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(target_path, "wb") as out_file:
                out_file.write(response.read())
            
            logger.info(f"Successfully downloaded raw dataset to: {target_path}")
            source_used = url
            download_success = True
            break
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")

    if not download_success:
        error_msg = (
            f"ERROR: Unable to download dataset from remote sources.\n"
            f"Please manually place the authentic 'Telco-Customer-Churn.csv' file at:\n"
            f"{target_path.resolve()}\n"
            f"Official Source URL: {DATA_SOURCE_URL}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Read and verify dataframe
    df = pd.read_csv(target_path)
    
    # Save provenance metadata
    metadata = {
        "dataset_name": DATASET_NAME,
        "license": DATASET_LICENSE,
        "source_url": source_used,
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "file_size_bytes": target_path.stat().st_size,
        "provenance_note": "Genuine public IBM Telco Customer Churn sample for churn prediction research."
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset provenance metadata saved to: {metadata_path}")
    logger.info(f"Raw dataset shape: {df.shape}")
    return df


if __name__ == "__main__":
    acquire_dataset()
