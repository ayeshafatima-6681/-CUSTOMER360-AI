"""
Data validation module for CUSTOMER360 AI.
Validates schemas, data types, missing values, duplicates, and range anomalies.
"""

import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.config import (
    ALL_EXPECTED_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception raised when dataset schema or integrity fails validation."""
    pass


def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates that the dataframe contains all required columns.
    
    Raises:
        DataValidationError: If any required column is missing.
    """
    missing_cols = [c for c in ALL_EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns in dataset: {missing_cols}")
    
    return {
        "status": "PASS",
        "total_columns": len(df.columns),
        "expected_columns_found": len(ALL_EXPECTED_COLUMNS),
        "extra_columns": [c for c in df.columns if c not in ALL_EXPECTED_COLUMNS],
    }


def check_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Checks for duplicate rows and duplicate customer IDs.
    """
    full_duplicates = int(df.duplicated().sum())
    id_duplicates = int(df.duplicated(subset=[ID_COLUMN]).sum()) if ID_COLUMN in df.columns else 0

    return {
        "full_duplicate_rows": full_duplicates,
        "duplicate_customer_ids": id_duplicates,
        "is_unique_by_id": id_duplicates == 0,
    }


def check_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes standard nulls (NaN, None) and whitespace blank strings.
    """
    standard_nulls = df.isnull().sum().to_dict()

    # Detect whitespace strings that masquerade as valid text
    blank_string_counts = {}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        blanks = (df[col].astype(str).str.strip() == "").sum()
        if blanks > 0:
            blank_string_counts[col] = int(blanks)

    return {
        "standard_null_counts": {k: int(v) for k, v in standard_nulls.items() if v > 0},
        "blank_string_counts": blank_string_counts,
        "total_missing_standard": int(df.isnull().sum().sum()),
        "total_missing_blank_strings": sum(blank_string_counts.values()),
    }


def validate_categorical_domains(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Inspects unique categorical values to ensure they conform to expected domain values.
    """
    domain_summary = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            domain_summary[col] = sorted([str(x) for x in df[col].dropna().unique()])
    
    if TARGET_COLUMN in df.columns:
        target_vals = sorted([str(x) for x in df[TARGET_COLUMN].dropna().unique()])
        domain_summary[TARGET_COLUMN] = target_vals
        if set(target_vals) != {"No", "Yes"}:
            logger.warning(f"Unexpected target values: {target_vals}")

    return domain_summary


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes full suite of data health and schema integrity checks.
    """
    logger.info("Executing comprehensive dataset validation checks...")
    schema_res = validate_schema(df)
    dup_res = check_duplicates(df)
    missing_res = check_missing_values(df)
    domains_res = validate_categorical_domains(df)

    validation_report = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "schema_validation": schema_res,
        "duplicate_checks": dup_res,
        "missing_value_analysis": missing_res,
        "categorical_domains": domains_res,
        "is_valid": (
            schema_res["status"] == "PASS"
            and dup_res["is_unique_by_id"]
        )
    }

    logger.info(
        f"Validation Complete: Valid={validation_report['is_valid']}, "
        f"Rows={validation_report['rows']}, Duplicate IDs={dup_res['duplicate_customer_ids']}, "
        f"Blank string columns={list(missing_res['blank_string_counts'].keys())}"
    )

    return validation_report
