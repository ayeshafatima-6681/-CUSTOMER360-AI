"""
Tests for data ingestion, provenance, and validation.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.config import RAW_DATA_FILE, ALL_EXPECTED_COLUMNS
from src.data.ingestion import acquire_dataset
from src.data.validation import (
    validate_schema,
    check_duplicates,
    check_missing_values,
    validate_categorical_domains,
    validate_dataset,
    DataValidationError,
)


def test_acquire_dataset_loads_genuine_data():
    """Verify raw dataset is loaded with genuine IBM Telco shape and schema."""
    df = acquire_dataset(force_download=False)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (7043, 21)
    assert "customerID" in df.columns
    assert "Churn" in df.columns


def test_validate_schema_success():
    """Verify schema passes when all 21 expected columns exist."""
    df = pd.read_csv(RAW_DATA_FILE)
    res = validate_schema(df)
    assert res["status"] == "PASS"
    assert res["expected_columns_found"] == len(ALL_EXPECTED_COLUMNS)


def test_validate_schema_raises_on_missing_column():
    """Verify DataValidationError is raised if a required column is missing."""
    df = pd.DataFrame({"customerID": ["123"], "gender": ["Male"]})
    with pytest.raises(DataValidationError):
        validate_schema(df)


def test_check_duplicates():
    """Verify no duplicate customer IDs exist in authentic dataset."""
    df = pd.read_csv(RAW_DATA_FILE)
    dup_res = check_duplicates(df)
    assert dup_res["full_duplicate_rows"] == 0
    assert dup_res["duplicate_customer_ids"] == 0
    assert dup_res["is_unique_by_id"] is True


def test_check_missing_values():
    """Verify detection of the 11 blank strings in TotalCharges."""
    df = pd.read_csv(RAW_DATA_FILE)
    missing_res = check_missing_values(df)
    assert missing_res["total_missing_standard"] == 0
    assert "TotalCharges" in missing_res["blank_string_counts"]
    assert missing_res["blank_string_counts"]["TotalCharges"] == 11


def test_validate_categorical_domains():
    """Verify categorical domains match expected values."""
    df = pd.read_csv(RAW_DATA_FILE)
    domains = validate_categorical_domains(df)
    assert set(domains["Churn"]) == {"No", "Yes"}
    assert set(domains["Contract"]) == {"Month-to-month", "One year", "Two year"}
    assert set(domains["InternetService"]) == {"DSL", "Fiber optic", "No"}


def test_validate_dataset_comprehensive():
    """Verify full validation suite runs cleanly."""
    df = pd.read_csv(RAW_DATA_FILE)
    report = validate_dataset(df)
    assert report["is_valid"] is True
    assert report["rows"] == 7043
    assert report["columns"] == 21
