"""
Tests for data cleaning, preprocessing, and feature transformation.
"""

import pytest
import pandas as pd
import numpy as np

from src.config import RAW_DATA_FILE, CLEANED_DATA_CSV
from src.data.preprocessing import clean_raw_data, create_preprocessor, get_feature_lists


def test_clean_raw_data_handles_total_charges_blanks():
    """Verify that the 11 blank TotalCharges strings are converted to 0.0."""
    raw_df = pd.read_csv(RAW_DATA_FILE)
    cleaned_df = clean_raw_data(raw_df)

    assert cleaned_df["TotalCharges"].dtype in [np.float64, np.float32, float]
    assert cleaned_df["TotalCharges"].isna().sum() == 0
    # Specifically inspect the 11 tenure=0 rows
    tenure_zero = cleaned_df[cleaned_df["tenure"] == 0]
    assert len(tenure_zero) == 11
    assert (tenure_zero["TotalCharges"] == 0.0).all()


def test_clean_raw_data_creates_derived_features():
    """Verify creation of derived business features."""
    raw_df = pd.read_csv(RAW_DATA_FILE)
    cleaned = clean_raw_data(raw_df)

    expected_derived = [
        "tenure_group",
        "security_services_count",
        "streaming_services_count",
        "total_addon_services",
        "has_internet",
        "Churn_Binary",
    ]
    for col in expected_derived:
        assert col in cleaned.columns
    
    assert cleaned["Churn_Binary"].isin([0, 1]).all()
    assert (cleaned["security_services_count"] >= 0).all()
    assert (cleaned["security_services_count"] <= 4).all()


def test_preprocessor_pipeline_transformation():
    """Verify ColumnTransformer fits and transforms without leakage."""
    cleaned_df = pd.read_csv(CLEANED_DATA_CSV)
    feat_lists = get_feature_lists()
    all_features = feat_lists["numerical"] + feat_lists["categorical"]

    X = cleaned_df[all_features].copy()
    preprocessor = create_preprocessor()
    X_trans = preprocessor.fit_transform(X)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == len(cleaned_df)
    # Check transformed feature count
    feature_names = preprocessor.get_feature_names_out()
    assert len(feature_names) == X_trans.shape[1]
    assert X_trans.shape[1] > len(all_features)  # Due to one-hot encoding
