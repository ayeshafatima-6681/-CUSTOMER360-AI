"""
Tests for single and batch model inference, probability calibration, and input handling.
"""

import pytest
import pandas as pd
import numpy as np

from src.ml.predict import ChurnPredictor
from src.config import CLEANED_DATA_CSV


@pytest.fixture(scope="module")
def predictor():
    return ChurnPredictor()


def test_predict_single_structure(predictor):
    """Verify single customer prediction produces expected fields and valid probability."""
    sample = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 255.0,
    }
    res = predictor.predict_single(sample)

    assert "churn_prediction" in res
    assert res["churn_prediction"] in ["Churn", "Retained"]
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert res["risk_tier"] in ["Low", "Medium", "High", "Critical"]
    assert "value_metrics" in res
    assert "recommended_actions" in res
    assert "explanation" in res
    assert len(res["recommended_actions"]) > 0


def test_predict_batch(predictor):
    """Verify batch prediction scores a DataFrame accurately."""
    df = pd.read_csv(CLEANED_DATA_CSV).head(20)
    scored_df = predictor.predict_batch(df)

    assert len(scored_df) == 20
    assert "churn_probability" in scored_df.columns
    assert "risk_tier" in scored_df.columns
    assert "annual_revenue_exposure_usd" in scored_df.columns
    assert (scored_df["churn_probability"] >= 0.0).all()
    assert (scored_df["churn_probability"] <= 1.0).all()


def test_predict_single_edge_case_tenure_zero(predictor):
    """Verify inference gracefully handles brand new customer with 0 tenure."""
    new_customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 45.0,
        "TotalCharges": 0.0,
    }
    res = predictor.predict_single(new_customer)
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert res["value_metrics"]["tenure_months"] == 0
