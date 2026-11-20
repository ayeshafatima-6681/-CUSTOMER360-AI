"""
Tests for Explainability Engine and exact Tree SHAP attributions.
"""

import pytest
import pandas as pd

from src.explainability import ExplainabilityEngine
from src.config import CLEANED_DATA_CSV


@pytest.fixture(scope="module")
def explainer():
    return ExplainabilityEngine()


def test_get_global_importance(explainer):
    """Verify global feature importance extraction."""
    top_feats = explainer.get_global_importance(top_n=10)
    assert len(top_feats) == 10
    assert "feature" in top_feats[0]
    assert "importance_score" in top_feats[0]
    assert top_feats[0]["importance_score"] >= top_feats[1]["importance_score"]


def test_explain_instance_tree_shap(explainer):
    """Verify individual customer prediction explanation with exact Tree SHAP values."""
    df = pd.read_csv(CLEANED_DATA_CSV).head(1)
    res = explainer.explain_instance(df, calibrated_probability=0.65)

    assert "predicted_probability" in res
    assert "calibrated_probability" in res
    assert "raw_probability" in res
    assert res["calibrated_probability"] == 0.65
    assert "top_risk_drivers" in res
    assert "top_protective_factors" in res
    assert "narrative_summary" in res
    assert "causality_caveat" in res
    assert len(res["top_risk_drivers"]) > 0 or len(res["top_protective_factors"]) > 0
    # Risk drivers have positive SHAP contribution
    for d in res["top_risk_drivers"]:
        assert d["shap_contribution"] > 0
    # Protective factors have negative SHAP contribution
    for p in res["top_protective_factors"]:
        assert p["shap_contribution"] < 0


def test_explain_instance_feature_fidelity(explainer):
    """Verify explanation reflects the customer's actual submitted feature values."""
    sample = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
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
        "MonthlyCharges": 85.50,
        "TotalCharges": 171.00,
        "security_services_count": 0,
        "streaming_services_count": 1,
        "total_addon_services": 1,
    }])
    res = explainer.explain_instance(sample)

    all_feats = [d["feature"] for d in res["top_risk_drivers"] + res["top_protective_factors"]]
    # Verify feature strings reflect the actual submitted values
    contract_matches = [f for f in all_feats if "Contract" in f]
    assert len(contract_matches) > 0
    # Must show Month-to-month, never Two year
    assert any("Month-to-month" in f for f in contract_matches)
    assert not any("Two year" in f for f in contract_matches)

    # Verify Internet Service shows Fiber optic
    internet_matches = [f for f in all_feats if "Internet Technology" in f or "InternetService" in f]
    assert any("Fiber optic" in f for f in internet_matches)
