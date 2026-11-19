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
    res = explainer.explain_instance(df)

    assert "predicted_probability" in res
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
