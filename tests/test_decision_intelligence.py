"""
Tests for customer risk tier classification, revenue exposure, action recommendations, and scenario simulations.
"""

import pytest
import pandas as pd

from src.decision_intelligence import (
    classify_risk_tier,
    calculate_customer_value_metrics,
    recommend_retention_actions,
    simulate_retention_campaign,
)
from src.config import CLEANED_DATA_CSV


def test_classify_risk_tier():
    """Verify correct categorization of probabilities into operational risk tiers."""
    assert classify_risk_tier(0.85) == "Critical"
    assert classify_risk_tier(0.75) == "Critical"
    assert classify_risk_tier(0.60) == "High"
    assert classify_risk_tier(0.50) == "High"
    assert classify_risk_tier(0.35) == "Medium"
    assert classify_risk_tier(0.25) == "Medium"
    assert classify_risk_tier(0.15) == "Low"
    assert classify_risk_tier(0.00) == "Low"


def test_calculate_customer_value_metrics():
    """Verify customer value and annual exposure calculations."""
    metrics = calculate_customer_value_metrics(
        monthly_charges=80.0,
        total_charges=960.0,
        tenure=12,
        churn_probability=0.50,
    )
    assert metrics["monthly_charges"] == 80.0
    assert metrics["projected_annual_charges"] == 960.0
    assert metrics["expected_annual_exposure"] == 480.0  # 0.50 * 960
    assert metrics["customer_lifetime_value_proxy"] == 960.0 + (1.0 - 0.50) * 960.0


def test_recommend_retention_actions_month_to_month():
    """Verify recommendation logic prioritizes contract upgrade for month-to-month customers."""
    cust = {
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic",
        "TechSupport": "No",
        "tenure": 4,
        "MonthlyCharges": 90.0,
    }
    actions = recommend_retention_actions(cust, churn_probability=0.80, risk_tier="Critical")
    action_types = [a["action_type"] for a in actions]
    assert "Contract Commitment Incentive" in action_types
    assert "AutoPay Conversion Offer" in action_types
    assert "Tech Care & Fiber Optimization Bundle" in action_types


def test_simulate_retention_campaign():
    """Verify retention scenario simulation mathematics."""
    df = pd.read_csv(CLEANED_DATA_CSV).head(100).copy()
    df["churn_probability"] = 0.60
    df["Churn_Risk_Tier"] = "High"

    sim = simulate_retention_campaign(
        customers_df=df,
        target_tiers=["High"],
        campaign_cost_per_customer=50.0,
        offer_acceptance_rate=0.40,
        retention_horizon_months=12,
    )

    assert sim["status"] == "SUCCESS"
    assert sim["targeted_customers_count"] == 100
    proj = sim["financial_projections"]
    assert proj["total_campaign_investment_usd"] == 5000.0  # 100 * $50
    assert proj["expected_retained_customers"] == 40  # 100 * 0.40
    assert proj["gross_revenue_protected_usd"] > 0
    assert proj["net_value_created_usd"] == proj["gross_revenue_protected_usd"] - 5000.0
    assert proj["break_even_acceptance_rate_pct"] > 0
