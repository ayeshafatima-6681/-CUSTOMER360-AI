"""
Customer Retention Decision Intelligence Module for CUSTOMER360 AI.
Translates statistical churn probabilities into business risk tiers, financial exposure estimates,
prescriptive action recommendations, and hypothetical scenario simulations.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from src.config import (
    RISK_THRESHOLDS,
    DEFAULT_RETENTION_ASSUMPTIONS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def classify_risk_tier(probability: float) -> str:
    """
    Categorizes predicted churn probability into standard operational risk tiers.
    - Low: [0.00, 0.25)
    - Medium: [0.25, 0.50)
    - High: [0.50, 0.75)
    - Critical: [0.75, 1.00]
    """
    prob = float(probability)
    if prob >= 0.75:
        return "Critical"
    elif prob >= 0.50:
        return "High"
    elif prob >= 0.25:
        return "Medium"
    else:
        return "Low"


def calculate_customer_value_metrics(
    monthly_charges: float,
    total_charges: float,
    tenure: int,
    churn_probability: float,
) -> Dict[str, float]:
    """
    Computes financial value indicators and estimated annual revenue at risk.
    """
    m_charges = float(monthly_charges)
    t_charges = float(total_charges)
    ten = int(tenure)
    prob = float(churn_probability)

    projected_annual_charges = round(m_charges * 12.0, 2)
    # Annual revenue exposure: Expected loss over 12 months weighted by churn probability
    expected_annual_exposure = round(prob * projected_annual_charges, 2)
    
    # Customer lifetime value proxy: historical spend + projected retained future spend
    projected_future_spend = round((1.0 - prob) * projected_annual_charges, 2)
    clv_proxy = round(t_charges + projected_future_spend, 2)

    return {
        "monthly_charges": round(m_charges, 2),
        "total_historical_spend": round(t_charges, 2),
        "tenure_months": ten,
        "projected_annual_charges": projected_annual_charges,
        "expected_annual_exposure": expected_annual_exposure,
        "projected_future_spend": projected_future_spend,
        "customer_lifetime_value_proxy": clv_proxy,
    }


def recommend_retention_actions(
    customer_data: Dict[str, Any],
    churn_probability: float,
    risk_tier: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Prescribes targeted, evidence-based retention interventions based on specific risk factors.
    Prevents margin erosion by avoiding discounts on low-risk customers.
    """
    tier = risk_tier if risk_tier else classify_risk_tier(churn_probability)
    actions = []

    contract = customer_data.get("Contract", "")
    payment = customer_data.get("PaymentMethod", "")
    internet = customer_data.get("InternetService", "")
    tech_supp = customer_data.get("TechSupport", "")
    tenure = int(customer_data.get("tenure", 0))
    monthly = float(customer_data.get("MonthlyCharges", 0.0))

    if tier in ["Critical", "High"]:
        # 1. Contract lock-in incentive
        if contract == "Month-to-month":
            actions.append({
                "action_type": "Contract Commitment Incentive",
                "priority": "P1 - Critical",
                "offer_description": (
                    "Offer 1-Year Contract upgrade with $15/month discount for the first 6 months "
                    "or a $50 one-time billing credit."
                ),
                "empirical_rationale": (
                    "Empirical churn on Month-to-month is 42.7% versus 11.3% on One-year contracts. "
                    "Securing a contract term represents the highest single retention uplift."
                ),
                "estimated_offer_cost_usd": 50.0,
            })

        # 2. Automated billing conversion
        if payment == "Electronic check":
            actions.append({
                "action_type": "AutoPay Conversion Offer",
                "priority": "P2 - High",
                "offer_description": (
                    "Provide $10 one-time billing credit upon enrolling in automated Credit Card "
                    "or Bank Transfer payments."
                ),
                "empirical_rationale": (
                    "Electronic check users experience 45.3% churn compared to ~16% for automated billing, "
                    "frequently caused by payment friction and statement shock."
                ),
                "estimated_offer_cost_usd": 10.0,
            })

        # 3. Fiber optic onboarding & technical care
        if internet == "Fiber optic" and tech_supp == "No":
            actions.append({
                "action_type": "Tech Care & Fiber Optimization Bundle",
                "priority": "P2 - High",
                "offer_description": (
                    "Complimentary 3 months of Premium Tech Support and Online Security bundle, "
                    "with a proactive fiber speed and router diagnostic call."
                ),
                "empirical_rationale": (
                    "Fiber optic customers have high spend ($91.50/mo) but 41.9% churn. Customers with "
                    "Tech Support have substantially lower churn and higher service satisfaction."
                ),
                "estimated_offer_cost_usd": 25.0,
            })

        # 4. Early-tenure onboarding intervention
        if tenure <= 6:
            actions.append({
                "action_type": "Early Lifecycle VIP Onboarding",
                "priority": "P1 - Critical",
                "offer_description": (
                    "Assign dedicated customer success concierge to review usage, verify speed satisfaction, "
                    "and resolve initial onboarding pain points."
                ),
                "empirical_rationale": (
                    "Customers in their first 12 months account for 47.7% of all company churn. "
                    "Intervening in months 1-6 dramatically reduces first-year mortality."
                ),
                "estimated_offer_cost_usd": 20.0,
            })

        # 5. Long-tenure loyalty recognition
        if tenure >= 36:
            actions.append({
                "action_type": "Loyalty VIP Retention Package",
                "priority": "P2 - High",
                "offer_description": (
                    "Executive loyalty recognition reward: Free device/equipment upgrade or "
                    "perpetual $10/month loyalty credit for accounts >3 years."
                ),
                "empirical_rationale": (
                    "High-tenure customers represent high historical lifetime value. Losing mature accounts "
                    "destroys accumulated customer equity."
                ),
                "estimated_offer_cost_usd": 60.0,
            })

    elif tier == "Medium":
        actions.append({
            "action_type": "Proactive Value Check-In & Service Audit",
            "priority": "P3 - Medium",
            "offer_description": (
                "Deliver personalized 'Usage & Savings' statement highlighting bundle benefits, "
                "with an invitation to test complementary streaming or security features for 30 days."
            ),
            "empirical_rationale": (
                "Medium-risk accounts respond well to educational and value-reinforcement touchpoints "
                "without requiring expensive margin discounts."
            ),
            "estimated_offer_cost_usd": 5.0,
        })

    else:  # Low risk
        actions.append({
            "action_type": "Account Health Maintenance & Natural Upsell",
            "priority": "P4 - Low",
            "offer_description": (
                "Maintain baseline automated communication. Customer is highly stable. "
                "Eligible for premium add-on upsell (e.g. streaming tiers or speed boost) when appropriate."
            ),
            "empirical_rationale": (
                "Low-risk customers (<25% churn prob) should not receive defensive discounting, "
                "preserving gross margins."
            ),
            "estimated_offer_cost_usd": 0.0,
        })

    return actions


def simulate_retention_campaign(
    customers_df: pd.DataFrame,
    target_tiers: List[str] = ["Critical", "High"],
    campaign_cost_per_customer: float = 40.0,
    offer_acceptance_rate: float = 0.35,
    retention_horizon_months: int = 12,
) -> Dict[str, Any]:
    """
    Executes a retention scenario simulation across a customer cohort.
    
    Parameters:
        customers_df: DataFrame containing at least 'MonthlyCharges' and 'churn_probability' (or 'Churn_Risk_Tier').
        target_tiers: List of risk tiers targeted by the retention initiative.
        campaign_cost_per_customer: Average cost of the retention incentive in USD.
        offer_acceptance_rate: Estimated fraction of targeted customers who accept the offer and stay (0.0 to 1.0).
        retention_horizon_months: Time window for preserved revenue (default 12 months).
        
    Returns:
        Dict with financial and operational projection metrics.
    """
    df = customers_df.copy()
    
    # Ensure risk tier is computed if missing
    if "Churn_Risk_Tier" not in df.columns:
        if "churn_probability" in df.columns:
            df["Churn_Risk_Tier"] = df["churn_probability"].apply(classify_risk_tier)
        elif "Churn" in df.columns:
            # Fallback based on empirical status
            df["Churn_Risk_Tier"] = df["Churn"].apply(lambda c: "Critical" if c == "Yes" else "Low")
        else:
            raise ValueError("customers_df must include 'churn_probability' or 'Churn_Risk_Tier'.")

    total_pool = len(df)
    targeted_mask = df["Churn_Risk_Tier"].isin(target_tiers)
    targeted_customers = df[targeted_mask]
    targeted_count = len(targeted_customers)

    if targeted_count == 0:
        return {
            "status": "NO_TARGETS",
            "total_customers_in_pool": total_pool,
            "targeted_customers_count": 0,
            "message": "No customers found matching the selected target risk tiers.",
        }

    # Total campaign investment
    total_campaign_cost = targeted_count * campaign_cost_per_customer

    # Expected successfully retained customers
    expected_retained_count = targeted_count * offer_acceptance_rate

    # Calculate gross revenue protected
    # For each retained customer, revenue protected = MonthlyCharges * retention_horizon_months
    avg_monthly_charge_targeted = targeted_customers["MonthlyCharges"].mean()
    total_monthly_revenue_targeted = targeted_customers["MonthlyCharges"].sum()
    
    # Revenue protected over horizon
    gross_revenue_protected = (
        total_monthly_revenue_targeted * offer_acceptance_rate * (retention_horizon_months / 1.0)
    )

    # Net value created = gross revenue saved - campaign budget
    net_value_saved = gross_revenue_protected - total_campaign_cost

    # Return on Investment %
    roi_pct = (
        (net_value_saved / total_campaign_cost) * 100.0 if total_campaign_cost > 0 else 0.0
    )

    # Break-even acceptance rate
    # cost = total_monthly * break_even * horizon
    # break_even = total_campaign_cost / (total_monthly * horizon)
    break_even_rate = (
        total_campaign_cost / (total_monthly_revenue_targeted * retention_horizon_months)
        if (total_monthly_revenue_targeted * retention_horizon_months) > 0
        else 0.0
    )

    return {
        "status": "SUCCESS",
        "total_customers_in_pool": total_pool,
        "targeted_customers_count": targeted_count,
        "targeted_pct_of_pool": round(100.0 * targeted_count / total_pool, 2),
        "target_tiers_selected": target_tiers,
        "assumptions": {
            "campaign_cost_per_customer_usd": campaign_cost_per_customer,
            "offer_acceptance_rate_pct": round(offer_acceptance_rate * 100.0, 1),
            "retention_horizon_months": retention_horizon_months,
            "avg_targeted_monthly_charge_usd": round(float(avg_monthly_charge_targeted), 2),
        },
        "financial_projections": {
            "total_campaign_investment_usd": round(total_campaign_cost, 2),
            "expected_retained_customers": int(round(expected_retained_count)),
            "gross_revenue_protected_usd": round(gross_revenue_protected, 2),
            "net_value_created_usd": round(net_value_saved, 2),
            "campaign_roi_pct": round(roi_pct, 2),
            "break_even_acceptance_rate_pct": round(break_even_rate * 100.0, 2),
        },
        "disclaimer": (
            "Scenario Simulation Notice: Calculations represent hypothetical projections based "
            "on user-defined assumptions. Real campaign performance depends on market conditions, "
            "competitor activity, and customer sentiment."
        ),
    }
