"""
Retention Decision Intelligence & Campaign Simulator for CUSTOMER360 AI.
Allows executives to model budget allocation, acceptance rates, and projected retention ROI.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.config import CLEANED_DATA_CSV
from src.decision_intelligence import simulate_retention_campaign, classify_risk_tier
from src.ml.predict import ChurnPredictor


@st.cache_data
def get_scored_portfolio_data():
    """Scores the full verified customer base once to enable fast interactive simulation."""
    df = pd.read_csv(CLEANED_DATA_CSV)
    predictor = ChurnPredictor()
    scored = predictor.predict_batch(df)
    return scored


def render_retention_simulator():
    st.markdown("### 💼 Customer Retention Decision Intelligence & Campaign Simulator")
    st.markdown(
        "Design targeted retention campaigns, test incentive budgets, and evaluate projected financial ROI "
        "across real customer risk cohorts."
    )

    with st.spinner("Loading scored customer portfolio for simulation..."):
        portfolio_df = get_scored_portfolio_data()

    # Enterprise Risk Distribution Summary
    st.markdown("##### 👥 Enterprise Risk Tier Portfolio Breakdown")
    tier_counts = portfolio_df["risk_tier"].value_counts().reindex(["Critical", "High", "Medium", "Low"]).fillna(0)
    tier_rev = portfolio_df.groupby("risk_tier")["annual_revenue_exposure_usd"].sum().reindex(["Critical", "High", "Medium", "Low"]).fillna(0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Critical Risk (≥ 75%)",
            f"{int(tier_counts.get('Critical', 0)):,} accounts",
            f"${tier_rev.get('Critical', 0):,.0f} exposure",
            delta_color="inverse",
        )
    with c2:
        st.metric(
            "High Risk (50%–75%)",
            f"{int(tier_counts.get('High', 0)):,} accounts",
            f"${tier_rev.get('High', 0):,.0f} exposure",
            delta_color="inverse",
        )
    with c3:
        st.metric(
            "Medium Risk (25%–50%)",
            f"{int(tier_counts.get('Medium', 0)):,} accounts",
            f"${tier_rev.get('Medium', 0):,.0f} exposure",
        )
    with c4:
        st.metric(
            "Low Risk (< 25%)",
            f"{int(tier_counts.get('Low', 0)):,} accounts",
            f"${tier_rev.get('Low', 0):,.0f} exposure",
        )

    st.markdown("---")

    # Interactive Scenario Controls
    st.markdown("##### ⚙️ Interactive Retention Campaign Parameters")
    st.caption("Adjust target cohorts, offer costs, and acceptance assumptions to simulate campaign economics.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_tiers = st.multiselect(
            "Target Risk Cohorts:",
            options=["Critical", "High", "Medium", "Low"],
            default=["Critical", "High"],
            help="Select which customer risk tiers will receive the proactive retention offer.",
        )
    with col2:
        offer_cost = st.slider(
            "Incentive Cost per Customer ($):",
            min_value=10.0, max_value=120.0, value=40.0, step=5.0,
            help="Cost of discount, bill credit, or loyalty equipment subsidy.",
        )
    with col3:
        acceptance_rate = st.slider(
            "Estimated Offer Acceptance Rate (%):",
            min_value=5, max_value=75, value=35, step=5,
            help="Fraction of targeted churn-risk customers who accept offer and stay.",
        ) / 100.0
    with col4:
        horizon_months = st.selectbox(
            "Retention Horizon Window:",
            options=[6, 12, 18, 24],
            index=1,
            format_func=lambda x: f"{x} Months Preserved",
            help="Projected horizon over which customer revenue is preserved.",
        )

    if not selected_tiers:
        st.warning("Please select at least one risk cohort to target.")
        return

    # Execute simulation
    sim_res = simulate_retention_campaign(
        customers_df=portfolio_df,
        target_tiers=selected_tiers,
        campaign_cost_per_customer=offer_cost,
        offer_acceptance_rate=acceptance_rate,
        retention_horizon_months=horizon_months,
    )

    proj = sim_res["financial_projections"]

    # Results KPI Cards
    st.markdown("#### 📈 Projected Campaign Financial Outcomes")
    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        st.metric(
            label="Targeted Accounts",
            value=f"{sim_res['targeted_customers_count']:,}",
            delta=f"{sim_res['targeted_pct_of_pool']:.1f}% of base",
        )
    with r2:
        st.metric(
            label="Total Campaign Budget",
            value=f"${proj['total_campaign_investment_usd']:,.0f}",
            help="Total cost of offering the retention incentive to all targeted accounts.",
        )
    with r3:
        st.metric(
            label="Accounts Retained",
            value=f"{proj['expected_retained_customers']:,}",
            delta=f"{acceptance_rate*100:.0f}% acceptance",
        )
    with r4:
        st.metric(
            label="Net Value Created",
            value=f"${proj['net_value_created_usd']:,.0f}",
            delta=f"ROI: {proj['campaign_roi_pct']:,.0f}%",
            delta_color="normal" if proj["net_value_created_usd"] > 0 else "inverse",
        )
    with r5:
        st.metric(
            label="Break-Even Acceptance",
            value=f"{proj['break_even_acceptance_rate_pct']:.1f}%",
            help="Minimum acceptance rate required to fully recoup the campaign budget.",
        )

    # Sensitivity Analysis Curve: Net Value vs Acceptance Rate
    st.markdown("##### 📊 Sensitivity Analysis: Net Savings across Acceptance Rates (10% to 70%)")
    test_rates = np.linspace(0.10, 0.70, 13)
    sens_data = []
    for rate in test_rates:
        temp_sim = simulate_retention_campaign(
            customers_df=portfolio_df,
            target_tiers=selected_tiers,
            campaign_cost_per_customer=offer_cost,
            offer_acceptance_rate=rate,
            retention_horizon_months=horizon_months,
        )
        sens_data.append({
            "Acceptance Rate (%)": round(rate * 100, 1),
            "Net Value Created ($)": temp_sim["financial_projections"]["net_value_created_usd"],
            "Gross Revenue Saved ($)": temp_sim["financial_projections"]["gross_revenue_protected_usd"],
            "Campaign Cost ($)": temp_sim["financial_projections"]["total_campaign_investment_usd"],
        })

    sens_df = pd.DataFrame(sens_data)
    fig_sens = px.line(
        sens_df,
        x="Acceptance Rate (%)",
        y="Net Value Created ($)",
        markers=True,
        labels={"Net Value Created ($)": "Net Value Created (USD)"},
    )
    fig_sens.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-Even ($0 Net)")
    fig_sens.update_traces(line=dict(color="#16a34a", width=3), marker=dict(size=8, color="#15803d"))
    fig_sens.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
    st.plotly_chart(fig_sens, use_container_width=True)

    # Simulation Notice Callout
    st.caption(
        "⚠️ **Simulation Notice:** All calculations reflect hypothetical business models parameterized "
        "by the user's chosen assumptions. Real retention campaigns require continuous measurement and control groups."
    )
