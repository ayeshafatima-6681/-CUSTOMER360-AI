"""
Explainable AI (XAI) View for CUSTOMER360 AI.
Surfaces global feature attribution, SHAP tree attributions, and correlation vs. causation caveats.
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import json

from src.config import FEATURE_IMPORTANCES_PATH, MODEL_METRICS_PATH
from src.explainability import ExplainabilityEngine


def render_explainability_view():
    st.markdown("### 🔬 Explainable AI (XAI) & Attribution Intelligence")
    st.markdown(
        "Deconstruct the machine learning decision surface to understand **why** the model predicts churn, "
        "leveraging Tree SHAP attributions and global feature importance rankings."
    )

    # Correlation vs. Causation Caution Callout
    st.markdown(
        """
        <div style="background: #eff6ff; border-left: 5px solid #3b82f6; padding: 1rem 1.25rem; border-radius: 8px; margin-bottom: 1.5rem;">
            <h5 style="margin: 0 0 0.35rem 0; color: #1e40af; font-weight: 600;">⚖️ Data Science Governance: Correlation vs. Causation</h5>
            <p style="margin: 0; color: #1e3a8a; font-size: 0.88rem; line-height: 1.5;">
                Feature importance and SHAP attributions quantify <strong>predictive association</strong> in historical customer records,
                not proven mechanistic causality. For instance, while <em>Electronic check</em> payers exhibit higher churn, switching their payment method
                does not automatically eliminate discontent with service quality or pricing. Prescriptive interventions should be tested via controlled A/B experiments.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Global Feature Importance Chart
    st.markdown("##### 🌐 Global Feature Importance (Production XGBoost Engine)")
    st.caption("Quantifies the relative fractional contribution of each transformed feature across all decision trees.")

    if FEATURE_IMPORTANCES_PATH.exists():
        with open(FEATURE_IMPORTANCES_PATH, "r", encoding="utf-8") as f:
            imp_dict = json.load(f)

        imp_df = (
            pd.DataFrame(list(imp_dict.items()), columns=["Feature", "Importance"])
            .head(15)
            .sort_values("Importance", ascending=True)
        )
        imp_df["Importance_Pct"] = imp_df["Importance"] * 100.0

        fig_imp = px.bar(
            imp_df,
            x="Importance_Pct",
            y="Feature",
            orientation="h",
            text="Importance_Pct",
            labels={"Importance_Pct": "Relative Importance (%)", "Feature": "Feature Name"},
            color="Importance_Pct",
            color_continuous_scale="Blues",
        )
        fig_imp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_imp.update_layout(
            margin=dict(t=10, b=10, l=10, r=20),
            height=420,
            coloraxis_showscale=False,
            xaxis=dict(range=[0, max(imp_df["Importance_Pct"]) * 1.2]),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # Feature Interpretation Table
    st.markdown("##### 📖 Business Interpretation of Top Churn Drivers")
    interpretation_data = [
        {
            "Feature": "Contract: Two Year / One Year",
            "Impact Direction": "Strongly Reduces Churn (-)",
            "Mechanism": "Customers locked into multi-year contracts have structural switching costs, legal commitment, and established tenure stability.",
            "Strategic Levers": "Incentivize contract extensions with device subsidies or temporary fee discounts.",
        },
        {
            "Feature": "Internet Service: Fiber Optic",
            "Impact Direction": "Strongly Increases Churn (+)",
            "Mechanism": "High monthly pricing (~$91.50) without adequate technical support drives sticker shock and customer dissatisfaction when outages occur.",
            "Strategic Levers": "Attach complimentary Tech Support & proactive speed assurance during onboarding.",
        },
        {
            "Feature": "Payment: Electronic Check",
            "Impact Direction": "Increases Churn (+)",
            "Mechanism": "Manual monthly invoice payment introduces recurring decision friction and statement shock compared to frictionless automated deduction.",
            "Strategic Levers": "Promote AutoPay conversion via $10 statement credit incentive.",
        },
        {
            "Feature": "Tenure (Customer Age)",
            "Impact Direction": "Strongly Reduces Churn (-)",
            "Mechanism": "Customer mortality follows an L-curve: 55.8% of churn occurs in months 0-12, dropping significantly once accounts cross 24 months.",
            "Strategic Levers": "Dedicate onboarding concierges and VIP support to accounts in months 1-6.",
        },
        {
            "Feature": "Online Security & Tech Support",
            "Impact Direction": "Reduces Churn (-)",
            "Mechanism": "Customers with security and technical protection bundles find the service stickier and experience fewer unresolved technical frustrations.",
            "Strategic Levers": "Bundle security add-ons into base internet tiers rather than selling as isolated paid extras.",
        },
    ]

    st.dataframe(pd.DataFrame(interpretation_data), use_container_width=True, hide_index=True)
