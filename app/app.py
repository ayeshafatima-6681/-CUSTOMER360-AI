"""
CUSTOMER360 AI - Customer Churn Prediction & Retention Decision Intelligence Platform.
Main Streamlit Application Controller.
"""

import sys
from pathlib import Path

# Add project root and app directory to sys.path to ensure both direct and package imports resolve
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

try:
    from components.styles import CUSTOM_CSS
    from components.executive import render_executive_overview
    from components.analytics import render_customer_analytics
    from components.prediction import render_churn_prediction
    from components.explainability_view import render_explainability_view
    from components.retention_sim import render_retention_simulator
    from components.performance import render_model_performance
    from components.data_quality import render_data_quality
except (ModuleNotFoundError, ImportError):
    from app.components.styles import CUSTOM_CSS
    from app.components.executive import render_executive_overview
    from app.components.analytics import render_customer_analytics
    from app.components.prediction import render_churn_prediction
    from app.components.explainability_view import render_explainability_view
    from app.components.retention_sim import render_retention_simulator
    from app.components.performance import render_model_performance
    from app.components.data_quality import render_data_quality


def main():
    st.set_page_config(
        page_title="CUSTOMER360 AI — Churn Intelligence Platform",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom styling
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar Navigation & Platform Status
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
                <h2 style="margin: 0; color: #1e293b; font-weight: 800; letter-spacing: -0.03em;">🎯 CUSTOMER360 AI</h2>
                <div style="font-size: 0.8rem; color: #64748b; font-weight: 500;">Decision Intelligence Platform</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        nav_selection = st.radio(
            "Platform Navigation:",
            options=[
                "🏢 Executive Overview",
                "📊 Customer Analytics",
                "🔮 Churn Prediction",
                "💼 Retention Simulator",
                "🔬 Explainable AI",
                "🏆 Model Performance",
                "🛡️ Data Quality & Provenance",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown(
            """
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.85rem; font-size: 0.8rem;">
                <div style="font-weight: 700; color: #0f172a; margin-bottom: 0.35rem;">🚀 System Architecture</div>
                <div style="color: #475569; margin-bottom: 0.2rem;">• <strong>Dataset:</strong> IBM Telco (7,043 rows)</div>
                <div style="color: #475569; margin-bottom: 0.2rem;">• <strong>Database:</strong> SQLite Warehouse</div>
                <div style="color: #475569; margin-bottom: 0.2rem;">• <strong>Model:</strong> Calibrated XGBoost</div>
                <div style="color: #475569; margin-bottom: 0.2rem;">• <strong>Test ROC-AUC:</strong> 0.8467</div>
                <div style="color: #475569;">• <strong>Explainability:</strong> Tree SHAP</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.caption("CUSTOMER360 AI v1.0 • Enterprise Data Science Portfolio")

    # Brand Header Banner
    st.markdown(
        """
        <div class="brand-header">
            <h1 class="brand-title">CUSTOMER360 AI</h1>
            <p class="brand-subtitle">Customer Churn Prediction & Retention Decision Intelligence Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Route to selected view
    if nav_selection == "🏢 Executive Overview":
        render_executive_overview()
    elif nav_selection == "📊 Customer Analytics":
        render_customer_analytics()
    elif nav_selection == "🔮 Churn Prediction":
        render_churn_prediction()
    elif nav_selection == "💼 Retention Simulator":
        render_retention_simulator()
    elif nav_selection == "🔬 Explainable AI":
        render_explainability_view()
    elif nav_selection == "🏆 Model Performance":
        render_model_performance()
    elif nav_selection == "🛡️ Data Quality & Provenance":
        render_data_quality()


if __name__ == "__main__":
    main()
