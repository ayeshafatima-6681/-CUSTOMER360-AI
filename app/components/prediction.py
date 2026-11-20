"""
Customer Churn Prediction Component for CUSTOMER360 AI.
Provides single customer real-time inference with archetypes and batch file scoring with export.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.ml.predict import ChurnPredictor
from src.config import DECISION_THRESHOLD, CLEANED_DATA_CSV


# Customer Archetype Presets for instant recruiter demonstration
ARCHETYPES = {
    "Custom Customer (Manual Inputs)": {},
    "Vulnerable New Fiber Optic Customer (High Risk)": {
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
    },
    "Loyal Multi-Year Contract Customer (Low Risk)": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 64.00,
        "TotalCharges": 3840.00,
    },
    "At-Risk Senior Citizen on Electronic Billing (Critical Risk)": {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 98.75,
        "TotalCharges": 493.75,
    },
}


@st.cache_resource
def get_predictor():
    return ChurnPredictor()


def render_churn_prediction():
    st.markdown("### 🔮 Real-Time Customer Churn Prediction Engine")
    st.markdown(
        "Generate live, calibrated churn risk predictions for individual accounts or score bulk portfolios "
        "using the production XGBoost model."
    )

    predictor = get_predictor()

    mode = st.radio(
        "Select Prediction Mode:",
        ["👤 Single Customer Risk Scoring", "📁 Batch Portfolio Scoring & Export"],
        horizontal=True,
    )

    if mode == "👤 Single Customer Risk Scoring":
        render_single_prediction(predictor)
    else:
        render_batch_prediction(predictor)


def render_single_prediction(predictor: ChurnPredictor):
    st.markdown("##### 👤 Customer Profile & Attribute Parameters")

    # Archetype selector
    selected_archetype = st.selectbox(
        "Load Pre-Configured Customer Archetype (or enter custom values):",
        options=list(ARCHETYPES.keys()),
    )
    preset = ARCHETYPES[selected_archetype]

    # Input Form
    with st.form("customer_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics & Tenancy**")
            gender = st.selectbox("Gender", ["Male", "Female"], index=0 if preset.get("gender") == "Male" else 1)
            senior = st.selectbox(
                "Senior Citizen", [0, 1],
                index=preset.get("SeniorCitizen", 0),
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            partner = st.selectbox("Partner", ["No", "Yes"], index=0 if preset.get("Partner", "No") == "No" else 1)
            dependents = st.selectbox("Dependents", ["No", "Yes"], index=0 if preset.get("Dependents", "No") == "No" else 1)
            tenure = st.number_input(
                "Tenure (Months with Company)", min_value=0, max_value=72,
                value=int(preset.get("tenure", 12))
            )

        with col2:
            st.markdown("**Services & Subscriptions**")
            phone = st.selectbox("Phone Service", ["Yes", "No"], index=0 if preset.get("PhoneService", "Yes") == "Yes" else 1)
            mult_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"],
                index=["No", "Yes", "No phone service"].index(preset.get("MultipleLines", "No"))
            )
            internet = st.selectbox(
                "Internet Technology", ["Fiber optic", "DSL", "No"],
                index=["Fiber optic", "DSL", "No"].index(preset.get("InternetService", "Fiber optic"))
            )
            security = st.selectbox(
                "Online Security", ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(preset.get("OnlineSecurity", "No"))
            )
            backup = st.selectbox(
                "Online Backup", ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(preset.get("OnlineBackup", "No"))
            )
            device_prot = st.selectbox(
                "Device Protection", ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(preset.get("DeviceProtection", "No"))
            )
            tech_supp = st.selectbox(
                "Tech Support", ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(preset.get("TechSupport", "No"))
            )
            stream_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(preset.get("StreamingTV", "No"))
            )
            stream_mov = st.selectbox(
                "Streaming Movies", ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(preset.get("StreamingMovies", "No"))
            )

        with col3:
            st.markdown("**Billing, Contract & Charges**")
            contract = st.selectbox(
                "Contract Type", ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(preset.get("Contract", "Month-to-month"))
            )
            paperless = st.selectbox(
                "Paperless Billing", ["Yes", "No"],
                index=0 if preset.get("PaperlessBilling", "Yes") == "Yes" else 1
            )
            payment = st.selectbox(
                "Payment Method", [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=[
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ].index(preset.get("PaymentMethod", "Electronic check"))
            )
            monthly = st.number_input(
                "Monthly Charges ($)", min_value=18.0, max_value=125.0,
                value=float(preset.get("MonthlyCharges", 70.0)), step=1.0
            )
            total_calc_default = float(preset.get("TotalCharges", monthly * tenure))
            total = st.number_input(
                "Total Charges ($)", min_value=0.0, max_value=10000.0,
                value=total_calc_default, step=10.0
            )

        threshold = st.slider(
            "Classification Decision Threshold:",
            min_value=0.10, max_value=0.90, value=0.50, step=0.05,
            help="Threshold above which customer is classified as Churn positive."
        )

        submit_btn = st.form_submit_button("⚡ Compute Churn Risk & Explain", use_container_width=True)

    if submit_btn or selected_archetype != "Custom Customer (Manual Inputs)":
        # Package record
        customer_dict = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": mult_lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device_prot,
            "TechSupport": tech_supp,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_mov,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        with st.spinner("Executing calibrated ML inference and feature attribution..."):
            result = predictor.predict_single(customer_dict, threshold=threshold)

        st.markdown("---")
        st.markdown("#### 🎯 Prediction Results & Risk Diagnostics")

        prob = result["churn_probability"]
        raw_prob = result.get("raw_probability", prob)
        tier = result["risk_tier"]
        val_met = result["value_metrics"]
        pred_label = result["churn_prediction"]
        pred_binary = result["prediction_binary"]

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            tier_colors = {
                "Critical": "#dc2626",
                "High": "#ea580c",
                "Medium": "#d97706",
                "Low": "#16a34a",
            }
            st.metric(
                label="Risk Tier",
                value=tier,
                delta=f"Predicted: {pred_label} (Cutoff: {threshold:.2f})",
                delta_color="inverse" if pred_binary == 1 else "normal",
            )
        with r2:
            st.metric(
                label="Calibrated Churn Probability",
                value=f"{prob * 100:.1f}%",
                delta=f"Raw Tree Score: {raw_prob * 100:.1f}%",
            )
        with r3:
            st.metric(
                label="Annual Revenue Exposure",
                value=f"${val_met['expected_annual_exposure']:,.2f}",
                delta=f"From ${val_met['projected_annual_charges']:,.2f} ACV",
                delta_color="inverse",
            )
        with r4:
            st.metric(
                label="Estimated CLV Proxy",
                value=f"${val_met['customer_lifetime_value_proxy']:,.2f}",
                delta=f"Tenure: {tenure} mos",
            )

        # Transparency Box on Probability Calibration vs Raw Tree Score vs Decision Cutoff
        with st.expander("ℹ️ Model Scoring, Calibration & Decision Consistency Details", expanded=False):
            st.markdown(
                f"""
                - **Calibrated Churn Probability ({prob * 100:.1f}%):** Computed via **Sigmoidal Platt Scaling** (`CalibratedClassifierCV`) on the trained model. This is the **authoritative operational probability** used for risk classification, risk tiering (`{tier}`), and revenue exposure calculations.
                - **Raw Tree Ensemble Score ({raw_prob * 100:.1f}%):** The uncalibrated score from the raw XGBoost boosted decision trees. Tree SHAP feature contributions are computed directly in this raw log-odds margin space.
                - **Classification Decision ({pred_label}):** Derived by comparing the calibrated probability (`{prob * 100:.1f}%`) against your selected decision threshold (`{threshold * 100:.1f}%`). Because `{prob * 100:.1f}%` {'≥' if pred_binary == 1 else '<'} `{threshold * 100:.1f}%`, the customer is classified as **{pred_label}**.
                """
            )

        # Gauge Chart for Probability
        g1, g2 = st.columns([1, 1.3])
        with g1:
            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": "Calibrated Churn Risk (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": tier_colors.get(tier, "#3b82f6")},
                        "steps": [
                            {"range": [0, 25], "color": "#dcfce7"},
                            {"range": [25, 50], "color": "#fef3c7"},
                            {"range": [50, 75], "color": "#ffedd5"},
                            {"range": [75, 100], "color": "#fee2e2"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.75,
                            "value": threshold * 100,
                        },
                    },
                )
            )
            gauge_fig.update_layout(margin=dict(t=30, b=20, l=30, r=30), height=240)
            st.plotly_chart(gauge_fig, use_container_width=True)

        with g2:
            st.markdown("##### 📋 Prescriptive Retention Action Recommendations")
            for act in result["recommended_actions"]:
                st.markdown(
                    f"""
                    <div style="background: #f8fafc; border-left: 4px solid #3b82f6; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 0.5rem;">
                        <span style="font-size: 0.75rem; font-weight: 700; color: #2563eb; text-transform: uppercase;">{act['priority']} — {act['action_type']}</span>
                        <div style="font-size: 0.88rem; font-weight: 600; color: #1e293b; margin: 0.2rem 0;">{act['offer_description']}</div>
                        <div style="font-size: 0.78rem; color: #64748b; font-style: italic;">{act['empirical_rationale']} (Estimated cost: ${act['estimated_offer_cost_usd']:.0f})</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # SHAP Waterfall/Bar Explanation
        if "explanation" in result:
            st.markdown("##### 🔬 Individual Feature Contribution (Tree SHAP)")
            exp = result["explanation"]
            st.caption(exp["narrative_summary"])

            top_drivers = exp["top_risk_drivers"][:6] + exp["top_protective_factors"][:6]
            if top_drivers:
                shap_df = pd.DataFrame(top_drivers)
                fig_shap = px.bar(
                    shap_df,
                    x="shap_contribution",
                    y="feature",
                    orientation="h",
                    color="impact",
                    color_discrete_map={
                        "Increases Risk": "#ef4444",
                        "Reduces Risk": "#22c55e",
                    },
                    labels={"shap_contribution": "SHAP Margin Contribution", "feature": "Customer Attribute & Submitted Value"},
                )
                fig_shap.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=320,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_shap, use_container_width=True)


def render_batch_prediction(predictor: ChurnPredictor):
    st.markdown("##### 📁 Portfolio Batch Scoring")
    st.markdown(
        "Upload a CSV file containing customer records or score a pre-loaded sample from the test partition."
    )

    sample_opt = st.radio(
        "Data Source for Batch Scoring:",
        ["Evaluate 50 Random Unseen Customers from Cleaned Dataset", "Upload Custom CSV File"],
        horizontal=True,
    )

    batch_df = None
    if sample_opt == "Evaluate 50 Random Unseen Customers from Cleaned Dataset":
        cleaned_df = pd.read_csv(CLEANED_DATA_CSV)
        batch_df = cleaned_df.sample(50, random_state=42).copy()
    else:
        uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)

    if batch_df is not None:
        st.write(f"Loaded **{len(batch_df)} customer records** for scoring.")
        th_batch = st.slider("Batch Classification Decision Threshold:", 0.10, 0.90, 0.50, 0.05)

        if st.button("🚀 Score Entire Customer Batch"):
            with st.spinner("Processing batch through preprocessing pipeline and calibrated classifier..."):
                scored_df = predictor.predict_batch(batch_df, threshold=th_batch)

            st.success(f"Successfully scored {len(scored_df)} customer accounts!")

            # Batch Summary Metrics
            b1, b2, b3, b4 = st.columns(4)
            churn_pct = 100.0 * (scored_df["churn_prediction"] == 1).mean()
            crit_count = (scored_df["risk_tier"] == "Critical").sum()
            high_count = (scored_df["risk_tier"] == "High").sum()
            total_exp = scored_df["annual_revenue_exposure_usd"].sum()

            with b1:
                st.metric("Predicted Churn %", f"{churn_pct:.1f}%")
            with b2:
                st.metric("Critical Accounts", f"{crit_count}")
            with b3:
                st.metric("High Risk Accounts", f"{high_count}")
            with b4:
                st.metric("Total Annual Exposure", f"${total_exp:,.2f}")

            # Preview Scored Table
            cols_to_show = [
                "customerID", "Contract", "tenure", "MonthlyCharges",
                "churn_probability", "risk_tier", "annual_revenue_exposure_usd"
            ]
            valid_cols = [c for c in cols_to_show if c in scored_df.columns]
            st.dataframe(
                scored_df[valid_cols].sort_values("churn_probability", ascending=False),
                use_container_width=True,
            )

            # Export button
            csv_data = scored_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Complete Scored Batch (CSV)",
                data=csv_data,
                file_name="customer360_scored_batch.csv",
                mime="text/csv",
            )
