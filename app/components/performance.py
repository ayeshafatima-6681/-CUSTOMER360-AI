"""
Model Performance & Technical Validation Component for CUSTOMER360 AI.
Surfaces rigorous 5-fold cross-validation metrics, test partition ROC/PR curves,
confusion matrices, probability calibration, and threshold optimization.
"""

import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.config import MODEL_METRICS_PATH


def render_model_performance():
    st.markdown("### 🏆 Machine Learning Model Evaluation & Calibration")
    st.markdown(
        "Scientific validation benchmarks evaluated strictly on the **unseen held-out test set** "
        "(1,409 accounts: 374 churn positives, 1,035 active negatives) with 5-fold cross-validation."
    )

    if not MODEL_METRICS_PATH.exists():
        st.error("Model metrics artifact not found. Please run training pipeline first.")
        return

    with open(MODEL_METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    best_name = metrics["best_model_name"]
    best_roc = metrics["best_test_roc_auc"]
    calib = metrics["calibration"]

    # Top Benchmark Banner
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Production Model", best_name)
    with c2:
        st.metric("Test ROC-AUC", f"{best_roc:.4f}", "Held-out test set")
    with c3:
        st.metric("Calibrated Brier Score", f"{calib['post_calibration_brier']:.4f}", f"Improved from {calib['pre_calibration_brier']:.4f}")
    with c4:
        st.metric("Calibrated ROC-AUC", f"{calib['post_calibration_roc_auc']:.4f}", "Sigmoidal Platt scaling")

    st.markdown("---")

    # Model Comparison Table
    st.markdown("##### 📊 Candidate Model Comparison Matrix")
    cv_res = metrics["cv_results"]
    test_res = metrics["test_results"]

    table_rows = []
    for model_name in cv_res:
        cv_m = cv_res[model_name]
        t_m = test_res[model_name]
        table_rows.append({
            "Model Architecture": model_name,
            "5-Fold CV ROC-AUC": f"{cv_m['cv_roc_auc_mean']:.4f} (±{cv_m['cv_roc_auc_std']:.3f})",
            "5-Fold CV PR-AUC": f"{cv_m['cv_pr_auc_mean']:.4f}",
            "5-Fold CV F1": f"{cv_m['cv_f1_mean']:.4f}",
            "5-Fold CV Recall": f"{cv_m['cv_recall_mean']:.4f}",
            "Test ROC-AUC": f"{t_m['roc_auc']:.4f}",
            "Test PR-AUC": f"{t_m['pr_auc']:.4f}",
            "Test Recall": f"{t_m['recall']:.4f}",
            "Test Precision": f"{t_m['precision']:.4f}",
            "Test F1-Score": f"{t_m['f1_score']:.4f}",
            "Brier Score": f"{t_m['brier_score']:.4f}",
        })

    comp_df = pd.DataFrame(table_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Curves Section: ROC and PR
    cr1, cr2 = st.columns(2)

    with cr1:
        st.markdown("##### 📈 Test ROC Curves (Receiver Operating Characteristic)")
        roc_fig = go.Figure()
        # Add diagonal no-skill line
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="No Skill (0.50)"))
        colors = {"XGBoost Classifier": "#2563eb", "Random Forest": "#16a34a", "Logistic Regression": "#ea580c"}

        for m_name in test_res:
            roc_data = test_res[m_name]["roc_curve"]
            roc_auc_val = test_res[m_name]["roc_auc"]
            roc_fig.add_trace(
                go.Scatter(
                    x=roc_data["fpr"],
                    y=roc_data["tpr"],
                    mode="lines",
                    line=dict(color=colors.get(m_name, "#6b7280"), width=2),
                    name=f"{m_name} (AUC = {roc_auc_val:.3f})",
                )
            )
        roc_fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=340,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Recall)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(roc_fig, use_container_width=True)

    with cr2:
        st.markdown("##### 🎯 Test Precision-Recall Curves")
        pr_fig = go.Figure()
        for m_name in test_res:
            pr_data = test_res[m_name]["pr_curve"]
            pr_auc_val = test_res[m_name]["pr_auc"]
            pr_fig.add_trace(
                go.Scatter(
                    x=pr_data["recall"],
                    y=pr_data["precision"],
                    mode="lines",
                    line=dict(color=colors.get(m_name, "#6b7280"), width=2),
                    name=f"{m_name} (PR-AUC = {pr_auc_val:.3f})",
                )
            )
        pr_fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=340,
            xaxis_title="Recall (Sensitivity)",
            yaxis_title="Precision",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(pr_fig, use_container_width=True)

    st.markdown("---")

    # Confusion Matrix & Probability Calibration
    m1, m2 = st.columns(2)

    with m1:
        st.markdown(f"##### 🔲 Confusion Matrix ({best_name} @ Threshold = 0.50)")
        cm = test_res[best_name]["confusion_matrix"]
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]

        cm_matrix = [[tn, fp], [fn, tp]]
        cm_fig = px.imshow(
            cm_matrix,
            text_auto=True,
            x=["Predicted Retained (0)", "Predicted Churn (1)"],
            y=["Actual Retained (0)", "Actual Churn (1)"],
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Class", y="True Class", color="Count"),
        )
        cm_fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300, coloraxis_showscale=False)
        st.plotly_chart(cm_fig, use_container_width=True)

    with m2:
        st.markdown(f"##### 🎯 Probability Calibration Reliability Curve")
        cal_data = test_res[best_name]["calibration_curve"]
        cal_fig = go.Figure()
        cal_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="Perfect Calibration"))
        cal_fig.add_trace(
            go.Scatter(
                x=cal_data["prob_pred"],
                y=cal_data["prob_true"],
                mode="lines+markers",
                line=dict(color="#2563eb", width=2),
                marker=dict(size=8),
                name=f"{best_name} Reliability",
            )
        )
        cal_fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Observed Fraction of Churn Positives",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(cal_fig, use_container_width=True)

    # Threshold Optimization Explorer
    st.markdown("---")
    st.markdown("##### 🎚️ Decision Threshold Optimization Explorer")
    st.caption(
        "Analyze the business trade-off between Precision and Recall across classification decision thresholds."
    )
    th_list = metrics["threshold_analysis"]
    th_df = pd.DataFrame(th_list)

    th_fig = px.line(
        th_df,
        x="threshold",
        y=["precision", "recall", "f1"],
        labels={"value": "Score", "threshold": "Decision Threshold", "variable": "Metric"},
        color_discrete_map={"precision": "#2563eb", "recall": "#ef4444", "f1": "#10b981"},
        markers=True,
    )
    th_fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
    st.plotly_chart(th_fig, use_container_width=True)
