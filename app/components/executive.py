"""
Executive Overview component for CUSTOMER360 AI.
Surfaces macro enterprise KPIs, revenue exposure, customer segmentation, and strategic findings.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.database import run_query


def render_executive_overview():
    st.markdown("### 🏢 Executive Overview & Enterprise Churn Diagnostics")
    st.markdown(
        "Macro KPI metrics and enterprise revenue exposure derived directly from "
        "the **7,043 genuine IBM Telco customer accounts** in the verified data warehouse."
    )

    # Load real SQL KPIs
    kpi_df = run_query("executive_kpi_summary")
    if kpi_df.empty:
        st.error("Unable to load KPI summary from database.")
        return

    row = kpi_df.iloc[0]
    total_cust = int(row["total_customers"])
    churn_rate = float(row["churn_rate_pct"])
    churned_cust = int(row["churned_customers"])
    active_cust = int(row["active_customers"])
    total_rev = float(row["total_monthly_revenue"])
    churned_rev = float(row["churned_monthly_revenue_exposure"])
    rev_risk_pct = float(row["revenue_at_risk_pct"])
    avg_tenure = float(row["avg_tenure_months"])
    avg_charge = float(row["avg_monthly_charges"])

    # 4 Key Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Total Accounts",
            value=f"{total_cust:,}",
            help="Total verified customer records in the enterprise warehouse.",
        )
    with c2:
        st.metric(
            label="Overall Churn Rate",
            value=f"{churn_rate:.1f}%",
            delta=f"{churned_cust:,} churned accounts",
            delta_color="inverse",
            help="Empirical historical churn rate across all customer cohorts.",
        )
    with c3:
        st.metric(
            label="Monthly Revenue Exposure",
            value=f"${churned_rev:,.0f}",
            delta=f"{rev_risk_pct:.1f}% of monthly billing",
            delta_color="inverse",
            help="Total monthly recurring billing lost to customer departures.",
        )
    with c4:
        st.metric(
            label="Avg Customer Lifetime",
            value=f"{avg_tenure:.1f} mos",
            delta=f"${avg_charge:.2f}/mo avg spend",
            help="Mean customer account tenure across the customer base.",
        )

    st.markdown("---")

    # Visualizations Row 1: Churn Breakdown Donut & Revenue Exposure
    v1, v2 = st.columns([1, 1.2])

    with v1:
        st.markdown("##### 👥 Account Retention vs. Churn Breakdown")
        donut_fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Active Accounts", "Churned Accounts"],
                    values=[active_cust, churned_cust],
                    hole=0.55,
                    marker=dict(colors=["#2563eb", "#ef4444"]),
                    textinfo="label+percent",
                    insidetextorientation="radial",
                    hoverinfo="label+value+percent",
                )
            ]
        )
        donut_fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
            showlegend=False,
        )
        st.plotly_chart(donut_fig, use_container_width=True)

    with v2:
        st.markdown("##### 💵 Monthly Recurring Revenue (MRR) Exposure ($)")
        retained_rev = total_rev - churned_rev
        bar_fig = go.Figure(
            data=[
                go.Bar(
                    x=["Retained MRR", "Churned MRR Lost"],
                    y=[retained_rev, churned_rev],
                    text=[f"${retained_rev:,.0f} ({100-rev_risk_pct:.1f}%)", f"${churned_rev:,.0f} ({rev_risk_pct:.1f}%)"],
                    textposition="auto",
                    marker=dict(color=["#3b82f6", "#dc2626"]),
                )
            ]
        )
        bar_fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
            yaxis_title="Monthly Billing (USD)",
            xaxis_title="",
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("---")

    # Visualizations Row 2: Customer Spend Segmentation
    st.markdown("##### 🎯 Customer Value Segmentation & Churn Vulnerability")
    st.markdown(
        "Accounts classified into High Value Tier (≥ $80/mo), Medium Value ($40–$80/mo), and Budget Tier (< $40/mo)."
    )

    seg_df = run_query("customer_value_segmentation")
    if not seg_df.empty:
        s1, s2 = st.columns([1.3, 1])
        with s1:
            fig_seg = px.bar(
                seg_df,
                x="spend_segment",
                y=["segment_monthly_revenue", "monthly_revenue_at_risk"],
                barmode="group",
                labels={
                    "value": "Monthly Revenue (USD)",
                    "spend_segment": "Spend Tier",
                    "variable": "Revenue Category",
                },
                color_discrete_map={
                    "segment_monthly_revenue": "#3b82f6",
                    "monthly_revenue_at_risk": "#ef4444",
                },
            )
            fig_seg.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_seg, use_container_width=True)

        with s2:
            st.dataframe(
                seg_df[["spend_segment", "customer_count", "churn_rate_pct", "monthly_revenue_at_risk"]].rename(
                    columns={
                        "spend_segment": "Spend Segment",
                        "customer_count": "Customers",
                        "churn_rate_pct": "Churn %",
                        "monthly_revenue_at_risk": "Lost MRR ($)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    # Strategic Findings Callout
    st.markdown(
        """
        <div style="background: #f8fafc; border-left: 5px solid #2563eb; padding: 1.1rem 1.4rem; border-radius: 8px; margin-top: 1.25rem;">
            <h5 style="margin: 0 0 0.5rem 0; color: #1e293b; font-weight: 600;">Executive Takeaways & Key Revenue Risk Drivers</h5>
            <ul style="margin: 0; padding-left: 1.25rem; color: #334155; font-size: 0.9rem; line-height: 1.6;">
                <li><strong>Revenue Concentration:</strong> Churned customers represent <strong>$139,130/month (30.5%)</strong> of total enterprise monthly billing, exceeding the raw customer churn rate of 26.5% due to higher churn in high-ARPU fiber cohorts.</li>
                <li><strong>Contract Disparity:</strong> Month-to-month contracts suffer a <strong>42.7% churn rate</strong> versus only <strong>2.8% on Two-year contracts</strong>, demonstrating that term commitments are the primary structural churn defense.</li>
                <li><strong>Payment Friction:</strong> Electronic check payers churn at <strong>45.3%</strong>, generating over $68,000/month in lost billing, indicating substantial billing friction compared to automated payment channels (~16% churn).</li>
                <li><strong>Early Lifecycle Mortality:</strong> Customers with ≤ 12 months tenure account for <strong>55.8% of all churn events</strong> ($68,954/mo lost), highlighting an acute need for early-stage customer success interventions.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
