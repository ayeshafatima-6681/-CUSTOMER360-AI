"""
Integration tests for Streamlit application components and controller routing.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import plotly.express as px

from app.components.styles import CUSTOM_CSS
from app.components.executive import render_executive_overview
from app.components.analytics import render_customer_analytics
from app.components.explainability_view import render_explainability_view
from app.components.performance import render_model_performance
from app.components.data_quality import render_data_quality
from app.components.prediction import render_churn_prediction


def test_custom_css_is_non_empty():
    """Verify custom CSS string is defined and contains styling rules."""
    assert len(CUSTOM_CSS) > 100
    assert "brand-header" in CUSTOM_CSS
    assert "kpi-card" in CUSTOM_CSS


@patch("streamlit.plotly_chart")
@patch("streamlit.metric")
def test_render_executive_overview(mock_metric, mock_chart):
    """Verify executive overview renders without error."""
    render_executive_overview()
    assert mock_metric.called


@patch("streamlit.plotly_chart")
def test_render_customer_analytics(mock_chart):
    """Verify customer analytics renders without error."""
    render_customer_analytics()


@patch("streamlit.plotly_chart")
def test_render_explainability_view(mock_chart):
    """Verify explainability view renders without error."""
    render_explainability_view()


@patch("streamlit.plotly_chart")
def test_render_model_performance(mock_chart):
    """Verify model performance renders without error."""
    render_model_performance()


def test_render_data_quality():
    """Verify data quality view renders without error."""
    render_data_quality()


@patch("streamlit.plotly_chart")
@patch("streamlit.form_submit_button", return_value=False)
def test_render_churn_prediction(mock_btn, mock_chart):
    """Verify churn prediction page renders without import or execution errors."""
    render_churn_prediction()


def test_plotly_express_bar_chart_shap():
    """Verify px.bar executes without NameError on SHAP attributions."""
    sample_drivers = [
        {"feature": "Contract Commitment: Month-to-month", "shap_contribution": 0.56, "impact": "Increases Risk"},
        {"feature": "Tenure: 2 mos", "shap_contribution": 0.52, "impact": "Increases Risk"},
        {"feature": "Internet Technology: Fiber optic", "shap_contribution": 0.37, "impact": "Increases Risk"},
        {"feature": "Multiple Lines: No", "shap_contribution": -0.13, "impact": "Reduces Risk"},
    ]
    df = pd.DataFrame(sample_drivers)
    fig = px.bar(
        df,
        x="shap_contribution",
        y="feature",
        orientation="h",
        color="impact",
        color_discrete_map={"Increases Risk": "#ef4444", "Reduces Risk": "#22c55e"},
    )
    assert fig is not None
    assert len(fig.data) >= 1
