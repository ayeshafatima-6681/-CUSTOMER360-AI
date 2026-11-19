"""
Integration tests for Streamlit application components and controller routing.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from app.components.styles import CUSTOM_CSS
from app.components.executive import render_executive_overview
from app.components.analytics import render_customer_analytics
from app.components.explainability_view import render_explainability_view
from app.components.performance import render_model_performance
from app.components.data_quality import render_data_quality


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
