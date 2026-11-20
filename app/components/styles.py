"""
Custom CSS styling and Plotly formatting for CUSTOMER360 AI.
"""

import plotly.io as pio

# Custom Plotly Template
pio.templates["customer360"] = pio.templates["plotly_white"]

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }
    
    /* Top Brand Header */
    .brand-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 6px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .brand-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin: 0;
        color: #f8fafc;
    }
    .brand-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 0.25rem;
        margin-bottom: 0;
    }

    /* KPI Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }
    .kpi-subtitle {
        font-size: 0.8rem;
        color: #64748b;
    }

    /* Streamlit Native Metric Sizing for Responsive Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        white-space: nowrap;
    }

    /* Risk Badges */
    .badge-critical {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-high {
        background-color: #ffedd5;
        color: #9a3412;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }

    /* Action Card */
    .action-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .action-title {
        font-weight: 600;
        color: #1e293b;
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
    }
    .action-desc {
        font-size: 0.85rem;
        color: #475569;
        margin-bottom: 0.35rem;
    }
    .action-rationale {
        font-size: 0.8rem;
        color: #64748b;
        font-style: italic;
    }

    /* Insight Banner */
    .insight-banner {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #22c55e;
        border-radius: 8px;
        padding: 0.85rem 1.15rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        color: #166534;
        font-size: 0.88rem;
    }
</style>
"""
