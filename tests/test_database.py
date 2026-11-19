"""
Tests for SQLite analytical database, schema, and analytical SQL queries.
"""

import pytest
import sqlite3
import pandas as pd

from src.config import SQLITE_DB_PATH
from src.database import get_db_connection, run_query, parse_queries_file, execute_all_analytical_queries


def test_database_connection_and_table_count():
    """Verify customer360 database connects and contains 7043 customer rows."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM customers;")
        count = cursor.fetchone()[0]
        assert count == 7043
    finally:
        conn.close()


def test_parse_queries_file():
    """Verify that queries.sql is parsed into expected named queries."""
    queries = parse_queries_file()
    expected_queries = [
        "executive_kpi_summary",
        "churn_by_contract",
        "churn_by_tenure_group",
        "churn_by_payment_method",
        "churn_by_internet_service",
        "churn_by_tech_support",
        "churn_by_security_bundle_count",
        "customer_value_segmentation",
        "high_risk_cohort_breakdown",
        "senior_citizen_churn_profile",
    ]
    for q_name in expected_queries:
        assert q_name in queries
        assert len(queries[q_name]) > 10


def test_executive_kpi_summary_query():
    """Verify executive KPI summary query returns mathematically valid enterprise metrics."""
    df = run_query("executive_kpi_summary")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["total_customers"] == 7043
    assert row["churned_customers"] + row["active_customers"] == 7043
    assert 26.0 <= row["churn_rate_pct"] <= 27.0
    assert row["total_monthly_revenue"] > 400000.0
    assert row["churned_monthly_revenue_exposure"] > 100000.0


def test_churn_by_contract_query():
    """Verify Month-to-month contracts exhibit significantly higher churn than Two-year."""
    df = run_query("churn_by_contract")
    assert len(df) == 3
    contract_rates = dict(zip(df["Contract"], df["churn_rate_pct"]))
    assert contract_rates["Month-to-month"] > 40.0
    assert contract_rates["Two year"] < 5.0


def test_execute_all_analytical_queries():
    """Verify every named query executes without SQL syntax or execution errors."""
    all_results = execute_all_analytical_queries()
    assert len(all_results) >= 10
    for name, df in all_results.items():
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, f"Query '{name}' returned empty DataFrame"
