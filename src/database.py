"""
Database module for CUSTOMER360 AI.
Manages the SQLite analytical database, schema creation, data loading, and query execution.
"""

import logging
import sqlite3
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

from src.config import (
    SQLITE_DB_PATH,
    SCHEMA_SQL_PATH,
    QUERIES_SQL_PATH,
    CLEANED_DATA_CSV,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Returns a SQLite connection to the customer360 database."""
    target_path = Path(db_path) if db_path else SQLITE_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_database(
    db_path: Optional[Path] = None,
    schema_path: Optional[Path] = None,
    data_path: Optional[Path] = None,
    force_recreate: bool = False,
) -> None:
    """
    Initializes the SQLite database using schema.sql and loads cleaned customer records.
    """
    target_db = Path(db_path) if db_path else SQLITE_DB_PATH
    target_schema = Path(schema_path) if schema_path else SCHEMA_SQL_PATH
    target_data = Path(data_path) if data_path else CLEANED_DATA_CSV

    if target_db.exists() and not force_recreate:
        with get_db_connection(target_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='customers';")
            table_exists = cursor.fetchone()[0] > 0
            if table_exists:
                cursor.execute("SELECT count(*) FROM customers;")
                count = cursor.fetchone()[0]
                if count > 0:
                    logger.info(f"Database already initialized with {count} customer rows.")
                    return

    logger.info(f"Creating database schema from: {target_schema}")
    with open(target_schema, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with get_db_connection(target_db) as conn:
        conn.executescript(schema_sql)
        logger.info("Schema executed successfully.")

        if not target_data.exists():
            raise FileNotFoundError(f"Cleaned dataset not found at {target_data}. Run preprocessing first.")

        logger.info(f"Loading data into customers table from: {target_data}")
        df = pd.read_csv(target_data)
        
        # Insert records into SQLite
        df.to_sql("customers", conn, if_exists="append", index=False)
        logger.info(f"Successfully loaded {len(df)} rows into 'customers' table.")


def parse_queries_file(queries_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Parses database/queries.sql and extracts named queries.
    Named queries are tagged with `-- NAME: query_name`.
    """
    target_queries = Path(queries_path) if queries_path else QUERIES_SQL_PATH
    if not target_queries.exists():
        raise FileNotFoundError(f"Queries file not found at: {target_queries}")

    with open(target_queries, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by -- NAME: <name>
    pattern = r"--\s*NAME:\s*([a-zA-Z0-9_]+)\s*\n(.*?)(?=(?:--\s*NAME:|\Z))"
    matches = re.findall(pattern, content, flags=re.DOTALL)

    queries = {}
    for name, sql in matches:
        clean_sql = sql.strip()
        # Remove trailing semicolons for consistency with pandas
        if clean_sql.endswith(";"):
            clean_sql = clean_sql[:-1]
        queries[name] = clean_sql

    return queries


def run_query(
    query_name: Optional[str] = None,
    raw_sql: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Executes a named query from queries.sql or raw SQL against SQLite.
    Returns the result as a pandas DataFrame.
    """
    if query_name:
        queries = parse_queries_file()
        if query_name not in queries:
            raise ValueError(f"Unknown query name: '{query_name}'. Available: {list(queries.keys())}")
        sql = queries[query_name]
    elif raw_sql:
        sql = raw_sql
    else:
        raise ValueError("Must provide either 'query_name' or 'raw_sql'.")

    conn = get_db_connection(db_path)
    try:
        if params:
            df = pd.read_sql_query(sql, conn, params=params)
        else:
            df = pd.read_sql_query(sql, conn)
        return df
    finally:
        conn.close()


def execute_all_analytical_queries() -> Dict[str, pd.DataFrame]:
    """
    Executes all predefined queries in queries.sql and returns a dictionary of DataFrames.
    Useful for automated verification and test assertions.
    """
    queries = parse_queries_file()
    results = {}
    for name in queries:
        logger.info(f"Executing analytical query: {name}")
        results[name] = run_query(query_name=name)
    return results


if __name__ == "__main__":
    init_database(force_recreate=True)
    all_res = execute_all_analytical_queries()
    for name, df in all_res.items():
        print(f"\n=== Query: {name} ({len(df)} rows) ===")
        print(df.head(5))
