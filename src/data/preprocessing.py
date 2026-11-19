"""
Data preprocessing and feature engineering pipeline for CUSTOMER360 AI.
Ensures rigorous data cleaning, type conversion, and leakage-free feature transformation.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    from sklearn.compose._column_transformer import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    RAW_DATA_FILE,
    CLEANED_DATA_CSV,
    CLEANED_DATA_FILE,
    ID_COLUMN,
    TARGET_COLUMN,
    TARGET_MAPPING,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    RANDOM_STATE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw IBM Telco Customer Churn data.
    - Strips whitespace from strings.
    - Converts TotalCharges to numeric, imputing the 11 tenure=0 rows with 0.0.
    - Converts SeniorCitizen to consistent integer format.
    - Adds business analytics columns: tenure_group, service counts, charge ratios.
    
    Returns:
        pd.DataFrame: Cleaned dataframe ready for SQL and feature pipelines.
    """
    df = df.copy()

    # Strip column names and string cells
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # TotalCharges conversion: blank strings become NaN then imputed with 0.0 (tenure == 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    tenure_zero_mask = (df["tenure"] == 0) & (df["TotalCharges"].isna())
    df.loc[tenure_zero_mask, "TotalCharges"] = 0.0
    
    # Any remaining NaNs in TotalCharges fill with 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Ensure numeric types
    df["tenure"] = df["tenure"].astype(int)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Derived business feature: tenure_group for cohort analytics
    bins = [-1, 12, 24, 48, 72]
    labels = ["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    # Derived feature: number of add-on security services subscribed
    security_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["security_services_count"] = df[security_cols].apply(
        lambda row: sum(val == "Yes" for val in row), axis=1
    )

    # Derived feature: streaming services subscribed
    streaming_cols = ["StreamingTV", "StreamingMovies"]
    df["streaming_services_count"] = df[streaming_cols].apply(
        lambda row: sum(val == "Yes" for val in row), axis=1
    )

    # Total add-on services count
    df["total_addon_services"] = df["security_services_count"] + df["streaming_services_count"]

    # Has Internet Service binary indicator
    df["has_internet"] = (df["InternetService"] != "No").astype(int)

    # Target binary indicator if Target column is present
    if TARGET_COLUMN in df.columns:
        df["Churn_Binary"] = df[TARGET_COLUMN].map(TARGET_MAPPING)

    return df


def get_feature_lists() -> Dict[str, list]:
    """
    Returns the designated feature lists for machine learning pipelines.
    Separates numerical features and categorical features.
    """
    num_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "security_services_count",
        "streaming_services_count",
        "total_addon_services",
    ]
    cat_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    return {
        "numerical": num_features,
        "categorical": cat_features,
    }


def create_preprocessor() -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer that:
    - Standardizes numerical features using StandardScaler.
    - One-hot encodes categorical features with drop='first' or handle_unknown='ignore'.
    """
    feat_lists = get_feature_lists()
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feat_lists["numerical"]),
            ("cat", categorical_transformer, feat_lists["categorical"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def process_and_save_data(
    raw_path: Optional[Path] = None,
    csv_out: Optional[Path] = None,
    parquet_out: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Reads raw data, applies clean_raw_data, and saves to processed data directory.
    """
    in_file = Path(raw_path) if raw_path else RAW_DATA_FILE
    out_csv = Path(csv_out) if csv_out else CLEANED_DATA_CSV
    out_parquet = Path(parquet_out) if parquet_out else CLEANED_DATA_FILE

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading raw data from: {in_file}")
    df_raw = pd.read_csv(in_file)
    logger.info(f"Raw data shape: {df_raw.shape}")

    df_cleaned = clean_raw_data(df_raw)
    logger.info(f"Cleaned data shape: {df_cleaned.shape}")

    # Save to CSV and Parquet
    df_cleaned.to_csv(out_csv, index=False)
    # Convert categorical dtype to string for parquet serialization compatibility
    df_parquet = df_cleaned.copy()
    if "tenure_group" in df_parquet.columns:
        df_parquet["tenure_group"] = df_parquet["tenure_group"].astype(str)
    df_parquet.to_parquet(out_parquet, index=False)

    logger.info(f"Cleaned dataset saved to:\n  - CSV: {out_csv}\n  - Parquet: {out_parquet}")
    return df_cleaned


if __name__ == "__main__":
    process_and_save_data()
