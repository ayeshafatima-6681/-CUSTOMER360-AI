"""
Configuration settings, schema specifications, and constant paths for CUSTOMER360 AI.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATABASE_DIR = PROJECT_ROOT / "database"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Specific File Paths
RAW_DATA_FILE = RAW_DATA_DIR / "Telco-Customer-Churn.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_telco_churn.parquet"
CLEANED_DATA_CSV = PROCESSED_DATA_DIR / "cleaned_telco_churn.csv"
SQLITE_DB_PATH = DATABASE_DIR / "customer360.db"
SCHEMA_SQL_PATH = DATABASE_DIR / "schema.sql"
QUERIES_SQL_PATH = DATABASE_DIR / "queries.sql"

# Serialized Model Artifacts
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
CALIBRATED_MODEL_PATH = MODELS_DIR / "calibrated_model.joblib"
MODEL_METRICS_PATH = MODELS_DIR / "model_metrics.json"
FEATURE_IMPORTANCES_PATH = MODELS_DIR / "feature_importances.json"

# Dataset Provenance
DATA_SOURCE_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
FALLBACK_SOURCE_URL = "https://raw.githubusercontent.com/Treselle-Systems/Customer_Churn_Analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATASET_NAME = "IBM Telco Customer Churn"
DATASET_LICENSE = "Apache 2.0 / Open IBM Sample Data"

# Schema Definitions
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
TARGET_MAPPING = {"No": 0, "Yes": 1}
INV_TARGET_MAPPING = {0: "No", 1: "Yes"}

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
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

ALL_EXPECTED_COLUMNS = [ID_COLUMN] + CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET_COLUMN]

# Machine Learning Settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DECISION_THRESHOLD = 0.50

# Retention Intelligence Constants & Risk Thresholds
RISK_THRESHOLDS = {
    "Low": (0.0, 0.25),
    "Medium": (0.25, 0.50),
    "High": (0.50, 0.75),
    "Critical": (0.75, 1.00),
}

# Default Business Assumptions (explicitly labeled as customizable assumptions)
DEFAULT_RETENTION_ASSUMPTIONS = {
    "monthly_discount_usd": 15.0,
    "one_time_credit_usd": 30.0,
    "annual_contract_incentive_usd": 50.0,
    "default_success_rate": 0.35,  # 35% estimated probability customer accepts retention offer
    "extended_lifetime_months": 12,  # Projected horizon retained
    "cost_of_acquisition_proxy_usd": 250.0,
}
