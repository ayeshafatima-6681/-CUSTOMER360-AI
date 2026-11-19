-- CUSTOMER360 AI Database Schema (SQLite)
-- Schema for IBM Telco Customer Churn analytical data warehouse

DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customerID TEXT PRIMARY KEY,
    gender TEXT NOT NULL,
    SeniorCitizen INTEGER NOT NULL,
    Partner TEXT NOT NULL,
    Dependents TEXT NOT NULL,
    tenure INTEGER NOT NULL,
    PhoneService TEXT NOT NULL,
    MultipleLines TEXT NOT NULL,
    InternetService TEXT NOT NULL,
    OnlineSecurity TEXT NOT NULL,
    OnlineBackup TEXT NOT NULL,
    DeviceProtection TEXT NOT NULL,
    TechSupport TEXT NOT NULL,
    StreamingTV TEXT NOT NULL,
    StreamingMovies TEXT NOT NULL,
    Contract TEXT NOT NULL,
    PaperlessBilling TEXT NOT NULL,
    PaymentMethod TEXT NOT NULL,
    MonthlyCharges REAL NOT NULL,
    TotalCharges REAL NOT NULL,
    Churn TEXT NOT NULL,
    Churn_Binary INTEGER NOT NULL,
    tenure_group TEXT NOT NULL,
    security_services_count INTEGER NOT NULL,
    streaming_services_count INTEGER NOT NULL,
    total_addon_services INTEGER NOT NULL,
    has_internet INTEGER NOT NULL
);

-- Analytical indexes to accelerate dashboard queries and cohort aggregations
CREATE INDEX idx_customers_churn ON customers(Churn);
CREATE INDEX idx_customers_contract ON customers(Contract);
CREATE INDEX idx_customers_tenure_group ON customers(tenure_group);
CREATE INDEX idx_customers_internet ON customers(InternetService);
CREATE INDEX idx_customers_payment ON customers(PaymentMethod);
