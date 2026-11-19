-- CUSTOMER360 AI Analytical SQL Queries
-- Business Intelligence and Churn Diagnostics for Decision Support

-- 1. Executive Overall Churn & Revenue KPI Summary
-- Calculates total customer count, churned count, overall churn rate, total monthly charges, and churn revenue exposure.
-- NAME: executive_kpi_summary
SELECT 
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    SUM(CASE WHEN Churn = 'No' THEN 1 ELSE 0 END) AS active_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_revenue,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) AS churned_monthly_revenue_exposure,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END) / SUM(MonthlyCharges), 2) AS revenue_at_risk_pct,
    ROUND(AVG(tenure), 1) AS avg_tenure_months,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers;

-- 2. Churn Rate and Revenue by Contract Type
-- Evaluates retention stability across Month-to-month, One year, and Two year contracts.
-- NAME: churn_by_contract
SELECT 
    Contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_charges,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_lost,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge,
    ROUND(AVG(tenure), 1) AS avg_tenure
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- 3. Churn Rate and Revenue by Tenure Group
-- Analyzes customer lifecycle cohorts to pinpoint drop-off stages (e.g. 0-12m onboarding vs mature accounts).
-- NAME: churn_by_tenure_group
SELECT 
    tenure_group,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_charges,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_lost,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge
FROM customers
GROUP BY tenure_group
ORDER BY 
    CASE tenure_group
        WHEN '0-12 Months' THEN 1
        WHEN '13-24 Months' THEN 2
        WHEN '25-48 Months' THEN 3
        WHEN '49-72 Months' THEN 4
        ELSE 5
    END;

-- 4. Churn Rate and Monthly Revenue by Payment Method
-- Determines friction in billing and payment processing channels.
-- NAME: churn_by_payment_method
SELECT 
    PaymentMethod,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_charges,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_lost,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- 5. Churn Rate by Internet Service Category
-- Evaluates churn dynamics between Fiber optic, DSL, and No Internet customers.
-- NAME: churn_by_internet_service
SELECT 
    InternetService,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_charges,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_lost,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;

-- 6. Service Add-on Protection & Support Churn Impact
-- Evaluates whether customers with Tech Support and Online Security have significantly lower churn.
-- NAME: churn_by_tech_support
SELECT 
    TechSupport,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge
FROM customers
GROUP BY TechSupport
ORDER BY churn_rate_pct DESC;

-- 7. Churn by Security Services Bundle Count
-- Analyzes product stickiness: does subscribing to more security/backup/support services reduce churn?
-- NAME: churn_by_security_bundle_count
SELECT 
    security_services_count,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_charges,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge
FROM customers
GROUP BY security_services_count
ORDER BY security_services_count ASC;

-- 8. Customer Value Segmentation Analysis
-- Segments customers into High, Medium, and Low value tiers based on MonthlyCharges and TotalCharges.
-- NAME: customer_value_segmentation
WITH CustomerSegments AS (
    SELECT 
        customerID,
        Churn,
        MonthlyCharges,
        TotalCharges,
        tenure,
        CASE 
            WHEN MonthlyCharges >= 80.0 THEN 'High Value Tier (>= $80/mo)'
            WHEN MonthlyCharges >= 40.0 THEN 'Medium Value Tier ($40-$80/mo)'
            ELSE 'Budget Tier (< $40/mo)'
        END AS spend_segment
    FROM customers
)
SELECT 
    spend_segment,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_count,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2) AS segment_monthly_revenue,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_at_risk,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge,
    ROUND(AVG(tenure), 1) AS avg_tenure_months
FROM CustomerSegments
GROUP BY spend_segment
ORDER BY avg_monthly_charge DESC;

-- 9. High-Risk Cohort Identification
-- Pinpoints specific cohort combinations experiencing disproportionate churn: Month-to-month + Fiber optic + Electronic check.
-- NAME: high_risk_cohort_breakdown
SELECT 
    Contract,
    InternetService,
    PaymentMethod,
    COUNT(*) AS cohort_size,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_count,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_revenue,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_lost
FROM customers
GROUP BY Contract, InternetService, PaymentMethod
HAVING COUNT(*) >= 50
ORDER BY churn_rate_pct DESC
LIMIT 10;

-- 10. Senior Citizen Churn Comparison
-- Compares churn profile and revenue metrics between Senior Citizens and Non-Senior Citizens.
-- NAME: senior_citizen_churn_profile
SELECT 
    CASE SeniorCitizen WHEN 1 THEN 'Senior Citizen' ELSE 'Non-Senior Citizen' END AS demographic_group,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_charges,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_lost,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge
FROM customers
GROUP BY SeniorCitizen;
