# CUSTOMER360 AI — Technical Interview Preparation Guide

This guide is designed to help you confidently explain the architecture, data engineering choices, machine learning methodology, and business impact of **CUSTOMER360 AI** during Data Science, Data Analyst, and Machine Learning internship and junior role interviews.

---

## 1. Project Elevator Pitch (30–45 Seconds)

> *"CUSTOMER360 AI is an end-to-end customer churn prediction and retention decision intelligence platform built on authentic IBM Telco customer data (7,043 enterprise accounts). Rather than just predicting whether a customer will churn, the platform bridges machine learning and business action: it deploys a calibrated XGBoost classifier (Test ROC-AUC 0.8467) that outputs well-calibrated probabilities, leverages native Tree SHAP for individual and global feature explanations, and powers an interactive decision-support layer that calculates monthly revenue exposure and simulates targeted retention campaign ROI. The entire workflow is backed by an automated data pipeline, SQLite analytical warehouse, and a 33-test automated test suite."*

---

## 2. Core Concepts & Common Interview Questions

### Q1: Why predict churn probability instead of just binary 0 or 1?
**Answer:**
- Binary classification outputs (0 or 1) force an arbitrary 0.50 threshold that ignores business trade-offs. In subscription businesses, customer acquisition cost (CAC) often exceeds $200–$300, while retention incentives may cost $20–$50.
- Calibrated probabilities allow operational teams to stratify accounts into granular risk tiers (**Critical: ≥75%**, **High: 50–75%**, **Medium: 25–50%**, **Low: <25%**).
- Probabilities also enable expected revenue exposure calculations:
  $$\text{Expected Annual Exposure} = P(\text{Churn}) \times (\text{MonthlyCharges} \times 12)$$
  This allows businesses to prioritize a high-value customer with 60% churn probability over a low-value customer with 80% churn probability.

### Q2: What is Probability Calibration, and why did you use `CalibratedClassifierCV`?
**Answer:**
- Many modern machine learning models (like XGBoost or Random Forests) produce well-ranked scores (high ROC-AUC) but poorly calibrated raw probabilities due to tree ensembling, regularization, and class weighting (`scale_pos_weight`).
- A raw score of 0.70 from an uncalibrated boosted tree might only correspond to a 45% empirical event rate.
- By applying **Sigmoidal Platt Scaling** (`CalibratedClassifierCV(method='sigmoid', cv=3)`), we mapped the model's margins to true posterior probabilities without distorting ranking.
- In our test set, calibration reduced the Brier score loss from **0.1627** down to **0.1361**, ensuring that out of 100 customers assigned a 70% risk, approximately 70 actually churn.

### Q3: How did you prevent data leakage during preprocessing?
**Answer:**
- All transformations that learn parameters from data—such as numerical scaling (`StandardScaler` computing mean and variance) and categorical encoding (`OneHotEncoder` learning category domains)—were wrapped in a scikit-learn `ColumnTransformer`.
- We performed a **Stratified Train/Test Split (80% train / 20% test)** first.
- The `ColumnTransformer` was fitted strictly on `X_train` (`fit_transform`) and only used to transform `X_test` (`transform`).
- Feature engineering (such as tenure cohorts and service add-on counts) was computed using only row-level attributes, guaranteeing zero cross-row or target leakage.

### Q4: How did you handle class imbalance in the dataset?
**Answer:**
- The IBM Telco dataset exhibits an empirical churn rate of **26.5%** (1,869 churned vs. 5,174 retained accounts).
- We addressed this using three complementary techniques:
  1. **Stratified Splitting:** Maintained identical class prevalence across 5-fold cross-validation folds and the held-out test partition.
  2. **Cost-Sensitive Learning / Class Weighting:** Configured `scale_pos_weight = (N_neg / N_pos) ≈ 2.76` in XGBoost and `class_weight='balanced'` in Random Forest and Logistic Regression. This penalizes false negatives proportionally.
  3. **Decision Threshold Tuning:** Rather than blindly using the default 0.50 cutoff, we evaluated decision thresholds from 0.10 to 0.90, optimizing for the business trade-off between Precision and Recall.

### Q5: How did you implement Explainable AI (XAI)?
**Answer:**
- We implemented **Tree SHAP (SHapley Additive exPlanations)** natively via XGBoost's C++ inference engine (`pred_contribs=True`).
- Tree SHAP decomposes each individual prediction into additive feature contributions relative to a baseline enterprise churn expectation:
  $$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$
- This provides:
  - **Local explanations:** For any individual customer, the platform highlights the exact top risk drivers (e.g., *Month-to-month contract*, *Fiber optic*, *Electronic check*) and protective factors (e.g., *Long tenure*, *Online Security enabled*).
  - **Global explanations:** Identifies the macro-drivers across the entire enterprise customer base.
- **Critical Interview Nuance:** We explicitly highlight **Correlation vs. Causation** in the UI. Statistical feature attributions reflect observational associations in historical data, not verified causal mechanisms; intervention effectiveness should always be validated via controlled A/B experiments.

### Q6: Why did you create an SQLite database instead of just analyzing a pandas DataFrame?
**Answer:**
- In production analytics engineering, customer data resides in data warehouses (Snowflake, BigQuery, PostgreSQL) rather than static CSV files in memory.
- Adding SQLite showcases:
  - Genuine SQL capabilities (CTE queries, window functions, conditional aggregations, cohort breakdowns).
  - Analytical indexing (`CREATE INDEX`) to accelerate dashboard aggregations.
  - Separation of concerns between the storage/analytical layer and the application UI.

---

## 3. Key Dataset Metrics to Remember

| Metric | Empirical Value |
|---|---|
| **Total Verified Accounts** | 7,043 customers |
| **Historical Churn Rate** | 26.54% (1,869 churned) |
| **Total Monthly Revenue (MRR)** | $456,116.60 / month |
| **Monthly Revenue at Risk** | $139,130.85 (30.5% of total billing) |
| **Month-to-Month Contract Churn** | **42.71%** (vs. 2.83% for Two-Year) |
| **Electronic Check Payment Churn** | **45.29%** (vs. ~16% for AutoPay) |
| **First-Year Customer Mortality (0–12 mos)** | **47.67%** (55.8% of all churn events) |
| **Test Set Size** | 1,409 unseen accounts (80/20 split) |
| **Production Model** | Calibrated XGBoost Classifier |
| **Test Set ROC-AUC** | **0.8467** |
| **Calibrated Brier Score** | **0.1361** |
| **Automated Test Suite** | 33 passing pytest tests |
