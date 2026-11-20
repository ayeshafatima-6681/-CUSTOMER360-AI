"""
Explainable AI (XAI) engine for CUSTOMER360 AI.
Provides local and global feature attribution using XGBoost native Tree SHAP (pred_contribs=True)
aggregated back to genuine customer-submitted features, accompanied by correlation vs. causation guidelines.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (
    MODELS_DIR,
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    FEATURE_IMPORTANCES_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Mapping from original categorical column to transformed dummy column names
CATEGORICAL_MAP = {
    "gender": ["gender_Male"],
    "SeniorCitizen": ["SeniorCitizen_1"],
    "Partner": ["Partner_Yes"],
    "Dependents": ["Dependents_Yes"],
    "PhoneService": ["PhoneService_Yes"],
    "MultipleLines": ["MultipleLines_No phone service", "MultipleLines_Yes"],
    "InternetService": ["InternetService_Fiber optic", "InternetService_No"],
    "OnlineSecurity": ["OnlineSecurity_No internet service", "OnlineSecurity_Yes"],
    "OnlineBackup": ["OnlineBackup_No internet service", "OnlineBackup_Yes"],
    "DeviceProtection": ["DeviceProtection_No internet service", "DeviceProtection_Yes"],
    "TechSupport": ["TechSupport_No internet service", "TechSupport_Yes"],
    "StreamingTV": ["StreamingTV_No internet service", "StreamingTV_Yes"],
    "StreamingMovies": ["StreamingMovies_No internet service", "StreamingMovies_Yes"],
    "Contract": ["Contract_One year", "Contract_Two year"],
    "PaperlessBilling": ["PaperlessBilling_Yes"],
    "PaymentMethod": [
        "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic check",
        "PaymentMethod_Mailed check",
    ],
}

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "security_services_count",
    "streaming_services_count",
    "total_addon_services",
]

FEATURE_DISPLAY_LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior Citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Lines",
    "InternetService": "Internet Technology",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "Contract": "Contract Commitment",
    "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Channel",
    "tenure": "Tenure",
    "MonthlyCharges": "Monthly Charges",
    "TotalCharges": "Total Charges",
    "security_services_count": "Security Add-ons",
    "streaming_services_count": "Streaming Add-ons",
    "total_addon_services": "Total Add-on Services",
}


def format_feature_value(col_name: str, val: Any) -> str:
    """Formats customer input values cleanly for display."""
    if col_name == "SeniorCitizen":
        return "Yes" if int(val) == 1 else "No"
    elif col_name == "tenure":
        return f"{int(val)} mos"
    elif col_name == "MonthlyCharges":
        return f"${float(val):.2f}/mo"
    elif col_name == "TotalCharges":
        return f"${float(val):.2f}"
    elif col_name in ["security_services_count", "streaming_services_count", "total_addon_services"]:
        return f"{int(val)} services"
    else:
        return str(val)


class ExplainabilityEngine:
    """
    Computes local and global explanations for churn predictions.
    Utilizes exact tree SHAP attributions directly from the trained XGBoost booster,
    aggregated back to the actual submitted customer features.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        preprocessor_path: Optional[Path] = None,
        feature_names_path: Optional[Path] = None,
    ):
        self.model_path = Path(model_path) if model_path else BEST_MODEL_PATH
        self.preprocessor_path = Path(preprocessor_path) if preprocessor_path else PREPROCESSOR_PATH
        self.feature_names_path = (
            Path(feature_names_path) if feature_names_path else MODELS_DIR / "feature_names.json"
        )

        if not self.model_path.exists() or not self.preprocessor_path.exists():
            raise FileNotFoundError("Model or preprocessor artifact missing. Run training first.")

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

        if self.feature_names_path.exists():
            with open(self.feature_names_path, "r", encoding="utf-8") as f:
                self.feature_names = json.load(f)
        else:
            self.feature_names = self.preprocessor.get_feature_names_out().tolist()

        # Cache global feature importances if available
        if FEATURE_IMPORTANCES_PATH.exists():
            with open(FEATURE_IMPORTANCES_PATH, "r", encoding="utf-8") as f:
                self.global_importances = json.load(f)
        else:
            self.global_importances = {}

    def get_global_importance(self, top_n: int = 15) -> List[Dict[str, Any]]:
        """
        Returns top global feature importances formatted for visualization.
        """
        items = list(self.global_importances.items())[:top_n]
        return [
            {
                "feature": k,
                "importance_score": round(float(v), 4),
                "relative_importance_pct": round(float(v * 100), 2),
            }
            for k, v in items
        ]

    def explain_instance(
        self,
        input_df: pd.DataFrame,
        top_k: int = 6,
        calibrated_probability: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generates individual prediction explanation with exact SHAP feature contributions,
        faithfully mapped to the actual customer-submitted features.
        
        Args:
            input_df: Single row pandas DataFrame with raw customer attributes.
            top_k: Number of positive and negative drivers to extract.
            calibrated_probability: Optional calibrated probability from CalibratedClassifierCV.
            
        Returns:
            Dict containing churn probabilities, base value, and top risk/protective drivers.
        """
        # Preprocess single record
        transformed_row = self.preprocessor.transform(input_df)

        # Predict raw probability from underlying booster
        raw_prob = float(self.model.predict_proba(transformed_row)[0, 1])

        # Compute native Tree SHAP using XGBoost booster
        booster = self.model.get_booster()
        dmat = xgb.DMatrix(transformed_row, feature_names=self.feature_names)
        # pred_contribs=True returns (1, n_features + 1) where last value is bias
        contribs = booster.predict(dmat, pred_contribs=True)[0]

        feature_contribs = contribs[:-1]
        base_value_margin = float(contribs[-1])
        base_probability = float(1.0 / (1.0 + np.exp(-base_value_margin)))

        # Aggregate dummy one-hot features back to genuine original customer attributes
        feat_scores = {}
        for f in NUMERICAL_FEATURES:
            if f in self.feature_names:
                idx = self.feature_names.index(f)
                feat_scores[f] = float(feature_contribs[idx])

        for orig_col, dummy_cols in CATEGORICAL_MAP.items():
            total_dummy_score = 0.0
            for dummy in dummy_cols:
                if dummy in self.feature_names:
                    idx = self.feature_names.index(dummy)
                    total_dummy_score += float(feature_contribs[idx])
            feat_scores[orig_col] = total_dummy_score

        # Build attributions with actual submitted customer values
        attributions = []
        for col_name, score in feat_scores.items():
            actual_val = input_df[col_name].iloc[0] if col_name in input_df.columns else "N/A"
            formatted_val = format_feature_value(col_name, actual_val)
            display_label = FEATURE_DISPLAY_LABELS.get(col_name, col_name)
            
            attributions.append({
                "raw_feature_name": col_name,
                "feature": f"{display_label}: {formatted_val}",
                "customer_value": formatted_val,
                "shap_contribution": round(float(score), 4),
                "impact": "Increases Risk" if score > 0 else "Reduces Risk",
            })

        # Separate into risk-increasing and protective factors
        risk_drivers = [a for a in attributions if a["shap_contribution"] > 0]
        protective_factors = [a for a in attributions if a["shap_contribution"] < 0]

        risk_drivers.sort(key=lambda x: x["shap_contribution"], reverse=True)
        protective_factors.sort(key=lambda x: x["shap_contribution"])

        # Create natural language narrative using actual customer features
        top_risk_names = [d["feature"] for d in risk_drivers[:3]]
        top_protect_names = [d["feature"] for d in protective_factors[:3]]

        effective_prob = calibrated_probability if calibrated_probability is not None else raw_prob

        narrative = (
            f"The customer exhibits a calibrated churn probability of {effective_prob * 100:.1f}% "
            f"(raw ensemble model score: {raw_prob * 100:.1f}%). "
            f"Primary factors driving churn risk upward include: "
            f"{', '.join(top_risk_names) if top_risk_names else 'None identified'}. "
            f"Key stabilizing factors offsetting churn risk include: "
            f"{', '.join(top_protect_names) if top_protect_names else 'None identified'}."
        )

        return {
            "calibrated_probability": round(effective_prob, 4),
            "raw_probability": round(raw_prob, 4),
            "predicted_probability": round(effective_prob, 4),
            "base_probability": round(base_probability, 4),
            "top_risk_drivers": risk_drivers[:top_k],
            "top_protective_factors": protective_factors[:top_k],
            "all_attributions": attributions,
            "narrative_summary": narrative,
            "causality_caveat": (
                "Correlation Notice: Feature attributions reflect empirical statistical associations "
                "in historical customer records, not proven causal levers. Business interventions should be "
                "evaluated through controlled retention experiments or A/B testing."
            ),
        }
