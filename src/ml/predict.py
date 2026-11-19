"""
Customer Churn Prediction and Scoring Service for CUSTOMER360 AI.
Performs single-customer scoring and batch scoring using the calibrated ML pipeline,
enriching predictions with risk tiers, financial exposures, and prescriptive recommendations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    CALIBRATED_MODEL_PATH,
    PREPROCESSOR_PATH,
    MODELS_DIR,
    DECISION_THRESHOLD,
)
from src.data.preprocessing import get_feature_lists
from src.decision_intelligence import (
    classify_risk_tier,
    calculate_customer_value_metrics,
    recommend_retention_actions,
)
from src.explainability import ExplainabilityEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ChurnPredictor:
    """
    Production-grade customer churn prediction service.
    Loads calibrated model artifacts and provides single and batch inference.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        calibrated_model_path: Optional[Path] = None,
        preprocessor_path: Optional[Path] = None,
    ):
        self.model_path = Path(model_path) if model_path else BEST_MODEL_PATH
        self.calibrated_path = Path(calibrated_model_path) if calibrated_model_path else CALIBRATED_MODEL_PATH
        self.preprocessor_path = Path(preprocessor_path) if preprocessor_path else PREPROCESSOR_PATH

        if not self.model_path.exists() or not self.preprocessor_path.exists():
            raise FileNotFoundError("Model artifacts missing. Please run model training first.")

        self.raw_model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        
        # Use calibrated model for reliable real-world probabilities if available
        if self.calibrated_path.exists():
            self.model = joblib.load(self.calibrated_path)
            logger.info("Loaded CalibratedClassifierCV for probability inference.")
        else:
            self.model = self.raw_model
            logger.info("Loaded raw best model for probability inference.")

        # Initialize explainability engine
        self.explainer = ExplainabilityEngine(
            model_path=self.model_path,
            preprocessor_path=self.preprocessor_path,
        )

        self.feature_lists = get_feature_lists()

    def _prepare_input_df(self, input_data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """
        Validates and formats incoming customer data to ensure all expected features exist.
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()

        # Handle derived counts if not supplied by the caller
        if "security_services_count" not in df.columns:
            sec_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
            present_sec = [c for c in sec_cols if c in df.columns]
            if present_sec:
                df["security_services_count"] = df[present_sec].apply(
                    lambda r: sum(val == "Yes" for val in r), axis=1
                )
            else:
                df["security_services_count"] = 0

        if "streaming_services_count" not in df.columns:
            stream_cols = ["StreamingTV", "StreamingMovies"]
            present_stream = [c for c in stream_cols if c in df.columns]
            if present_stream:
                df["streaming_services_count"] = df[stream_cols].apply(
                    lambda r: sum(val == "Yes" for val in r), axis=1
                )
            else:
                df["streaming_services_count"] = 0

        if "total_addon_services" not in df.columns:
            df["total_addon_services"] = df["security_services_count"] + df["streaming_services_count"]

        # Ensure correct numeric types
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0).astype(float)
        
        if "TotalCharges" not in df.columns or df["TotalCharges"].isna().all():
            df["TotalCharges"] = df["MonthlyCharges"] * df["tenure"]
        else:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0).astype(float)

        return df

    def predict_single(
        self,
        customer_record: Dict[str, Any],
        threshold: float = DECISION_THRESHOLD,
        include_explanation: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs full inference on a single customer record.
        Returns probability, risk tier, revenue exposure, prescriptive actions, and SHAP explanation.
        """
        df = self._prepare_input_df(customer_record)
        
        # Transform features and predict probability
        X_trans = self.preprocessor.transform(df)
        prob = float(self.model.predict_proba(X_trans)[0, 1])
        prediction = 1 if prob >= threshold else 0
        risk_tier = classify_risk_tier(prob)

        # Financial value metrics
        m_charges = float(df["MonthlyCharges"].iloc[0])
        t_charges = float(df["TotalCharges"].iloc[0])
        tenure_val = int(df["tenure"].iloc[0])
        value_metrics = calculate_customer_value_metrics(m_charges, t_charges, tenure_val, prob)

        # Retention action recommendations
        actions = recommend_retention_actions(customer_record, prob, risk_tier)

        response = {
            "churn_prediction": "Churn" if prediction == 1 else "Retained",
            "prediction_binary": prediction,
            "churn_probability": round(prob, 4),
            "churn_risk_score_pct": round(prob * 100.0, 1),
            "risk_tier": risk_tier,
            "threshold_used": threshold,
            "value_metrics": value_metrics,
            "recommended_actions": actions,
        }

        if include_explanation:
            explanation = self.explainer.explain_instance(df)
            response["explanation"] = explanation

        return response

    def predict_batch(
        self,
        batch_df: pd.DataFrame,
        threshold: float = DECISION_THRESHOLD,
    ) -> pd.DataFrame:
        """
        Scores a batch of customers, appending churn probabilities, risk tiers, and revenue exposure.
        """
        df = self._prepare_input_df(batch_df)
        X_trans = self.preprocessor.transform(df)
        probs = self.model.predict_proba(X_trans)[:, 1]

        result_df = batch_df.copy()
        result_df["churn_probability"] = np.round(probs, 4)
        result_df["churn_risk_pct"] = np.round(probs * 100.0, 1)
        result_df["churn_prediction"] = (probs >= threshold).astype(int)
        result_df["churn_prediction_label"] = np.where(probs >= threshold, "Churn", "Retained")
        result_df["risk_tier"] = [classify_risk_tier(p) for p in probs]
        
        # Revenue exposure = churn_prob * (MonthlyCharges * 12)
        result_df["annual_revenue_exposure_usd"] = np.round(
            result_df["churn_probability"] * (result_df["MonthlyCharges"].astype(float) * 12.0), 2
        )

        return result_df
