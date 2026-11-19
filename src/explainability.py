"""
Explainable AI (XAI) engine for CUSTOMER360 AI.
Provides local and global feature attribution using XGBoost native Tree SHAP (pred_contribs=True)
and Logistic Regression log-odds coefficients, accompanied by correlation vs. causation guidelines.
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


class ExplainabilityEngine:
    """
    Computes local and global explanations for churn predictions.
    Utilizes exact tree SHAP attributions directly from the trained XGBoost booster.
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
        top_k: int = 6
    ) -> Dict[str, Any]:
        """
        Generates individual prediction explanation with exact SHAP feature contributions.
        
        Args:
            input_df: Single row pandas DataFrame with raw customer attributes.
            top_k: Number of positive and negative drivers to extract.
            
        Returns:
            Dict containing churn probability, base value, and top risk/protective drivers.
        """
        # Preprocess single record
        transformed_row = self.preprocessor.transform(input_df)

        # Predict probability
        prob = float(self.model.predict_proba(transformed_row)[0, 1])

        # Compute native Tree SHAP using XGBoost booster
        booster = self.model.get_booster()
        dmat = xgb.DMatrix(transformed_row, feature_names=self.feature_names)
        # pred_contribs=True returns (1, n_features + 1) where last value is bias
        contribs = booster.predict(dmat, pred_contribs=True)[0]

        feature_contribs = contribs[:-1]
        base_value_margin = float(contribs[-1])
        base_probability = float(1.0 / (1.0 + np.exp(-base_value_margin)))

        attributions = []
        for name, score in zip(self.feature_names, feature_contribs):
            attributions.append({
                "feature": name,
                "shap_contribution": float(score),
                "impact": "Increases Risk" if score > 0 else "Reduces Risk",
            })

        # Separate into risk-increasing and protective factors
        risk_drivers = [a for a in attributions if a["shap_contribution"] > 0]
        protective_factors = [a for a in attributions if a["shap_contribution"] < 0]

        risk_drivers.sort(key=lambda x: x["shap_contribution"], reverse=True)
        protective_factors.sort(key=lambda x: x["shap_contribution"])

        # Create natural language narrative
        top_risk_names = [d["feature"].replace("_", " = ") for d in risk_drivers[:3]]
        top_protect_names = [d["feature"].replace("_", " = ") for d in protective_factors[:3]]

        narrative = (
            f"The customer exhibits a predicted churn probability of {prob * 100:.1f}%. "
            f"Primary factors driving churn risk upward include: "
            f"{', '.join(top_risk_names) if top_risk_names else 'None identified'}. "
            f"Key stabilizing factors offsetting churn risk include: "
            f"{', '.join(top_protect_names) if top_protect_names else 'None identified'}."
        )

        return {
            "predicted_probability": round(prob, 4),
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
