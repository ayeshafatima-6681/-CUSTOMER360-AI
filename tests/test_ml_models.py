"""
Tests for serialized ML models, cross-validation metrics, and probability calibration.
"""

import json
import pytest
import joblib
from pathlib import Path

from src.config import (
    BEST_MODEL_PATH,
    CALIBRATED_MODEL_PATH,
    PREPROCESSOR_PATH,
    MODEL_METRICS_PATH,
    FEATURE_IMPORTANCES_PATH,
)


def test_model_artifacts_exist():
    """Verify that all trained model and preprocessor artifacts exist on disk."""
    assert BEST_MODEL_PATH.exists()
    assert CALIBRATED_MODEL_PATH.exists()
    assert PREPROCESSOR_PATH.exists()
    assert MODEL_METRICS_PATH.exists()
    assert FEATURE_IMPORTANCES_PATH.exists()


def test_model_metrics_payload():
    """Verify metrics JSON contains genuine evaluation scores on unseen test data."""
    with open(MODEL_METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "best_model_name" in metrics
    assert "best_test_roc_auc" in metrics
    assert metrics["best_test_roc_auc"] > 0.80  # Demonstrating strong discriminative performance

    # Test calibration metrics
    calib = metrics["calibration"]
    assert "post_calibration_brier" in calib
    assert calib["post_calibration_brier"] < 0.20

    # Test CV results exist for all 3 models
    cv_res = metrics["cv_results"]
    for m in ["Logistic Regression", "Random Forest", "XGBoost Classifier"]:
        assert m in cv_res
        assert cv_res[m]["cv_roc_auc_mean"] > 0.80
        assert cv_res[m]["cv_recall_mean"] > 0.70

    # Test confusion matrix structure
    test_res = metrics["test_results"]
    for m in test_res:
        cm = test_res[m]["confusion_matrix"]
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == 1409  # Total test samples


def test_feature_importances_payload():
    """Verify feature importances are non-empty and contract features rank high."""
    with open(FEATURE_IMPORTANCES_PATH, "r", encoding="utf-8") as f:
        importances = json.load(f)

    assert len(importances) > 10
    top_feature = list(importances.keys())[0]
    # Two year contract or fiber optic is top driver
    assert any(k in top_feature for k in ["Contract", "InternetService"])
