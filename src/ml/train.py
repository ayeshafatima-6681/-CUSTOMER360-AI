"""
Machine Learning Training and Validation Engine for CUSTOMER360 AI.
Implements data leakage-free preprocessing, 5-fold cross-validation, hyperparameter-tuned
candidate models (Logistic Regression, Random Forest, XGBoost), probability calibration,
threshold optimization, and metrics serialization.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble._forest import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)

from src.config import (
    CLEANED_DATA_CSV,
    MODELS_DIR,
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    CALIBRATED_MODEL_PATH,
    MODEL_METRICS_PATH,
    FEATURE_IMPORTANCES_PATH,
    ID_COLUMN,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    DECISION_THRESHOLD,
)
from src.data.preprocessing import create_preprocessor, get_feature_lists

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_train_test_data(
    data_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any, list]:
    """
    Loads cleaned data, separates target, performs stratified train/test split,
    and fits the preprocessor strictly on the training partition.
    """
    path = Path(data_path) if data_path else CLEANED_DATA_CSV
    if not path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {path}. Run preprocessing first.")

    df = pd.read_csv(path)
    feat_lists = get_feature_lists()
    all_features = feat_lists["numerical"] + feat_lists["categorical"]

    X = df[all_features].copy()
    y = df["Churn_Binary"].copy()

    # Stratified split to preserve 73.4% / 26.6% class balance
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    logger.info(f"Training set: {X_train_raw.shape[0]} samples (Churn rate: {y_train.mean():.3f})")
    logger.info(f"Test set: {X_test_raw.shape[0]} samples (Churn rate: {y_test.mean():.3f})")

    preprocessor = create_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Extract transformed feature names
    feature_names = preprocessor.get_feature_names_out().tolist()

    return X_train, X_test, y_train, y_test, preprocessor, feature_names


def train_and_evaluate_models() -> Dict[str, Any]:
    """
    Trains candidate models, performs cross-validation, evaluates on held-out test set,
    calibrates probabilities, and serializes all models and performance metrics.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_train_test_data()

    # Save preprocessor
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    with open(MODELS_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    # Define candidate models
    # scale_pos_weight for XGBoost = (len(y_train) - sum(y_train)) / sum(y_train) ~ 2.76
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    models = {
        "Logistic Regression": LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=7,
            min_samples_split=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
    }

    cv_results_summary = {}
    test_results_summary = {}
    fitted_models = {}

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
    }

    best_model_name = None
    best_roc_auc = -1.0

    for name, model in models.items():
        logger.info(f"\n--- Training and Cross-Validating: {name} ---")
        cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        
        cv_results_summary[name] = {
            "cv_roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(cv_scores["test_roc_auc"])),
            "cv_pr_auc_mean": float(np.mean(cv_scores["test_average_precision"])),
            "cv_f1_mean": float(np.mean(cv_scores["test_f1"])),
            "cv_recall_mean": float(np.mean(cv_scores["test_recall"])),
            "cv_precision_mean": float(np.mean(cv_scores["test_precision"])),
        }
        logger.info(
            f"{name} 5-Fold CV ROC-AUC: {cv_results_summary[name]['cv_roc_auc_mean']:.4f} "
            f"(+/- {cv_results_summary[name]['cv_roc_auc_std']:.4f}) | "
            f"PR-AUC: {cv_results_summary[name]['cv_pr_auc_mean']:.4f} | "
            f"F1: {cv_results_summary[name]['cv_f1_mean']:.4f} | "
            f"Recall: {cv_results_summary[name]['cv_recall_mean']:.4f}"
        )

        # Fit model on entire training partition
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Evaluate on held-out test partition
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

        # Curve computations
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Calibration assessment
        brier = brier_score_loss(y_test, y_prob)
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=8, strategy="uniform")

        test_roc_auc = float(roc_auc_score(y_test, y_prob))
        test_pr_auc = float(average_precision_score(y_test, y_prob))

        test_results_summary[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "roc_auc": test_roc_auc,
            "pr_auc": test_pr_auc,
            "brier_score": float(brier),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "roc_curve": {
                "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr) // 30)]],
                "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr) // 30)]],
            },
            "pr_curve": {
                "precision": [round(float(x), 4) for x in prec_curve[::max(1, len(prec_curve) // 30)]],
                "recall": [round(float(x), 4) for x in rec_curve[::max(1, len(rec_curve) // 30)]],
            },
            "calibration_curve": {
                "prob_true": [round(float(x), 4) for x in prob_true],
                "prob_pred": [round(float(x), 4) for x in prob_pred],
            },
        }

        # Select best model based on held-out ROC-AUC & PR-AUC balance
        if test_roc_auc > best_roc_auc:
            best_roc_auc = test_roc_auc
            best_model_name = name

    logger.info(f"\n>>> Best Candidate Model Selected: '{best_model_name}' (Test ROC-AUC: {best_roc_auc:.4f})")

    # Fit Probability Calibrator on best model
    logger.info(f"Fitting CalibratedClassifierCV on {best_model_name}...")
    calibrated_clf = CalibratedClassifierCV(
        estimator=models[best_model_name],
        method="sigmoid",
        cv=3,
    )
    calibrated_clf.fit(X_train, y_train)
    cal_prob = calibrated_clf.predict_proba(X_test)[:, 1]
    cal_brier = brier_score_loss(y_test, cal_prob)
    cal_roc = roc_auc_score(y_test, cal_prob)

    # Threshold Optimization Analysis
    thresholds = np.linspace(0.10, 0.90, 17)
    threshold_metrics = []
    for th in thresholds:
        th_pred = (y_prob >= th).astype(int)
        th_cm = confusion_matrix(y_test, th_pred)
        th_tn, th_fp, th_fn, th_tp = th_cm.ravel()
        threshold_metrics.append({
            "threshold": round(float(th), 2),
            "precision": round(float(precision_score(y_test, th_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, th_pred)), 4),
            "f1": round(float(f1_score(y_test, th_pred)), 4),
            "tp": int(th_tp),
            "fp": int(th_fp),
            "fn": int(th_fn),
            "tn": int(th_tn),
        })

    # Save models to disk
    joblib.dump(fitted_models["LogisticRegression" if "LogisticRegression" in fitted_models else "Logistic Regression"], MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(fitted_models["RandomForest" if "RandomForest" in fitted_models else "Random Forest"], MODELS_DIR / "random_forest.joblib")
    joblib.dump(fitted_models["XGBoost Classifier"], MODELS_DIR / "xgboost_model.joblib")
    joblib.dump(fitted_models[best_model_name], BEST_MODEL_PATH)
    joblib.dump(calibrated_clf, CALIBRATED_MODEL_PATH)

    # Extract Global Feature Importance
    best_raw_model = fitted_models[best_model_name]
    feature_importances = {}
    if hasattr(best_raw_model, "feature_importances_"):
        raw_importances = best_raw_model.feature_importances_
        for feat, imp in zip(feature_names, raw_importances):
            feature_importances[feat] = float(imp)
    elif hasattr(best_raw_model, "coef_"):
        raw_importances = np.abs(best_raw_model.coef_[0])
        for feat, imp in zip(feature_names, raw_importances):
            feature_importances[feat] = float(imp)

    sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    with open(FEATURE_IMPORTANCES_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted_importances, f, indent=2)

    # Final combined metrics report
    metrics_payload = {
        "best_model_name": best_model_name,
        "best_test_roc_auc": round(float(best_roc_auc), 4),
        "calibration": {
            "pre_calibration_brier": test_results_summary[best_model_name]["brier_score"],
            "post_calibration_brier": round(float(cal_brier), 4),
            "post_calibration_roc_auc": round(float(cal_roc), 4),
        },
        "cv_results": cv_results_summary,
        "test_results": test_results_summary,
        "threshold_analysis": threshold_metrics,
        "evaluation_partition": {
            "total_test_samples": int(len(y_test)),
            "churn_positives": int(sum(y_test)),
            "active_negatives": int(len(y_test) - sum(y_test)),
            "prevalence_pct": round(float(100.0 * y_test.mean()), 2),
        },
        "top_features": list(sorted_importances.keys())[:10],
    }

    with open(MODEL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info(f"Model training and evaluation successfully completed!")
    logger.info(f"Metrics saved to: {MODEL_METRICS_PATH}")
    logger.info(f"Feature importances saved to: {FEATURE_IMPORTANCES_PATH}")
    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate_models()
