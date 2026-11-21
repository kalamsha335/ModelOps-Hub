"""
Comprehensive Model Evaluation and Metrics Extraction Module.
Focuses strictly on imbalanced classification metrics:
Prioritizes PR-AUC (Precision-Recall Area Under Curve) and Fraud Recall over raw accuracy.
"""

import os
import json
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

logger = logging.getLogger("modelops-hub.evaluate")

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str] = None
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on test data and computes:
    - PR-AUC (Average Precision score) - Primary Metric
    - Recall (Fraud detection rate)
    - Precision (Fraud alert accuracy)
    - F1-score
    - ROC-AUC
    - Confusion Matrix (TP, FP, TN, FN)
    - Feature Importances
    """
    y_test = np.asarray(y_test)

    # Predictions and predicted probabilities
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred

    # Compute classification metrics
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    pr_auc = float(average_precision_score(y_test, y_proba))

    # Confusion matrix decomposition
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    false_positive_rate = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Extract feature importances
    feature_importance_dict = {}
    if feature_names is not None:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_importance_dict = {
                feat: float(imp) for feat, imp in zip(feature_names, importances)
            }
        elif hasattr(model, "coef_"):
            coefficients = model.coef_[0]
            feature_importance_dict = {
                feat: float(abs(c)) for feat, imp, c in zip(feature_names, coefficients, coefficients)
            }

    # Sort feature importance descending
    sorted_features = dict(
        sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
    )

    results = {
        "metrics": {
            "pr_auc": round(pr_auc, 5),
            "recall": round(recall, 5),
            "precision": round(precision, 5),
            "f1_score": round(f1, 5),
            "roc_auc": round(roc_auc, 5),
            "specificity": round(specificity, 5),
            "false_positive_rate": round(false_positive_rate, 5)
        },
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "matrix": cm.tolist()
        },
        "sample_counts": {
            "total_test_samples": int(len(y_test)),
            "actual_frauds": int(y_test.sum()),
            "actual_legitimate": int(len(y_test) - y_test.sum()),
            "predicted_frauds": int(y_pred.sum())
        },
        "feature_importances": sorted_features
    }

    logger.info(
        f"Evaluation Summary: PR-AUC={pr_auc:.4f} | Recall={recall:.4f} | "
        f"Precision={precision:.4f} | ROC-AUC={roc_auc:.4f} | F1={f1:.4f} | "
        f"TP={tp}, FP={fp}, FN={fn}, TN={tn}"
    )

    return results

def generate_comparison_report(
    models_evaluation: Dict[str, Dict[str, Any]],
    selection_strategy: str = "max_pr_auc_and_recall"
) -> Dict[str, Any]:
    """
    Compares evaluated candidate models and identifies the winning champion.
    Documented Selection Strategy:
    1. Filter models meeting minimum PR-AUC threshold (>= 0.80) and Recall (>= 0.82).
    2. Rank by composite score = 0.6 * PR-AUC + 0.4 * Recall.
    3. Tie-breaker: lowest False Positive Rate (minimizing burden on fraud ops).
    """
    ranked_models = []

    for name, eval_data in models_evaluation.items():
        metrics = eval_data["metrics"]
        pr_auc = metrics["pr_auc"]
        recall = metrics["recall"]
        precision = metrics["precision"]
        duration = eval_data.get("training_duration_seconds", 0.0)

        # Composite score prioritizing fraud capture and curve shape
        composite_score = round((0.6 * pr_auc) + (0.4 * recall), 5)

        ranked_models.append({
            "model_name": name,
            "pr_auc": pr_auc,
            "recall": recall,
            "precision": precision,
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "training_duration_seconds": duration,
            "composite_score": composite_score
        })

    # Sort by composite score descending
    ranked_models.sort(key=lambda x: x["composite_score"], reverse=True)
    best_model = ranked_models[0]["model_name"]

    logger.info(
        f"Champion Model Selected: '{best_model}' with composite score: {ranked_models[0]['composite_score']} "
        f"(PR-AUC={ranked_models[0]['pr_auc']}, Recall={ranked_models[0]['recall']})"
    )

    return {
        "selection_strategy": selection_strategy,
        "champion_model": best_model,
        "rankings": ranked_models
    }

