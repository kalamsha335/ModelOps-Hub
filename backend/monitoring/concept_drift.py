"""
ModelOps Hub - Concept Drift & Model Performance Monitoring Service
Differentiates between Data Drift (P(X)) and Concept Drift (P(Y|X)).
Supports delayed ground-truth label ingestion and performance degradation alerting.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix
)
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.database.session import SessionLocal
from backend.app.models.schemas_db import PerformanceMetric, ModelVersion

logger = logging.getLogger("modelops.monitoring.concept_drift")

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class ConceptDriftDetector:
    """
    Monitors model performance over time and identifies concept drift (P(Y|X)).
    Handles both scenarios:
    1. Delayed Ground Truth: Monitors prediction probability distribution shifts P(Y_hat).
    2. Ground Truth Available: Evaluates actual classification metrics vs baseline and triggers alerts.
    """

    def __init__(self, performance_drop_threshold: float = 0.10):
        """
        :param performance_drop_threshold: Maximum tolerable relative drop in PR-AUC or Recall (default: 10%).
        """
        self.performance_drop_threshold = performance_drop_threshold

    def evaluate_prediction_drift_without_labels(
        self,
        predicted_probs: np.ndarray,
        reference_fraud_rate: float = 0.00172,
        reference_mean_prob: float = 0.002
    ) -> Dict[str, Any]:
        """
        Monitors output distribution P(Y_hat) when ground truth labels are pending/delayed.
        Provides early warning of possible concept drift before true labels arrive.
        """
        predicted_probs = np.asarray(predicted_probs, dtype=float)
        predicted_labels = (predicted_probs >= 0.5).astype(int)

        current_fraud_rate = float(np.mean(predicted_labels))
        current_mean_prob = float(np.mean(predicted_probs))
        current_std_prob = float(np.std(predicted_probs))

        # Relative shift in predicted fraud rate
        fraud_rate_ratio = current_fraud_rate / (reference_fraud_rate + 1e-6)

        if fraud_rate_ratio > 3.0 or fraud_rate_ratio < 0.2:
            pred_status = "WARNING"
            interpretation = (
                f"Significant prediction distribution shift: predicted fraud rate ({current_fraud_rate:.4%}) "
                f"is {fraud_rate_ratio:.1f}x reference rate ({reference_fraud_rate:.4%})."
            )
        else:
            pred_status = "HEALTHY"
            interpretation = "Prediction distribution consistent with reference baseline."

        return {
            "status": "AWAITING_GROUND_TRUTH_LABELS",
            "prediction_monitoring_status": pred_status,
            "sample_count": len(predicted_probs),
            "current_predicted_fraud_rate": round(current_fraud_rate, 5),
            "reference_fraud_rate": round(reference_fraud_rate, 5),
            "current_mean_probability": round(current_mean_prob, 5),
            "reference_mean_probability": round(reference_mean_prob, 5),
            "interpretation": interpretation
        }

    def evaluate_ground_truth_performance(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_version: str,
        baseline_metrics: Optional[Dict[str, float]] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Evaluates model quality with true labels, calculates degradation against reference metrics,
        and determines if Concept Drift occurred.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_pred_proba = np.asarray(y_pred_proba, dtype=float)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Calculate metrics
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # ROC-AUC & PR-AUC
        try:
            roc = float(roc_auc_score(y_true, y_pred_proba))
        except Exception:
            roc = 0.5

        try:
            p_curve, r_curve, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = float(auc(r_curve, p_curve))
        except Exception:
            pr_auc = 0.0

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        # Fetch baseline metrics from DB if not provided
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            if baseline_metrics is None:
                db_ver = db.query(ModelVersion).filter(ModelVersion.version == str(model_version)).first()
                if db_ver and db_ver.metrics:
                    baseline_metrics = db_ver.metrics
                else:
                    baseline_metrics = {
                        "pr_auc": settings.RETRAINING_MIN_PR_AUC_THRESHOLD,
                        "recall": settings.RETRAINING_MIN_RECALL_THRESHOLD,
                        "f1_score": 0.80
                    }

            base_pr_auc = float(baseline_metrics.get("pr_auc", 0.80))
            base_recall = float(baseline_metrics.get("recall", 0.75))

            # Calculate degradation drops
            pr_auc_drop = max(0.0, (base_pr_auc - pr_auc) / (base_pr_auc + 1e-6))
            recall_drop = max(0.0, (base_recall - rec) / (base_recall + 1e-6))

            has_degradation = (
                pr_auc_drop >= self.performance_drop_threshold or
                recall_drop >= self.performance_drop_threshold
            )

            if has_degradation:
                status = "CRITICAL_CONCEPT_DRIFT"
                interpretation = (
                    f"Concept drift detected! PR-AUC dropped by {pr_auc_drop:.1%} "
                    f"and Recall dropped by {recall_drop:.1%}. Automated retraining required."
                )
            else:
                status = "HEALTHY"
                interpretation = "Model performance is stable within tolerance thresholds."

            # Persist to PerformanceMetric table
            perf_record = PerformanceMetric(
                model_version=str(model_version),
                sample_count=len(y_true),
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1_score=round(f1, 4),
                roc_auc=round(roc, 4),
                pr_auc=round(pr_auc, 4),
                true_positives=tp,
                false_positives=fp,
                true_negatives=tn,
                false_negatives=fn,
                recorded_at=utc_now()
            )
            db.add(perf_record)
            db.commit()
            db.refresh(perf_record)

            logger.info(f"Concept Drift evaluation: Version {model_version}, PR-AUC={pr_auc:.4f}, Status={status}")

            return {
                "id": perf_record.id,
                "model_version": str(model_version),
                "status": status,
                "has_concept_drift": has_degradation,
                "metrics": {
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1_score": round(f1, 4),
                    "roc_auc": round(roc, 4),
                    "pr_auc": round(pr_auc, 4),
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn
                },
                "baseline_metrics": {
                    "pr_auc": base_pr_auc,
                    "recall": base_recall
                },
                "degradation": {
                    "pr_auc_drop": round(pr_auc_drop, 4),
                    "recall_drop": round(recall_drop, 4),
                    "threshold": self.performance_drop_threshold
                },
                "interpretation": interpretation,
                "recorded_at": perf_record.recorded_at.isoformat()
            }
        finally:
            if close_db:
                db.close()

    def get_performance_history(self, model_version: Optional[str] = None, limit: int = 20, db: Optional[Session] = None) -> List[Dict[str, Any]]:
        """Retrieves performance evaluation history."""
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            query = db.query(PerformanceMetric)
            if model_version:
                query = query.filter(PerformanceMetric.model_version == str(model_version))
            records = query.order_by(PerformanceMetric.id.desc()).limit(limit).all()

            return [{
                "id": r.id,
                "model_version": r.model_version,
                "sample_count": r.sample_count,
                "precision": r.precision,
                "recall": r.recall,
                "f1_score": r.f1_score,
                "pr_auc": r.pr_auc,
                "roc_auc": r.roc_auc,
                "tp": r.true_positives,
                "fp": r.false_positives,
                "tn": r.true_negatives,
                "fn": r.false_negatives,
                "recorded_at": r.recorded_at.isoformat() if r.recorded_at else None
            } for r in records]
        finally:
            if close_db:
                db.close()

# Module-level instance
concept_drift_detector = ConceptDriftDetector()

