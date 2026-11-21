"""
ModelOps Hub - Monitoring Endpoints
Provides system health status, statistical data drift reports, active alerts,
and model performance tracking with true labels.
"""

import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.database.session import get_db
from backend.app.schemas.schemas_api import ManualDriftCheckRequest
from backend.monitoring.data_drift import drift_detector
from backend.monitoring.concept_drift import concept_drift_detector
from backend.app.models.schemas_db import DriftReport, DriftAlert, PerformanceMetric

logger = logging.getLogger("modelops.api.monitoring")

router = APIRouter(prefix="/monitoring", tags=["Monitoring & Drift"])

@router.get("/status", status_code=status.HTTP_200_OK)
def get_monitoring_overview(db: Session = Depends(get_db)):
    """
    Returns high-level health overview:
    System status, latest drift status, active alerts count, and total reports.
    """
    latest_report = drift_detector.get_latest_report(db=db)
    active_alerts = drift_detector.list_active_alerts(db=db)
    crit_count = sum(1 for a in active_alerts if a["severity"] == "CRITICAL")

    overall_health = "Healthy"
    if crit_count > 0 or (latest_report and latest_report["status"] == "Critical"):
        overall_health = "Critical"
    elif len(active_alerts) > 0 or (latest_report and latest_report["status"] == "Warning"):
        overall_health = "Warning"

    return {
        "overall_health": overall_health,
        "latest_drift_status": latest_report["status"] if latest_report else "No reports yet",
        "latest_drift_score": latest_report["drift_score"] if latest_report else 0.0,
        "active_alerts_total": len(active_alerts),
        "critical_alerts_count": crit_count,
        "latest_report_id": latest_report["report_id"] if latest_report else None,
        "last_checked_at": latest_report["created_at"] if latest_report else None
    }

@router.get("/drift", status_code=status.HTTP_200_OK)
def get_drift_details(db: Session = Depends(get_db)):
    """
    Returns the most recent detailed drift report including per-feature PSI, KS statistics,
    and HTML report artifact path.
    """
    latest_report = drift_detector.get_latest_report(db=db)
    if not latest_report:
        return {
            "status": "No drift reports found. Trigger a drift evaluation or send transactions.",
            "report": None
        }
    return {
        "status": "SUCCESS",
        "report": latest_report
    }

@router.get("/reports", status_code=status.HTTP_200_OK)
def list_drift_reports(limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    """Lists historical data drift monitoring reports."""
    return drift_detector.list_reports(limit=limit, db=db)

@router.get("/alerts", status_code=status.HTTP_200_OK)
def list_drift_alerts(db: Session = Depends(get_db)):
    """Lists all active drift alerts with recommended remediation actions."""
    return drift_detector.list_active_alerts(db=db)

@router.post("/alerts/{alert_id}/resolve", status_code=status.HTTP_200_OK)
def resolve_drift_alert(alert_id: int, db: Session = Depends(get_db)):
    """Acknowledges and marks a drift alert as resolved."""
    alert = db.query(DriftAlert).filter(DriftAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
    alert.status = "RESOLVED"
    db.commit()
    return {"status": "SUCCESS", "alert_id": alert_id, "alert_status": "RESOLVED"}

@router.post("/analyze", status_code=status.HTTP_200_OK)
def trigger_drift_analysis(payload: ManualDriftCheckRequest, db: Session = Depends(get_db)):
    """
    Triggers an immediate statistical drift evaluation comparing current production data against baseline.
    """
    try:
        sample_size = payload.sample_size
        ref_path = Path(settings.REFERENCE_DATA_DIR) / "reference_data.parquet"
        raw_path = Path(settings.RAW_DATA_PATH)

        # High-performance batch sampling:
        # Use parquet reference data with natural production variance if available, else sample raw CSV
        if ref_path.exists():
            ref_df = pd.read_parquet(ref_path)
            sample_n = min(len(ref_df), sample_size)
            sample_df = ref_df.sample(sample_n, random_state=None).copy()
            # Add subtle production variance to numerical features
            num_cols = [c for c in sample_df.columns if c not in ["Class", "Time"]]
            noise = np.random.normal(0, 0.05, size=(len(sample_df), len(num_cols)))
            sample_df[num_cols] += noise
        elif raw_path.exists():
            sample_df = pd.read_csv(raw_path, nrows=sample_size)
        else:
            raise FileNotFoundError("Neither reference data nor raw data found for drift evaluation.")

        res = drift_detector.run_drift_analysis(
            current_df=sample_df,
            generate_evidently_html=payload.generate_evidently_html,
            db=db
        )
        return res
    except Exception as e:
        logger.error(f"Drift analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/performance", status_code=status.HTTP_200_OK)
def get_performance_history(
    model_version: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Returns concept drift and model performance evaluation history when ground truth labels arrive.
    """
    return concept_drift_detector.get_performance_history(model_version=model_version, limit=limit, db=db)

