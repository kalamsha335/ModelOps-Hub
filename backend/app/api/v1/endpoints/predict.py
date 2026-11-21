"""
ModelOps Hub - Prediction Endpoints
Serves single and batch fraud predictions with dynamic model loading,
latency monitoring, and prediction logging.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.schemas.schemas_api import (
    TransactionFeatures, BatchTransactionInput,
    PredictionResponse, BatchPredictionResponse,
    GroundTruthFeedbackRequest
)
from backend.app.services.serving import serving_service
from backend.app.models.schemas_db import PredictionLog

router = APIRouter(prefix="", tags=["Prediction"])

@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_transaction(
    payload: TransactionFeatures,
    db: Session = Depends(get_db)
):
    """
    Evaluates a single credit card transaction for fraudulent activity.
    Returns fraud classification, probability, serving model version, and inference latency.
    """
    try:
        response = serving_service.predict_single(payload, db=db)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@router.post("/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
def predict_batch_transactions(
    payload: BatchTransactionInput,
    db: Session = Depends(get_db)
):
    """
    Processes a batch of transactions in a single vectorized pass.
    """
    try:
        response = serving_service.predict_batch(payload.transactions, db=db)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}"
        )

@router.post("/feedback", status_code=status.HTTP_200_OK)
def ingest_ground_truth_feedback(
    payload: GroundTruthFeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Ingests delayed ground-truth chargeback feedback for a previously served transaction.
    Matches prediction log by ID and records whether fraud actually occurred.
    """
    log_record = db.query(PredictionLog).filter(PredictionLog.id == payload.prediction_log_id).first()
    if not log_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction log not found")

    log_record.ground_truth = payload.ground_truth
    db.commit()

    return {
        "status": "SUCCESS",
        "prediction_log_id": log_record.id,
        "recorded_ground_truth": payload.ground_truth,
        "original_prediction": log_record.prediction,
        "is_correct": bool(log_record.prediction == payload.ground_truth)
    }

@router.get("/logs", status_code=status.HTTP_200_OK)
def get_prediction_logs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Returns recent prediction logs for auditing and monitoring.
    """
    total = db.query(PredictionLog).count()
    logs = db.query(PredictionLog).order_by(PredictionLog.id.desc()).offset(offset).limit(limit).all()

    return {
        "total_records": total,
        "limit": limit,
        "offset": offset,
        "records": [{
            "id": l.id,
            "model_name": l.model_name,
            "model_version": l.model_version,
            "prediction": l.prediction,
            "prediction_label": l.prediction_label,
            "fraud_probability": l.fraud_probability,
            "latency_ms": l.latency_ms,
            "ground_truth": l.ground_truth,
            "logged_at": l.logged_at.isoformat() if l.logged_at else None
        } for l in logs]
    }
