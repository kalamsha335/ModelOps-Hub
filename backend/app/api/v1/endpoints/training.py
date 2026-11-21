"""
ModelOps Hub - Training Runs History Endpoints
Lists historical ML training experiments, parameters, and evaluation metrics.
"""

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.models.schemas_db import TrainingRun

router = APIRouter(prefix="/training", tags=["Training Runs"])

@router.get("/history", status_code=status.HTTP_200_OK)
def get_training_history(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Retrieves logged training experiments across algorithms (Logistic Regression, XGBoost),
    including hyperparameters, PR-AUC, Recall, F1, and training duration.
    """
    runs = db.query(TrainingRun).order_by(TrainingRun.id.desc()).limit(limit).all()
    return [{
        "id": r.id,
        "run_id": r.run_id,
        "experiment_name": r.experiment_name,
        "algorithm": r.algorithm,
        "status": r.status,
        "parameters": r.parameters,
        "metrics": r.metrics,
        "duration_seconds": r.duration_seconds,
        "artifact_uri": r.artifact_uri,
        "created_at": r.created_at.isoformat() if r.created_at else None
    } for r in runs]
