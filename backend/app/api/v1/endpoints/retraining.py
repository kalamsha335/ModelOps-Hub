"""
ModelOps Hub - Retraining Endpoints
Allows triggering automated retraining flows and viewing retraining history with guardrail decisions.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, status
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.schemas.schemas_api import RetrainingTriggerRequest
from backend.app.models.schemas_db import RetrainingJob
from backend.workflows.retraining_flow import execute_retraining_workflow

router = APIRouter(prefix="/retraining", tags=["Retraining"])

@router.post("/trigger", status_code=status.HTTP_200_OK)
def trigger_retraining(
    payload: RetrainingTriggerRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Triggers an automated retraining job with safeguard guardrails.
    Checks for active jobs, cooldown period, and prevents replacing production with an inferior model.
    """
    try:
        # Check active jobs before scheduling
        active = db.query(RetrainingJob).filter(RetrainingJob.status.in_(["PENDING", "RUNNING"])).first()
        if active and not payload.force:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Retraining rejected: Job {active.job_id} is already in progress."
            )

        # Execute retraining synchronously or via background task
        # For immediate feedback in dashboard, execute directly if quick_run
        if payload.quick_run:
            res = execute_retraining_workflow(
                trigger_reason=payload.trigger_reason,
                force=payload.force,
                quick_run=True,
                optuna_trials=payload.optuna_trials
            )
            return res
        else:
            # Full training scheduled as background task
            background_tasks.add_task(
                execute_retraining_workflow,
                trigger_reason=payload.trigger_reason,
                force=payload.force,
                quick_run=False,
                optuna_trials=payload.optuna_trials
            )
            return {
                "status": "ACCEPTED",
                "message": f"Full retraining workflow initiated for trigger: {payload.trigger_reason}",
                "trigger_reason": payload.trigger_reason
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/history", status_code=status.HTTP_200_OK)
def list_retraining_history(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Returns history of retraining runs, candidate metrics, production metrics,
    and Champion vs Challenger guardrail decisions.
    """
    jobs = db.query(RetrainingJob).order_by(RetrainingJob.id.desc()).limit(limit).all()
    return [{
        "id": j.id,
        "job_id": j.job_id,
        "trigger_reason": j.trigger_reason,
        "status": j.status,
        "previous_model_version": j.previous_model_version,
        "new_model_version": j.new_model_version,
        "candidate_metrics": j.candidate_metrics,
        "production_metrics": j.production_metrics,
        "promotion_decision": j.promotion_decision,
        "logs": j.logs,
        "started_at": j.started_at.isoformat() if j.started_at else None,
        "completed_at": j.completed_at.isoformat() if j.completed_at else None
    } for j in jobs]
