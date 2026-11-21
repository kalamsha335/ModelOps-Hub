"""
ModelOps Hub - Automated Retraining Workflow with Production Guardrails
Enforces duplicate prevention, cooldown protection, candidate validation,
and Champion vs Challenger evaluation to prevent replacing a healthy production model with an inferior model.
"""

import os
import uuid
import time
import logging
import traceback
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from prefect import flow, task
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.database.session import SessionLocal
from backend.app.models.schemas_db import RetrainingJob, ModelVersion
from backend.app.registry.model_registry import registry_service, ModelValidationError
from backend.training.train import run_training_pipeline

logger = logging.getLogger("modelops.workflows.retraining")

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def check_guardrails(trigger_reason: str, force: bool = False) -> Dict[str, Any]:
    """
    Safeguards:
    1. Prevents duplicate concurrent retraining jobs.
    2. Respects cooldown interval between consecutive retraining runs.
    """
    db = SessionLocal()
    try:
        # Check active runs
        active_job = db.query(RetrainingJob).filter(
            RetrainingJob.status.in_(["PENDING", "RUNNING"])
        ).first()

        if active_job and not force:
            raise ValueError(
                f"Retraining rejected: Job '{active_job.job_id}' is currently running (Trigger: {active_job.trigger_reason})."
            )

        # Check cooldown (unless force=True or trigger_reason == MANUAL)
        if not force and trigger_reason != "MANUAL":
            last_job = db.query(RetrainingJob).filter(
                RetrainingJob.status == "COMPLETED"
            ).order_by(RetrainingJob.id.desc()).first()

            if last_job and last_job.completed_at:
                cooldown_delta = timedelta(hours=settings.MAX_RETRAINING_COOLDOWN_HOURS)
                elapsed = utc_now() - last_job.completed_at
                if elapsed < cooldown_delta:
                    raise ValueError(
                        f"Retraining rejected: In cooldown period ({elapsed.total_seconds() / 3600:.1f}h elapsed, cooldown is {settings.MAX_RETRAINING_COOLDOWN_HOURS}h)."
                    )

        # Get current production model version
        prod_ver = registry_service.get_current_production_version(settings.MLFLOW_MODEL_NAME, db=db)
        prod_version_str = prod_ver["version"] if prod_ver else None

        job_id = f"retrain_{uuid.uuid4().hex[:12]}"
        new_job = RetrainingJob(
            job_id=job_id,
            trigger_reason=trigger_reason,
            status="RUNNING",
            previous_model_version=prod_version_str,
            started_at=utc_now()
        )
        db.add(new_job)
        db.commit()
        db.refresh(new_job)

        logger.info(f"Guardrails passed. Created retraining job {job_id} (Trigger: {trigger_reason})")
        return {
            "job_id": job_id,
            "previous_model_version": prod_version_str,
            "trigger_reason": trigger_reason
        }
    finally:
        db.close()

def train_candidates(job_info: Dict[str, Any], quick_run: bool = False, optuna_trials: int = 5) -> Dict[str, Any]:
    logger.info(f"Retraining models for job {job_info['job_id']}...")
    pipeline_result = run_training_pipeline(
        data_path=settings.RAW_DATA_PATH,
        optuna_trials=optuna_trials,
        quick_run=quick_run
    )
    return pipeline_result

def register_and_validate(job_info: Dict[str, Any], pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
    model_name = settings.MLFLOW_MODEL_NAME
    champion_algo = pipeline_result["champion_model"]
    champion_eval = pipeline_result["models_evaluated"][champion_algo]
    run_id = pipeline_result["mlflow_runs"][champion_algo]["run_id"]

    # Register candidate version
    reg_res = registry_service.register_model_version(
        run_id=run_id,
        model_name=model_name,
        description=f"Retrained candidate ({champion_algo}) from job {job_info['job_id']}"
    )
    new_version = reg_res["version"]

    # Validate against baseline thresholds
    val_res = registry_service.validate_candidate_for_production(
        model_name=model_name,
        version=new_version
    )

    return {
        "model_name": model_name,
        "new_version": new_version,
        "algorithm": champion_algo,
        "metrics": champion_eval["metrics"],
        "validation": val_res
    }

def evaluate_and_promote(job_info: Dict[str, Any], candidate_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Anti-Degradation Safety Guardrail:
    Compares candidate model against current production model.
    Only promotes candidate if candidate composite score is strictly superior or equal to production.
    """
    db = SessionLocal()
    job = None
    try:
        job_id = job_info["job_id"]
        job = db.query(RetrainingJob).filter(RetrainingJob.job_id == job_id).first()
        model_name = candidate_info["model_name"]
        new_version = candidate_info["new_version"]
        candidate_metrics = candidate_info["metrics"]
        cand_val = candidate_info["validation"]

        # Fetch current production metrics
        current_prod = registry_service.get_current_production_version(model_name, db=db)
        prod_metrics = current_prod["metrics"] if current_prod and current_prod.get("metrics") else {}

        # Composite score formula: 0.6 * PR-AUC + 0.4 * Recall
        cand_pr_auc = float(candidate_metrics.get("pr_auc", 0.0))
        cand_recall = float(candidate_metrics.get("recall", 0.0))
        cand_score = 0.6 * cand_pr_auc + 0.4 * cand_recall

        prod_pr_auc = float(prod_metrics.get("pr_auc", 0.0))
        prod_recall = float(prod_metrics.get("recall", 0.0))
        prod_score = 0.6 * prod_pr_auc + 0.4 * prod_recall

        # Guardrail Decision Logic
        if not cand_val["is_valid"]:
            decision = "REJECTED_FAILED_VALIDATION"
            final_status = "SKIPPED_INFERIOR"
            log_msg = f"Candidate v{new_version} failed validation criteria: {cand_val['reason']}"
            logger.warning(log_msg)

        elif current_prod and (cand_score < prod_score):
            # Candidate is strictly worse than existing production model!
            decision = "REJECTED_INFERIOR"
            final_status = "SKIPPED_INFERIOR"
            log_msg = (
                f"SAFETY GUARDRAIL TRIGGERED: Candidate v{new_version} composite score ({cand_score:.4f}) "
                f"is inferior to current Production v{current_prod['version']} score ({prod_score:.4f}). "
                f"Production model retained. Retraining skipped promotion."
            )
            logger.warning(log_msg)

        else:
            # Candidate is superior or equals production!
            decision = "PROMOTED"
            final_status = "COMPLETED"
            promote_res = registry_service.promote_stage(
                model_name=model_name,
                version=new_version,
                target_stage="Production",
                reason=f"Retrained model v{new_version} outperformed baseline ({cand_score:.4f} vs {prod_score:.4f})",
                deployed_by="automated_retraining_engine",
                db=db
            )
            log_msg = (
                f"Candidate v{new_version} passed validation and outperformed production model. "
                f"Successfully promoted v{new_version} to Production."
            )
            logger.info(log_msg)

        # Update RetrainingJob in database
        if job:
            job.status = final_status
            job.new_model_version = new_version
            job.candidate_metrics = candidate_metrics
            job.production_metrics = prod_metrics
            job.promotion_decision = decision
            job.logs = log_msg
            job.completed_at = utc_now()
            db.commit()
            db.refresh(job)

        return {
            "job_id": job_id,
            "status": final_status,
            "promotion_decision": decision,
            "candidate_version": new_version,
            "previous_production_version": job_info["previous_model_version"],
            "candidate_score": round(cand_score, 4),
            "production_score": round(prod_score, 4),
            "logs": log_msg,
            "completed_at": job.completed_at.isoformat() if job and job.completed_at else None
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error during candidate evaluation and promotion: {e}\n{traceback.format_exc()}")
        if job:
            job.status = "FAILED"
            job.logs = f"Error during promotion: {str(e)}"
            job.completed_at = utc_now()
            db.commit()
        raise e
    finally:
        db.close()

def execute_retraining_workflow(
    trigger_reason: str = "MANUAL",
    force: bool = False,
    quick_run: bool = False,
    optuna_trials: int = 5
) -> Dict[str, Any]:
    """
    Direct core orchestrator for automated retraining lifecycle.
    Can be called synchronously or wrapped in Prefect flow.
    """
    job_info = check_guardrails(trigger_reason, force)
    pipeline_res = train_candidates(job_info, quick_run, optuna_trials)
    candidate_info = register_and_validate(job_info, pipeline_res)
    decision_res = evaluate_and_promote(job_info, candidate_info)

    return {
        "job_id": job_info["job_id"],
        "trigger_reason": trigger_reason,
        "status": decision_res["status"],
        "promotion_decision": decision_res["promotion_decision"],
        "candidate_version": decision_res["candidate_version"],
        "previous_version": decision_res["previous_production_version"],
        "candidate_score": decision_res["candidate_score"],
        "production_score": decision_res["production_score"],
        "details": decision_res
    }

# Prefect Task Wrappers
@task(name="Check Retraining Guardrails (Duplicates & Cooldown)")
def check_retraining_guardrails_task(trigger_reason: str, force: bool = False) -> Dict[str, Any]:
    return check_guardrails(trigger_reason, force)

@task(name="Train Candidate Retraining Models")
def train_candidate_models_task(job_info: Dict[str, Any], quick_run: bool = False, optuna_trials: int = 5) -> Dict[str, Any]:
    return train_candidates(job_info, quick_run, optuna_trials)

@task(name="Register and Validate Candidate Model")
def register_and_validate_candidate_task(job_info: Dict[str, Any], pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
    return register_and_validate(job_info, pipeline_result)

@task(name="Evaluate Champion vs Challenger & Promote Safely")
def evaluate_and_promote_task(job_info: Dict[str, Any], candidate_info: Dict[str, Any]) -> Dict[str, Any]:
    return evaluate_and_promote(job_info, candidate_info)

@flow(name="Automated Retraining and Safe Promotion Flow", log_prints=True)
def automated_retraining_flow(
    trigger_reason: str = "MANUAL",
    force: bool = False,
    quick_run: bool = False,
    optuna_trials: int = 5
) -> Dict[str, Any]:
    """
    Prefect Flow coordinating automated retraining lifecycle with champion vs challenger guardrail.
    """
    logger.info(f"=== Starting Automated Retraining Flow (Trigger: {trigger_reason}) ===")
    return execute_retraining_workflow(trigger_reason, force, quick_run, optuna_trials)

if __name__ == "__main__":
    execute_retraining_workflow(trigger_reason="MANUAL", force=True, quick_run=True, optuna_trials=2)
