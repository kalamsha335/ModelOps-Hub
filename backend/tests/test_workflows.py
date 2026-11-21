"""
Test suite for Workflow Orchestration, Retraining Triggers, and Safety Guardrails.
"""

import pytest
import uuid
from datetime import datetime, timezone
from backend.app.database.session import SessionLocal, engine, Base
from backend.app.models.schemas_db import RetrainingJob, ModelVersion
from backend.workflows.retraining_flow import (
    check_guardrails,
    evaluate_and_promote,
    execute_retraining_workflow
)

@pytest.fixture(scope="module")
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield

@pytest.fixture(scope="module")
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

def test_duplicate_retraining_guardrail(db_session):
    """Verifies that duplicate concurrent retraining jobs are prevented."""
    # Insert an artificial running job
    active_job_id = f"test_active_{uuid.uuid4().hex[:8]}"
    job = RetrainingJob(
        job_id=active_job_id,
        trigger_reason="CRITICAL_FEATURE_DRIFT",
        status="RUNNING",
        started_at=datetime.now(timezone.utc)
    )
    db_session.add(job)
    db_session.commit()

    try:
        # Attempt to launch retraining without force=True -> Must raise ValueError
        with pytest.raises(ValueError, match="Retraining rejected: Job .* is currently running"):
            check_guardrails(
                trigger_reason="SCHEDULED",
                force=False
            )
    finally:
        # Cleanup
        db_session.delete(job)
        db_session.commit()

def test_champion_vs_challenger_inferior_rejection(db_session):
    """
    Verifies that an inferior retrained candidate model CANNOT replace
    a superior active production model.
    """
    job_id = f"test_inferior_{uuid.uuid4().hex[:8]}"
    job = RetrainingJob(
        job_id=job_id,
        trigger_reason="MANUAL",
        status="RUNNING",
        started_at=datetime.now(timezone.utc)
    )
    db_session.add(job)
    db_session.commit()

    job_info = {
        "job_id": job_id,
        "previous_model_version": "4",
        "trigger_reason": "MANUAL"
    }

    # Simulate an inferior candidate (PR-AUC 0.70, Recall 0.65)
    # compared to active Production version 4 (PR-AUC ~0.8245, Recall ~0.7973)
    candidate_info = {
        "model_name": "fraud-detection-model",
        "new_version": "999",
        "algorithm": "LogisticRegression",
        "metrics": {"pr_auc": 0.70, "recall": 0.65},
        "validation": {"is_valid": True, "reason": "Passed test threshold"}
    }

    decision_res = evaluate_and_promote(job_info, candidate_info)

    # Must be rejected because candidate composite score is lower than production
    assert decision_res["status"] == "SKIPPED_INFERIOR"
    assert decision_res["promotion_decision"] == "REJECTED_INFERIOR"
    assert "SAFETY GUARDRAIL TRIGGERED" in decision_res["logs"]

    # Verify DB record
    db_job = db_session.query(RetrainingJob).filter(RetrainingJob.job_id == job_id).first()
    assert db_job.status == "SKIPPED_INFERIOR"
    assert db_job.promotion_decision == "REJECTED_INFERIOR"

def test_retraining_workflow_quick_execution():
    """
    Tests end-to-end execution of execute_retraining_workflow in quick mode.
    """
    res = execute_retraining_workflow(
        trigger_reason="MANUAL",
        force=True,
        quick_run=True,
        optuna_trials=1
    )

    assert "job_id" in res
    assert res["status"] in ["COMPLETED", "SKIPPED_INFERIOR"]
    assert res["candidate_score"] > 0
