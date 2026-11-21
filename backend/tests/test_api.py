"""
Comprehensive Unit & Integration Test Suite for FastAPI Serving Layer (Phase 6).
Directly tests endpoint handlers, request validation, prediction serving,
feedback ingestion, registry governance, and monitoring endpoints.
"""

import pytest
from backend.app.main import health_check
from backend.app.api.v1.endpoints.predict import (
    predict_transaction, predict_batch_transactions,
    ingest_ground_truth_feedback, get_prediction_logs
)
from backend.app.api.v1.endpoints.model_info import get_model_info
from backend.app.api.v1.endpoints.registry import list_registered_models, list_model_versions
from backend.app.api.v1.endpoints.monitoring import (
    get_monitoring_overview, get_drift_details, list_drift_alerts
)
from backend.app.api.v1.endpoints.training import get_training_history
from backend.app.api.v1.endpoints.retraining import list_retraining_history
from backend.app.schemas.schemas_api import (
    TransactionFeatures, BatchTransactionInput, GroundTruthFeedbackRequest
)
from backend.app.database.session import SessionLocal, engine, Base

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

def test_health_endpoint(db_session):
    res = health_check(db=db_session)
    assert res["status"] == "healthy"
    assert "ModelOps Hub" in res["service"]
    assert res["database"] == "connected"

def test_model_info_endpoint():
    res = get_model_info()
    assert res["model_name"] == "fraud-detection-model"
    assert res["features_expected"] == 30
    assert "status" in res

def test_predict_single_and_feedback(db_session):
    # Sample transaction payload
    sample_feat = {f"V{i}": -1.0 for i in range(1, 29)}
    sample_feat["Time"] = 400.0
    sample_feat["Amount"] = 149.62
    payload = TransactionFeatures(**sample_feat)

    pred_res = predict_transaction(payload, db=db_session)
    assert pred_res.prediction in [0, 1]
    assert pred_res.prediction_label in ["Legitimate", "Fraudulent"]
    assert 0.0 <= pred_res.fraud_probability <= 1.0
    assert pred_res.model_name == "fraud-detection-model"
    assert pred_res.latency_ms >= 0.0
    assert pred_res.prediction_id is not None

    # Test ground-truth feedback ingestion
    fb_req = GroundTruthFeedbackRequest(
        prediction_log_id=pred_res.prediction_id,
        ground_truth=1
    )
    fb_res = ingest_ground_truth_feedback(fb_req, db=db_session)
    assert fb_res["status"] == "SUCCESS"
    assert fb_res["recorded_ground_truth"] == 1

def test_predict_batch_and_logs(db_session):
    sample_feat = {f"V{i}": 0.0 for i in range(1, 29)}
    sample_feat["Time"] = 50.0
    sample_feat["Amount"] = 20.0
    item = TransactionFeatures(**sample_feat)

    batch_payload = BatchTransactionInput(transactions=[item, item, item])
    batch_res = predict_batch_transactions(batch_payload, db=db_session)

    assert batch_res.total_processed == 3
    assert len(batch_res.predictions) == 3
    assert batch_res.total_latency_ms >= 0.0

    # Test retrieving logs
    logs_res = get_prediction_logs(limit=10, offset=0, db=db_session)
    assert logs_res["total_records"] >= 4
    assert len(logs_res["records"]) <= 10

def test_registry_endpoints(db_session):
    # Models list
    models = list_registered_models(db=db_session)
    assert isinstance(models, list)
    assert any(m["model_name"] == "fraud-detection-model" for m in models)

    # Versions list
    versions = list_model_versions(model_name="fraud-detection-model", db=db_session)
    assert len(versions) >= 1
    assert any(v["is_current_production"] is True for v in versions)

def test_monitoring_endpoints(db_session):
    overview = get_monitoring_overview(db=db_session)
    assert "overall_health" in overview
    assert "active_alerts_total" in overview

    drift = get_drift_details(db=db_session)
    assert "status" in drift

    alerts = list_drift_alerts(db=db_session)
    assert isinstance(alerts, list)

def test_training_and_retraining_history(db_session):
    t_hist = get_training_history(limit=10, db=db_session)
    assert isinstance(t_hist, list)

    r_hist = list_retraining_history(limit=10, db=db_session)
    assert isinstance(r_hist, list)
