"""
Test suite for Data Drift (PSI, KS Test, Evidently) and Concept Drift Monitoring.
"""

import pytest
import numpy as np
import pandas as pd
from backend.monitoring.psi import calculate_feature_psi, calculate_batch_psi
from backend.monitoring.ks_test import calculate_feature_ks, calculate_batch_ks
from backend.monitoring.data_drift import DataDriftDetector
from backend.monitoring.concept_drift import ConceptDriftDetector
from backend.app.database.session import SessionLocal, engine, Base
from backend.app.models.schemas_db import DriftReport, DriftAlert, PerformanceMetric

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

def test_psi_identical_vs_drifted():
    np.random.seed(42)
    # Reference distribution: standard normal
    reference = np.random.normal(loc=0.0, scale=1.0, size=2000)
    
    # Current identical distribution
    current_healthy = np.random.normal(loc=0.0, scale=1.0, size=2000)
    psi_healthy, breakdown_healthy = calculate_feature_psi(reference, current_healthy)
    assert psi_healthy < 0.10, f"Expected low PSI for identical distributions, got {psi_healthy}"
    assert breakdown_healthy["status"] == "HEALTHY"

    # Current drifted distribution (shifted mean and scaled variance)
    current_drifted = np.random.normal(loc=2.5, scale=2.0, size=2000)
    psi_drifted, breakdown_drifted = calculate_feature_psi(reference, current_drifted)
    assert psi_drifted >= 0.30, f"Expected critical PSI for drifted distribution, got {psi_drifted}"
    assert breakdown_drifted["status"] == "CRITICAL"

def test_ks_test_identical_vs_drifted():
    np.random.seed(42)
    reference = np.random.normal(0, 1, 1000)
    current_healthy = np.random.normal(0, 1, 1000)
    res_healthy = calculate_feature_ks(reference, current_healthy)
    assert res_healthy["is_drifted"] is False
    assert res_healthy["p_value"] > 0.05

    current_drifted = np.random.normal(1.5, 1, 1000)
    res_drifted = calculate_feature_ks(reference, current_drifted)
    assert res_drifted["is_drifted"] is True
    assert res_drifted["p_value"] < 0.05

def test_data_drift_detector_end_to_end(db_session):
    detector = DataDriftDetector()
    ref_df = detector.get_reference_data()
    assert len(ref_df) > 0

    # 1. Healthy production batch: random sample from reference
    sample_size = min(len(ref_df), 1000)
    healthy_batch = ref_df.sample(sample_size, random_state=123).copy()
    healthy_res = detector.run_drift_analysis(
        current_df=healthy_batch,
        reference_df=ref_df,
        generate_evidently_html=False,
        db=db_session
    )
    assert healthy_res["status"] in ["Healthy", "Warning"]
    assert healthy_res["drift_share"] < 0.25

    # 2. Injected Drift batch: deliberately shift Amount and key V features
    drifted_batch = healthy_batch.copy()
    drifted_batch["Amount"] = drifted_batch["Amount"] * 10.0 + 500.0
    for col in ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10"]:
        drifted_batch[col] = drifted_batch[col] + 3.5

    drifted_res = detector.run_drift_analysis(
        current_df=drifted_batch,
        reference_df=ref_df,
        generate_evidently_html=False,
        db=db_session
    )
    assert drifted_res["status"] == "Critical"
    assert drifted_res["drifted_features_count"] >= 10
    assert "Amount" in drifted_res["drifted_features"]
    assert drifted_res["alerts_created"] >= 10

    # Verify database persistence
    rep = db_session.query(DriftReport).filter(DriftReport.report_id == drifted_res["report_id"]).first()
    assert rep is not None
    assert rep.status == "Critical"

    alerts = db_session.query(DriftAlert).filter(DriftAlert.drift_report_id == rep.id).all()
    assert len(alerts) >= 10

def test_concept_drift_detector(db_session):
    detector = ConceptDriftDetector(performance_drop_threshold=0.10)

    # 1. Test when labels are awaiting
    probs = np.random.uniform(0.0001, 0.01, size=1000)
    pred_res = detector.evaluate_prediction_drift_without_labels(probs)
    assert pred_res["status"] == "AWAITING_GROUND_TRUTH_LABELS"
    assert "prediction_monitoring_status" in pred_res

    # 2. Test ground-truth evaluation: Healthy performance
    # 50 positives, 950 negatives
    y_true = np.array([1]*50 + [0]*950)
    # Model predicts accurately with high confidence
    y_proba_healthy = np.array([0.95]*45 + [0.1]*5 + [0.02]*940 + [0.8]*10)
    healthy_perf = detector.evaluate_ground_truth_performance(
        y_true=y_true,
        y_pred_proba=y_proba_healthy,
        model_version="4",
        baseline_metrics={"pr_auc": 0.82, "recall": 0.80},
        db=db_session
    )
    assert healthy_perf["status"] == "HEALTHY"
    assert healthy_perf["has_concept_drift"] is False

    # 3. Test ground-truth evaluation: Severe concept drift degradation
    # Model predictions miss almost all fraud cases (severe recall drop)
    y_proba_degraded = np.array([0.1]*45 + [0.9]*5 + [0.02]*940 + [0.8]*10)
    degraded_perf = detector.evaluate_ground_truth_performance(
        y_true=y_true,
        y_pred_proba=y_proba_degraded,
        model_version="4",
        baseline_metrics={"pr_auc": 0.82, "recall": 0.80},
        db=db_session
    )
    assert degraded_perf["status"] == "CRITICAL_CONCEPT_DRIFT"
    assert degraded_perf["has_concept_drift"] is True
    assert degraded_perf["degradation"]["recall_drop"] > 0.10

    # Verify DB persistence
    hist = detector.get_performance_history(model_version="4", db=db_session)
    assert len(hist) >= 2
