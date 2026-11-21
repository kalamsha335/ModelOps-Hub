from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey, func, Index
)
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid
from backend.app.database.session import Base

def utc_now():
    return datetime.now(timezone.utc)

class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_id = Column(String(64), unique=True, index=True, nullable=False)
    experiment_name = Column(String(128), index=True, nullable=False)
    algorithm = Column(String(64), nullable=False)
    status = Column(String(32), default="STARTED", nullable=False)
    parameters = Column(JSON, default=dict)
    metrics = Column(JSON, default=dict)
    duration_seconds = Column(Float, default=0.0)
    artifact_uri = Column(String(256), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_name = Column(String(128), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    problem_type = Column(String(64), default="binary_classification")
    target_column = Column(String(64), default="Class")
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    versions = relationship("ModelVersion", back_populates="metadata_rel", cascade="all, delete-orphan")

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_name = Column(String(128), ForeignKey("model_metadata.model_name"), index=True, nullable=False)
    version = Column(String(32), index=True, nullable=False)
    mlflow_run_id = Column(String(64), nullable=True)
    algorithm = Column(String(64), nullable=False)
    stage = Column(String(32), default="Development", index=True, nullable=False)  # Development, Staging, Production, Archived
    is_current_production = Column(Boolean, default=False, index=True)
    validation_status = Column(String(32), default="PENDING")  # PENDING, PASSED, FAILED
    metrics = Column(JSON, default=dict)
    artifacts_path = Column(String(256), nullable=True)
    promoted_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    metadata_rel = relationship("ModelMetadata", back_populates="versions")

class ModelDeployment(Base):
    __tablename__ = "model_deployments"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_name = Column(String(128), index=True, nullable=False)
    model_version = Column(String(32), nullable=False)
    stage = Column(String(32), nullable=False)
    deployed_by = Column(String(64), default="system")
    deployed_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    rollback_version = Column(String(32), nullable=True)
    reason = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_name = Column(String(128), index=True, nullable=False)
    model_version = Column(String(32), index=True, nullable=False)
    request_features = Column(JSON, nullable=False)
    prediction = Column(Integer, nullable=False)  # 0 or 1
    prediction_label = Column(String(32), nullable=False)  # Legitimate or Fraudulent
    fraud_probability = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False)
    ground_truth = Column(Integer, nullable=True)  # Populated when label arrives
    logged_at = Column(DateTime(timezone=True), default=utc_now, index=True, nullable=False)

    __table_args__ = (
        Index("idx_pred_model_time", "model_name", "logged_at"),
    )

class DriftReport(Base):
    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    report_id = Column(String(64), unique=True, index=True, nullable=False)
    status = Column(String(32), default="Healthy", index=True, nullable=False)  # Healthy, Warning, Critical
    drift_score = Column(Float, default=0.0)
    drifted_features_count = Column(Integer, default=0)
    total_features_count = Column(Integer, default=0)
    feature_metrics = Column(JSON, default=dict)
    evidently_report_html_path = Column(String(256), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, index=True, nullable=False)

    alerts = relationship("DriftAlert", back_populates="report", cascade="all, delete-orphan")

class DriftAlert(Base):
    __tablename__ = "drift_alerts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    drift_report_id = Column(Integer, ForeignKey("drift_reports.id"), nullable=True)
    severity = Column(String(32), default="WARNING", index=True, nullable=False)  # WARNING, CRITICAL
    affected_feature = Column(String(64), index=True, nullable=False)
    drift_metric = Column(String(32), nullable=False)  # PSI, KS_TEST, SHARE
    metric_value = Column(Float, nullable=False)
    threshold_value = Column(Float, nullable=False)
    recommended_action = Column(Text, nullable=True)
    status = Column(String(32), default="ACTIVE", index=True, nullable=False)  # ACTIVE, RESOLVED, ACKNOWLEDGED
    detected_at = Column(DateTime(timezone=True), default=utc_now, index=True, nullable=False)

    report = relationship("DriftReport", back_populates="alerts")

class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_version = Column(String(32), index=True, nullable=False)
    evaluation_window_start = Column(DateTime(timezone=True), nullable=True)
    evaluation_window_end = Column(DateTime(timezone=True), nullable=True)
    sample_count = Column(Integer, default=0)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)
    pr_auc = Column(Float, nullable=False)
    true_positives = Column(Integer, default=0)
    false_positives = Column(Integer, default=0)
    true_negatives = Column(Integer, default=0)
    false_negatives = Column(Integer, default=0)
    recorded_at = Column(DateTime(timezone=True), default=utc_now, index=True, nullable=False)

class RetrainingJob(Base):
    __tablename__ = "retraining_jobs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    job_id = Column(String(64), unique=True, index=True, nullable=False)
    trigger_reason = Column(String(64), nullable=False)  # CRITICAL_DRIFT, PERFORMANCE_DEGRADATION, SCHEDULED, MANUAL
    status = Column(String(32), default="PENDING", index=True, nullable=False)  # PENDING, RUNNING, COMPLETED, FAILED, SKIPPED_INFERIOR
    previous_model_version = Column(String(32), nullable=True)
    new_model_version = Column(String(32), nullable=True)
    candidate_metrics = Column(JSON, default=dict)
    production_metrics = Column(JSON, default=dict)
    promotion_decision = Column(String(64), default="PENDING")  # PROMOTED, REJECTED_INFERIOR, MANUAL_APPROVAL_REQUIRED
    logs = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), default=utc_now, index=True, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, default=0.0)
