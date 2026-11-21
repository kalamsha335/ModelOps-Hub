"""
ModelOps Hub - API Request and Response Pydantic Schemas
Defines structured validation contracts for prediction serving, registry governance,
monitoring metrics, and retraining workflows.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class TransactionFeatures(BaseModel):
    """Features of a single credit card transaction (Time, V1-V28, Amount)."""
    Time: float = Field(..., description="Seconds elapsed between this transaction and the first transaction in dataset", example=0.0)
    V1: float = Field(..., description="PCA Feature 1", example=-1.359807)
    V2: float = Field(..., description="PCA Feature 2", example=-0.072781)
    V3: float = Field(..., description="PCA Feature 3", example=2.536347)
    V4: float = Field(..., description="PCA Feature 4", example=1.378155)
    V5: float = Field(..., description="PCA Feature 5", example=-0.338321)
    V6: float = Field(..., description="PCA Feature 6", example=0.462388)
    V7: float = Field(..., description="PCA Feature 7", example=0.239599)
    V8: float = Field(..., description="PCA Feature 8", example=0.098698)
    V9: float = Field(..., description="PCA Feature 9", example=0.363787)
    V10: float = Field(..., description="PCA Feature 10", example=0.090794)
    V11: float = Field(..., description="PCA Feature 11", example=-0.551600)
    V12: float = Field(..., description="PCA Feature 12", example=-0.617801)
    V13: float = Field(..., description="PCA Feature 13", example=-0.991390)
    V14: float = Field(..., description="PCA Feature 14", example=-0.311169)
    V15: float = Field(..., description="PCA Feature 15", example=1.468177)
    V16: float = Field(..., description="PCA Feature 16", example=-0.470401)
    V17: float = Field(..., description="PCA Feature 17", example=0.207971)
    V18: float = Field(..., description="PCA Feature 18", example=0.025791)
    V19: float = Field(..., description="PCA Feature 19", example=0.403993)
    V20: float = Field(..., description="PCA Feature 20", example=0.251412)
    V21: float = Field(..., description="PCA Feature 21", example=-0.018307)
    V22: float = Field(..., description="PCA Feature 22", example=0.277838)
    V23: float = Field(..., description="PCA Feature 23", example=-0.110474)
    V24: float = Field(..., description="PCA Feature 24", example=0.066928)
    V25: float = Field(..., description="PCA Feature 25", example=0.128539)
    V26: float = Field(..., description="PCA Feature 26", example=-0.189115)
    V27: float = Field(..., description="PCA Feature 27", example=0.133558)
    V28: float = Field(..., description="PCA Feature 28", example=-0.021053)
    Amount: float = Field(..., ge=0.0, description="Transaction monetary amount", example=149.62)

class BatchTransactionInput(BaseModel):
    """Batch prediction request containing multiple transactions."""
    transactions: List[TransactionFeatures] = Field(..., min_items=1, max_items=1000)

class PredictionResponse(BaseModel):
    """Standardized single prediction response contract."""
    prediction: int = Field(..., description="0 for Legitimate, 1 for Fraudulent")
    prediction_label: str = Field(..., description="'Legitimate' or 'Fraudulent'")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability of fraud")
    model_name: str = Field(..., description="Name of serving model")
    model_version: str = Field(..., description="Version of model deployed in production")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    prediction_id: Optional[int] = Field(None, description="Database prediction log primary key ID")

class BatchPredictionResponse(BaseModel):
    """Response contract for batch prediction requests."""
    predictions: List[PredictionResponse]
    total_processed: int
    total_latency_ms: float
    model_name: str
    model_version: str

class GroundTruthFeedbackRequest(BaseModel):
    """Feedback payload when delayed ground-truth chargeback labels arrive."""
    prediction_log_id: int = Field(..., description="ID from prediction_logs table")
    ground_truth: int = Field(..., ge=0, le=1, description="0 = Legitimate, 1 = Confirmed Fraud")

# Registry Governance Schemas
class PromoteRequest(BaseModel):
    """Request to transition a model version to a target lifecycle stage."""
    model_name: str = Field(default="fraud-detection-model")
    version: str = Field(..., example="2")
    target_stage: str = Field(..., example="Production")  # Staging, Production, Archived
    reason: Optional[str] = Field(None, example="Validated champion candidate with PR-AUC 0.82")
    enforce_validation: bool = Field(default=True, description="Enforce strict PR-AUC and Recall gates")

class RollbackRequest(BaseModel):
    """Request to atomically revert active production model to previous validated version."""
    model_name: str = Field(default="fraud-detection-model")
    target_version: Optional[str] = Field(None, description="Optional target version. If omitted, uses previous production version.")
    reason: Optional[str] = Field(None, example="Emergency rollback: production latency spike")

# Retraining Schemas
class RetrainingTriggerRequest(BaseModel):
    """Manual or automated retraining trigger request."""
    trigger_reason: str = Field(default="MANUAL", example="CRITICAL_FEATURE_DRIFT")
    force: bool = Field(default=False, description="Bypass cooldown or active job check")
    quick_run: bool = Field(default=False, description="Fast subsampled training for testing")
    optuna_trials: int = Field(default=5, description="Number of Optuna tuning trials")

# Monitoring Request Schemas
class ManualDriftCheckRequest(BaseModel):
    """Request to run ad-hoc drift evaluation on a batch of transactions."""
    sample_size: int = Field(default=2000, ge=100, le=50000)
    generate_evidently_html: bool = Field(default=True)

