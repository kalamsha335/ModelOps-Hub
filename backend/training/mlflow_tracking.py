"""
MLflow Experiment Tracking and Model Artifact Persistence Module.
Logs training runs, tuned hyperparameters, evaluation metrics, confusion matrices,
feature importances, and serialized pipeline binaries.
"""

import os
import json
import joblib
import logging
from typing import Dict, Any, Optional
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from backend.app.core.config import settings

logger = logging.getLogger("modelops-hub.mlflow_tracking")

def setup_mlflow():
    """Configures MLflow tracking URI and active experiment."""
    tracking_uri = settings.MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    logger.info(
        f"MLflow initialized: URI='{tracking_uri}', "
        f"Experiment='{settings.MLFLOW_EXPERIMENT_NAME}' (ID: {experiment.experiment_id})"
    )
    return experiment

def log_training_run(
    algorithm: str,
    model: Any,
    preprocessor: Any,
    hyperparameters: Dict[str, Any],
    eval_results: Dict[str, Any],
    duration_seconds: float,
    run_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Logs an entire training and evaluation execution to MLflow:
    - Parameters (hyperparameters, dataset shapes)
    - Metrics (PR-AUC, Recall, Precision, F1, ROC-AUC)
    - Model & Preprocessor artifacts
    - Confusion Matrix & Feature Importances
    """
    setup_mlflow()

    run_name = run_name or f"{algorithm}-training"
    metrics = eval_results["metrics"]

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"Logging MLflow run '{run_name}' with ID: {run_id}")

        # 1. Log tags
        mlflow.set_tag("algorithm", algorithm)
        mlflow.set_tag("pipeline", "modelops-hub-training")
        mlflow.set_tag("target", "Class")

        # 2. Log hyperparameters
        clean_params = {
            k: (str(v) if not isinstance(v, (int, float, bool, str)) else v)
            for k, v in hyperparameters.items()
        }
        mlflow.log_params(clean_params)
        mlflow.log_param("duration_seconds", round(duration_seconds, 2))

        # 3. Log evaluation metrics
        for metric_name, val in metrics.items():
            mlflow.log_metric(metric_name, float(val))

        # 4. Save and log artifacts locally
        run_artifacts_dir = os.path.join(settings.ARTIFACTS_DIR, run_id)
        os.makedirs(run_artifacts_dir, exist_ok=True)

        # Preprocessor
        preprocessor_path = os.path.join(run_artifacts_dir, "preprocessor.joblib")
        joblib.dump(preprocessor, preprocessor_path)
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

        # Model binary
        model_path = os.path.join(run_artifacts_dir, "model.joblib")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        # Evaluation metrics JSON
        eval_path = os.path.join(run_artifacts_dir, "evaluation_report.json")
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        mlflow.log_artifact(eval_path, artifact_path="reports")

        # Confusion matrix JSON
        cm_path = os.path.join(run_artifacts_dir, "confusion_matrix.json")
        with open(cm_path, "w") as f:
            json.dump(eval_results["confusion_matrix"], f, indent=2)
        mlflow.log_artifact(cm_path, artifact_path="reports")

        # Feature importances JSON
        feat_path = os.path.join(run_artifacts_dir, "feature_importance.json")
        with open(feat_path, "w") as f:
            json.dump(eval_results.get("feature_importances", {}), f, indent=2)
        mlflow.log_artifact(feat_path, artifact_path="reports")

        artifact_uri = run.info.artifact_uri
        logger.info(f"MLflow Run {run_id} completed. Artifact URI: {artifact_uri}")

        return {
            "run_id": run_id,
            "experiment_id": run.info.experiment_id,
            "artifact_uri": artifact_uri,
            "run_name": run_name,
            "local_artifact_dir": run_artifacts_dir
        }

