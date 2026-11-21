"""
End-to-End Machine Learning Training, Optimization, and Evaluation Pipeline.
Coordinates data validation, leakage-free preprocessing, Bayesian Optuna tuning,
model training (Logistic Regression vs XGBoost), comprehensive evaluation,
MLflow tracking, and champion model selection.
"""

import os
import time
import logging
import json
import joblib
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from backend.app.core.config import settings
from backend.app.database.session import SessionLocal
from backend.app.models.schemas_db import TrainingRun
from backend.training.data_loader import load_data, validate_schema, perform_eda
from backend.training.preprocessing import CreditCardPreprocessor, stratified_split
from backend.training.evaluate import evaluate_model, generate_comparison_report
from backend.training.tuning import tune_logistic_regression, tune_xgboost
from backend.training.mlflow_tracking import log_training_run

logger = logging.getLogger("modelops-hub.train")

def run_training_pipeline(
    data_path: Optional[str] = None,
    optuna_trials: int = 10,
    quick_run: bool = False
) -> Dict[str, Any]:
    """
    Executes the full automated ML training workflow:
    1. Loads and validates dataset.
    2. Runs EDA.
    3. Splits data into Stratified Train (70%), Val (15%), Test (15%).
    4. Fits CreditCardPreprocessor strictly on Train partition.
    5. Tunes Logistic Regression and XGBoost with Optuna.
    6. Trains finalized models and evaluates on held-out Test partition.
    7. Selects Champion model based on PR-AUC and Recall.
    8. Logs runs to MLflow and persists application database records.
    """
    pipeline_start_time = time.time()
    logger.info("=== STARTING MODELOPS HUB TRAINING PIPELINE ===")

    # 1. Ingestion & Validation
    df = load_data(data_path)
    schema_info = validate_schema(df)
    eda_summary = perform_eda(df, save_report_dir=os.path.join(settings.ARTIFACTS_DIR, "eda"))

    # For fast testing or quick runs, optionally subsample if requested
    if quick_run:
        logger.info("Quick run requested: sampling 50,000 transactions (preserving all frauds)...")
        frauds = df[df["Class"] == 1]
        legit = df[df["Class"] == 0].sample(n=50000 - len(frauds), random_state=42)
        df = pd.concat([frauds, legit]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 2. Stratified Partitioning (70% Train, 15% Val, 15% Test)
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = stratified_split(df)

    # 3. Leakage-Free Preprocessing Pipeline
    preprocessor = CreditCardPreprocessor()
    preprocessor.fit(X_train_raw)

    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    feature_names = [f"V{i}" for i in range(1, 29)] + ["scaled_Time", "scaled_Amount"]

    models_results = {}
    mlflow_runs = {}

    # ==========================================
    # 4. Logistic Regression Training & Tuning
    # ==========================================
    lr_t0 = time.time()
    lr_tune_results = tune_logistic_regression(
        X_train, y_train.values,
        n_trials=optuna_trials,
        n_splits=3 if quick_run else 5
    )
    lr_best_params = lr_tune_results["best_params"]

    # Train final tuned Logistic Regression model on full training set
    lr_model = LogisticRegression(
        C=lr_best_params.get("C", 1.0),
        solver=lr_best_params.get("solver", "lbfgs"),
        class_weight=lr_best_params.get("class_weight", "balanced"),
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train, y_train.values)
    lr_duration = time.time() - lr_t0

    # Evaluate on held-out test partition
    lr_eval = evaluate_model(lr_model, X_test, y_test.values, feature_names=feature_names)
    lr_eval["training_duration_seconds"] = lr_duration
    models_results["LogisticRegression"] = lr_eval

    # Log to MLflow
    lr_mlflow = log_training_run(
        algorithm="LogisticRegression",
        model=lr_model,
        preprocessor=preprocessor,
        hyperparameters=lr_best_params,
        eval_results=lr_eval,
        duration_seconds=lr_duration,
        run_name="LogisticRegression-Tuned"
    )
    mlflow_runs["LogisticRegression"] = lr_mlflow

    # ==========================================
    # 5. XGBoost Classifier Training & Tuning
    # ==========================================
    xgb_t0 = time.time()
    xgb_tune_results = tune_xgboost(
        X_train, y_train.values,
        n_trials=optuna_trials,
        n_splits=3 if quick_run else 5
    )
    xgb_best_params = xgb_tune_results["best_params"]

    # Train final tuned XGBoost model on full training set
    xgb_model = XGBClassifier(**xgb_best_params)
    xgb_model.fit(X_train, y_train.values)
    xgb_duration = time.time() - xgb_t0

    # Evaluate on held-out test partition
    xgb_eval = evaluate_model(xgb_model, X_test, y_test.values, feature_names=feature_names)
    xgb_eval["training_duration_seconds"] = xgb_duration
    models_results["XGBClassifier"] = xgb_eval

    # Log to MLflow
    xgb_mlflow = log_training_run(
        algorithm="XGBClassifier",
        model=xgb_model,
        preprocessor=preprocessor,
        hyperparameters=xgb_best_params,
        eval_results=xgb_eval,
        duration_seconds=xgb_duration,
        run_name="XGBClassifier-Tuned"
    )
    mlflow_runs["XGBClassifier"] = xgb_mlflow

    # ==========================================
    # 6. Model Comparison & Champion Selection
    # ==========================================
    comparison_report = generate_comparison_report(models_results)
    champion_name = comparison_report["champion_model"]

    champion_model = xgb_model if champion_name == "XGBClassifier" else lr_model
    champion_eval = models_results[champion_name]
    champion_mlflow = mlflow_runs[champion_name]

    # Save champion model & preprocessor to dedicated deployment directory
    champion_dir = os.path.join(settings.ARTIFACTS_DIR, "champion")
    os.makedirs(champion_dir, exist_ok=True)
    joblib.dump(champion_model, os.path.join(champion_dir, "model.joblib"))
    joblib.dump(preprocessor, os.path.join(champion_dir, "preprocessor.joblib"))

    with open(os.path.join(champion_dir, "champion_info.json"), "w") as f:
        json.dump({
            "champion_algorithm": champion_name,
            "mlflow_run_id": champion_mlflow["run_id"],
            "metrics": champion_eval["metrics"],
            "selected_at": time.time()
        }, f, indent=2)

    total_pipeline_duration = time.time() - pipeline_start_time
    logger.info(
        f"Pipeline Completed in {total_pipeline_duration:.2f}s! "
        f"Champion: {champion_name} (PR-AUC: {champion_eval['metrics']['pr_auc']:.4f}, "
        f"Recall: {champion_eval['metrics']['recall']:.4f})"
    )

    # ==========================================
    # 7. Record in Application Database (PostgreSQL / SQLite)
    # ==========================================
    db = SessionLocal()
    try:
        for algo, eval_res in models_results.items():
            mlflow_info = mlflow_runs[algo]
            params = xgb_best_params if algo == "XGBClassifier" else lr_best_params
            record = TrainingRun(
                run_id=mlflow_info["run_id"],
                experiment_name=settings.MLFLOW_EXPERIMENT_NAME,
                algorithm=algo,
                status="COMPLETED",
                parameters=params,
                metrics=eval_res["metrics"],
                duration_seconds=eval_res["training_duration_seconds"],
                artifact_uri=mlflow_info["artifact_uri"]
            )
            db.add(record)
        db.commit()
        logger.info("Successfully recorded training runs in application database.")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to record training runs in database: {e}")
    finally:
        db.close()

    return {
        "status": "SUCCESS",
        "pipeline_duration_seconds": round(total_pipeline_duration, 2),
        "schema_validation": schema_info,
        "eda_summary": eda_summary,
        "models_evaluated": models_results,
        "mlflow_runs": mlflow_runs,
        "comparison_report": comparison_report,
        "champion_model": champion_name,
        "champion_metrics": champion_eval["metrics"],
        "champion_artifacts_dir": champion_dir
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training_pipeline()
