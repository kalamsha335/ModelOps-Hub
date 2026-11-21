"""
ModelOps Hub - Prefect Training Workflow
Orchestrates data ingestion, validation, leakage-free preprocessing,
Optuna hyperparameter tuning, model evaluation, and experiment tracking.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
from prefect import flow, task

from backend.app.core.config import settings
from backend.training.data_loader import load_data, validate_schema, perform_eda
from backend.training.preprocessing import CreditCardPreprocessor, stratified_split
from backend.training.train import run_training_pipeline
from backend.app.registry.model_registry import registry_service

logger = logging.getLogger("modelops.workflows.training")

@task(name="Ingest and Validate Dataset", retries=2, retry_delay_seconds=5)
def ingest_and_validate_task(data_path: Optional[str] = None) -> Dict[str, Any]:
    path = data_path or settings.RAW_DATA_PATH
    logger.info(f"Task: Ingesting dataset from {path}")
    df = load_data(path)
    schema_info = validate_schema(df)
    return {"data_path": path, "rows": len(df), "is_valid": schema_info["is_valid"]}

@task(name="Execute Automated Training Pipeline")
def execute_training_pipeline_task(
    data_path: Optional[str] = None,
    optuna_trials: int = 5,
    quick_run: bool = False
) -> Dict[str, Any]:
    logger.info(f"Task: Executing training pipeline (trials={optuna_trials}, quick={quick_run})")
    return run_training_pipeline(
        data_path=data_path or settings.RAW_DATA_PATH,
        optuna_trials=optuna_trials,
        quick_run=quick_run
    )

@task(name="Register Candidate Model")
def register_candidate_task(
    pipeline_result: Dict[str, Any],
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    model_name = model_name or settings.MLFLOW_MODEL_NAME
    champion_algo = pipeline_result["champion_model"]
    run_id = pipeline_result["mlflow_runs"][champion_algo]["run_id"]
    logger.info(f"Task: Registering champion {champion_algo} (run: {run_id}) to registry")

    reg_info = registry_service.register_model_version(
        run_id=run_id,
        model_name=model_name,
        description=f"Candidate {champion_algo} model trained by Prefect training flow"
    )
    return reg_info

@flow(name="Automated Model Training Flow", log_prints=True)
def model_training_flow(
    data_path: Optional[str] = None,
    optuna_trials: int = 5,
    quick_run: bool = False,
    auto_register: bool = True
) -> Dict[str, Any]:
    """
    Prefect Flow orchestrating end-to-end model training, evaluation, comparison, and registration.
    """
    logger.info("Starting Prefect Model Training Flow...")
    val_res = ingest_and_validate_task(data_path)
    train_res = execute_training_pipeline_task(data_path, optuna_trials, quick_run)

    reg_res = None
    if auto_register and train_res["status"] == "SUCCESS":
        reg_res = register_candidate_task(train_res)

    return {
        "status": "COMPLETED",
        "validation": val_res,
        "training": train_res,
        "registration": reg_res
    }

if __name__ == "__main__":
    model_training_flow(quick_run=True, optuna_trials=2)

