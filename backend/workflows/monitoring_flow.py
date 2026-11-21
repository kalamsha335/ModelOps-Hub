"""
ModelOps Hub - Prefect Drift Monitoring Workflow
Schedules and coordinates recurring data drift detection (PSI, KS test, Evidently)
and automatically triggers the retraining workflow when critical drift occurs.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
from prefect import flow, task

from backend.app.core.config import settings
from backend.monitoring.data_drift import drift_detector

logger = logging.getLogger("modelops.workflows.monitoring")

@task(name="Fetch Current Production Batch")
def fetch_production_batch_task(
    sample_size: int = 2000,
    current_data_path: Optional[str] = None
) -> pd.DataFrame:
    """Fetches the latest production batch or samples from raw data as live incoming stream."""
    if current_data_path:
        df = pd.read_parquet(current_data_path) if current_data_path.endswith(".parquet") else pd.read_csv(current_data_path)
    else:
        # Default: sample from raw data to simulate live incoming production transactions
        df = pd.read_csv(settings.RAW_DATA_PATH)
        df = df.sample(min(len(df), sample_size), random_state=None)

    logger.info(f"Task: Ingested production monitoring batch with {len(df)} transactions.")
    return df

@task(name="Execute Data Drift Analysis")
def execute_drift_analysis_task(
    current_df: pd.DataFrame,
    generate_evidently_html: bool = False
) -> Dict[str, Any]:
    """Runs statistical drift detection across all numerical features."""
    logger.info("Task: Running statistical drift analysis (PSI, KS test, Evidently)...")
    res = drift_detector.run_drift_analysis(
        current_df=current_df,
        generate_evidently_html=generate_evidently_html
    )
    return res

@task(name="Evaluate Retraining Trigger")
def evaluate_retraining_trigger_task(
    drift_result: Dict[str, Any],
    auto_trigger_retraining: bool = True
) -> Dict[str, Any]:
    """
    Checks if drift status requires automated retraining.
    If Critical, dispatches retraining workflow with trigger reason.
    """
    status = drift_result["status"]
    report_id = drift_result["report_id"]
    logger.info(f"Task: Evaluating drift report {report_id} (Status: {status})")

    should_retrain = (status == "Critical")
    retraining_dispatch = None

    if should_retrain and auto_trigger_retraining:
        logger.warning(f"CRITICAL DRIFT DETECTED in report {report_id}! Triggering automated retraining...")
        from backend.workflows.retraining_flow import automated_retraining_flow
        retraining_dispatch = automated_retraining_flow(
            trigger_reason="CRITICAL_FEATURE_DRIFT",
            quick_run=True  # fast execution in automated response
        )
    else:
        logger.info(f"Drift monitoring status is '{status}'. No emergency retraining required.")

    return {
        "report_id": report_id,
        "status": status,
        "should_retrain": should_retrain,
        "retraining_dispatch": retraining_dispatch
    }

@flow(name="Automated Drift Monitoring Flow", log_prints=True)
def drift_monitoring_flow(
    sample_size: int = 2000,
    current_data_path: Optional[str] = None,
    generate_evidently_html: bool = False,
    auto_trigger_retraining: bool = True
) -> Dict[str, Any]:
    """
    Prefect Flow for scheduled data drift monitoring and automatic retraining dispatch.
    """
    logger.info("Starting Prefect Drift Monitoring Flow...")
    current_df = fetch_production_batch_task(sample_size, current_data_path)
    drift_res = execute_drift_analysis_task(current_df, generate_evidently_html)
    eval_res = evaluate_retraining_trigger_task(drift_res, auto_trigger_retraining)

    return {
        "status": "COMPLETED",
        "drift_analysis": drift_res,
        "trigger_evaluation": eval_res
    }

if __name__ == "__main__":
    drift_monitoring_flow(sample_size=1000)
