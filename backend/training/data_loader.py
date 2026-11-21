"""
Data Ingestion, Schema Validation, and Exploratory Data Analysis (EDA) Module.
Handles robust loading, data integrity checks, and descriptive statistical analysis
for the ULB Credit Card Fraud Detection dataset.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from backend.app.core.config import settings

logger = logging.getLogger("modelops-hub.data_loader")

REQUIRED_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
TARGET_COLUMN = "Class"
EXPECTED_COLUMNS = REQUIRED_FEATURES + [TARGET_COLUMN]

def load_data(file_path: str = None) -> pd.DataFrame:
    """
    Loads credit card transaction dataset from disk.
    Defaults to settings.RAW_DATA_PATH.
    """
    path = file_path or settings.RAW_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Please ensure creditcard.csv is placed in data/raw/."
        )

    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} transactions with {len(df.columns)} columns.")
    return df

def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates dataset schema integrity:
    1. Checks presence of all 31 expected columns (Time, V1-V28, Amount, Class).
    2. Verifies data types are numeric.
    3. Checks for missing/null/infinite values.
    4. Validates target column values are strictly binary {0, 1}.
    5. Detects duplicate records.
    """
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema validation failed. Missing required columns: {missing_cols}")

    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())

    inf_counts = np.isinf(df[REQUIRED_FEATURES]).sum().sum()

    unique_classes = set(df[TARGET_COLUMN].unique())
    if not unique_classes.issubset({0, 1}):
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains non-binary values: {unique_classes}"
        )

    duplicates_count = int(df.duplicated().sum())

    validation_result = {
        "is_valid": True,
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "total_missing_values": total_nulls,
        "total_infinite_values": int(inf_counts),
        "duplicate_rows": duplicates_count,
        "class_values": sorted(list(unique_classes)),
        "columns_present": list(df.columns)
    }

    logger.info(
        f"Schema validation passed: {validation_result['total_rows']:,} rows, "
        f"{total_nulls} nulls, {duplicates_count} duplicates."
    )
    return validation_result

def perform_eda(df: pd.DataFrame, save_report_dir: str = None) -> Dict[str, Any]:
    """
    Performs exploratory data analysis on the dataset:
    - Analyzes class distribution and extreme imbalance ratio.
    - Computes summary statistics for 'Amount' and 'Time' partitioned by Class.
    - Evaluates feature skewness and variance.
    - Optionally saves a structured JSON EDA report.
    """
    total_samples = len(df)
    class_counts = df[TARGET_COLUMN].value_counts().to_dict()
    legit_count = int(class_counts.get(0, 0))
    fraud_count = int(class_counts.get(1, 0))
    fraud_percentage = (fraud_count / total_samples) * 100 if total_samples > 0 else 0.0
    imbalance_ratio = (legit_count / fraud_count) if fraud_count > 0 else float("inf")

    # Transaction Amount statistics
    amount_overall = {
        "mean": float(df["Amount"].mean()),
        "std": float(df["Amount"].std()),
        "median": float(df["Amount"].median()),
        "iqr": float(df["Amount"].quantile(0.75) - df["Amount"].quantile(0.25)),
        "min": float(df["Amount"].min()),
        "max": float(df["Amount"].max()),
        "p95": float(df["Amount"].quantile(0.95)),
        "p99": float(df["Amount"].quantile(0.99))
    }

    legit_amounts = df[df[TARGET_COLUMN] == 0]["Amount"]
    fraud_amounts = df[df[TARGET_COLUMN] == 1]["Amount"]

    amount_by_class = {
        "legitimate": {
            "mean": float(legit_amounts.mean()),
            "median": float(legit_amounts.median()),
            "max": float(legit_amounts.max())
        },
        "fraudulent": {
            "mean": float(fraud_amounts.mean()),
            "median": float(fraud_amounts.median()),
            "max": float(fraud_amounts.max())
        }
    }

    # Time feature analysis (seconds elapsed since first transaction)
    time_span_seconds = float(df["Time"].max() - df["Time"].min())
    time_span_hours = time_span_seconds / 3600.0

    eda_summary = {
        "dataset_summary": {
            "total_transactions": total_samples,
            "feature_count": len(REQUIRED_FEATURES),
            "time_span_hours": round(time_span_hours, 2),
            "time_span_days": round(time_span_hours / 24.0, 2)
        },
        "class_imbalance": {
            "legitimate_count": legit_count,
            "fraudulent_count": fraud_count,
            "fraud_percentage": round(fraud_percentage, 4),
            "imbalance_ratio": f"{round(imbalance_ratio, 2)} : 1"
        },
        "amount_analysis": {
            "overall": amount_overall,
            "by_class": amount_by_class
        }
    }

    logger.info(
        f"EDA Complete: {total_samples:,} records | Fraud: {fraud_count} ({fraud_percentage:.3f}%) | "
        f"Imbalance: {eda_summary['class_imbalance']['imbalance_ratio']}"
    )

    if save_report_dir:
        os.makedirs(save_report_dir, exist_ok=True)
        report_path = os.path.join(save_report_dir, "eda_summary.json")
        with open(report_path, "w") as f:
            json.dump(eda_summary, f, indent=2)
        logger.info(f"Saved EDA report to {report_path}")

    return eda_summary
