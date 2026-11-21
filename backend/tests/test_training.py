"""
Unit and Integration Test Suite for ML Training Pipeline (Phase 2).
Validates schema validation, exploratory data analysis, leakage-free preprocessing,
stratified partitioning, and evaluation metrics computation.
"""

import os
import pytest
import numpy as np
import pandas as pd
from backend.training.data_loader import validate_schema, perform_eda
from backend.training.preprocessing import CreditCardPreprocessor, stratified_split
from backend.training.evaluate import evaluate_model, generate_comparison_report
from sklearn.linear_model import LogisticRegression

@pytest.fixture
def sample_fraud_data():
    """Generates a small synthetic dataset mirroring exact creditcard.csv schema."""
    np.random.seed(42)
    n_samples = 1000
    n_frauds = 10  # 1% fraud rate for quick testing

    data = {f"V{i}": np.random.randn(n_samples) for i in range(1, 29)}
    data["Time"] = np.random.uniform(0, 172800, n_samples)
    data["Amount"] = np.random.exponential(scale=50, size=n_samples)

    labels = np.zeros(n_samples, dtype=int)
    fraud_indices = np.random.choice(n_samples, size=n_frauds, replace=False)
    labels[fraud_indices] = 1
    data["Class"] = labels

    return pd.DataFrame(data)

def test_schema_validation_success(sample_fraud_data):
    """Verifies that a valid dataset passes schema validation."""
    result = validate_schema(sample_fraud_data)
    assert result["is_valid"] is True
    assert result["total_rows"] == 1000
    assert result["total_columns"] == 31
    assert result["total_missing_values"] == 0

def test_schema_validation_failure():
    """Verifies that missing columns trigger a ValueError."""
    invalid_df = pd.DataFrame({"V1": [1.0], "Class": [0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(invalid_df)

def test_perform_eda(sample_fraud_data):
    """Verifies exploratory data analysis statistics."""
    eda = perform_eda(sample_fraud_data)
    assert eda["dataset_summary"]["total_transactions"] == 1000
    assert eda["class_imbalance"]["fraudulent_count"] == 10
    assert eda["class_imbalance"]["legitimate_count"] == 990
    assert "amount_analysis" in eda

def test_stratified_split(sample_fraud_data):
    """Verifies stratified train/val/test proportions and class preservation."""
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_split(
        sample_fraud_data, test_size=0.15, val_size=0.15, random_state=42
    )

    assert len(X_train) == 700
    assert len(X_val) == 150
    assert len(X_test) == 150

    # Ensure minority class is represented in every split
    assert y_train.sum() > 0
    assert y_val.sum() > 0
    assert y_test.sum() > 0

def test_preprocessor_fit_transform(sample_fraud_data):
    """Verifies RobustScaler transformation and absence of NaNs."""
    X = sample_fraud_data[[f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]]
    preprocessor = CreditCardPreprocessor()
    preprocessor.fit(X)

    X_transformed = preprocessor.transform(X)
    assert X_transformed.shape == (1000, 30)
    assert not np.isnan(X_transformed).any()

    # Verify transform_df preserves feature names
    df_transformed = preprocessor.transform_df(X)
    assert "scaled_Time" in df_transformed.columns
    assert "scaled_Amount" in df_transformed.columns
    assert len(df_transformed.columns) == 30

def test_evaluate_model():
    """Verifies model evaluation computes PR-AUC and confusion matrix."""
    np.random.seed(42)
    X_dummy = np.random.randn(200, 30)
    y_dummy = np.random.choice([0, 1], size=200, p=[0.9, 0.1])

    clf = LogisticRegression()
    clf.fit(X_dummy, y_dummy)

    results = evaluate_model(clf, X_dummy, y_dummy, feature_names=[f"F{i}" for i in range(30)])
    metrics = results["metrics"]

    assert "pr_auc" in metrics
    assert "recall" in metrics
    assert "precision" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert results["confusion_matrix"]["true_positives"] >= 0

def test_champion_model_selection():
    """Verifies comparison report correctly ranks models by PR-AUC and recall."""
    mock_results = {
        "ModelA": {
            "metrics": {"pr_auc": 0.85, "recall": 0.82, "precision": 0.88, "f1_score": 0.85, "roc_auc": 0.95},
            "training_duration_seconds": 12.0
        },
        "ModelB": {
            "metrics": {"pr_auc": 0.72, "recall": 0.70, "precision": 0.75, "f1_score": 0.72, "roc_auc": 0.90},
            "training_duration_seconds": 5.0
        }
    }

    report = generate_comparison_report(mock_results)
    assert report["champion_model"] == "ModelA"
    assert report["rankings"][0]["model_name"] == "ModelA"

