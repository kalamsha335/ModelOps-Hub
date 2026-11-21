"""
Data Preprocessing, Stratified Splitting, and Feature Transformation Pipeline.
Guarantees zero data leakage by fitting transformation parameters exclusively on
the training set and provides a serializable preprocessor for FastAPI production serving.
"""

import os
import joblib
import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from backend.app.core.config import settings

logger = logging.getLogger("modelops-hub.preprocessing")

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
TARGET_COLUMN = "Class"

class CreditCardPreprocessor(BaseEstimator, TransformerMixin):
    """
    Serializable feature preprocessor for tabular fraud detection.
    - Fits RobustScaler on 'Amount' and 'Time' (robust against extreme outliers).
    - Preserves PCA-transformed numerical features 'V1'-'V28'.
    - Stores fitted medians and IQRs for deterministic online inference.
    """
    def __init__(self):
        self.time_scaler = RobustScaler()
        self.amount_scaler = RobustScaler()
        self.feature_names = FEATURE_COLUMNS
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fits scalers strictly on training features.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)

        self.time_scaler.fit(X[["Time"]])
        self.amount_scaler.fit(X[["Amount"]])
        self.is_fitted = True
        logger.info("CreditCardPreprocessor successfully fit on training partition.")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms features using parameters learned during fit.
        Guarantees deterministic feature ordering [V1..V28, scaled_Time, scaled_Amount].
        """
        if not self.is_fitted:
            raise RuntimeError("CreditCardPreprocessor must be fit before calling transform.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)

        X_copy = X.copy()
        scaled_time = self.time_scaler.transform(X_copy[["Time"]])
        scaled_amount = self.amount_scaler.transform(X_copy[["Amount"]])

        # Extract PCA features V1-V28
        v_features = [f"V{i}" for i in range(1, 29)]
        pca_data = X_copy[v_features].values

        # Concatenate: V1-V28 followed by scaled Time and scaled Amount
        transformed = np.hstack([pca_data, scaled_time, scaled_amount])
        return transformed

    def transform_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms features and returns a labeled DataFrame with identical column names.
        """
        arr = self.transform(X)
        cols = [f"V{i}" for i in range(1, 29)] + ["scaled_Time", "scaled_Amount"]
        return pd.DataFrame(arr, columns=cols, index=X.index)

    def save(self, filepath: str):
        """Serializes fitted preprocessor to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor serialized to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "CreditCardPreprocessor":
        """Deserializes preprocessor artifact."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor artifact not found at {filepath}")
        return joblib.load(filepath)

def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Performs a 2-stage stratified split:
    Train (70%), Validation (15%), Test (15%).
    Preserves exact minority class ratio across all splits.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # First split off the test set (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Calculate validation proportion relative to remaining data
    # (0.15 / 0.85 approx 0.1765)
    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=adjusted_val_size,
        stratify=y_train_val,
        random_state=random_state
    )

    logger.info(
        f"Data split completed: "
        f"Train={len(X_train):,} ({y_train.sum()} frauds, {y_train.mean():.4%}) | "
        f"Val={len(X_val):,} ({y_val.sum()} frauds, {y_val.mean():.4%}) | "
        f"Test={len(X_test):,} ({y_test.sum()} frauds, {y_test.mean():.4%})"
    )

    # Save reference baseline dataset for drift monitoring
    save_reference_dataset(X_val, y_val)

    return X_train, y_train, X_val, y_val, X_test, y_test

def save_reference_dataset(X_ref: pd.DataFrame, y_ref: pd.Series):
    """
    Persists baseline reference partition as parquet for Evidently AI drift engine.
    """
    ref_dir = settings.REFERENCE_DATA_DIR
    os.makedirs(ref_dir, exist_ok=True)
    ref_df = X_ref.copy()
    ref_df[TARGET_COLUMN] = y_ref.values
    ref_path = os.path.join(ref_dir, "reference_data.parquet")
    ref_df.to_parquet(ref_path, index=False)
    logger.info(f"Persisted reference baseline dataset ({len(ref_df):,} rows) to: {ref_path}")
