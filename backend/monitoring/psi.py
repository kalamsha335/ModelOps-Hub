"""
ModelOps Hub - Population Stability Index (PSI) Drift Metric
Measures population shift between reference and current feature distributions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from backend.app.core.config import settings

def calculate_feature_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculates Population Stability Index (PSI) for a continuous numerical feature.
    
    Formula:
        PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
        
    Operational Interpretations:
        PSI < 0.20: Healthy (No significant distribution shift)
        0.20 <= PSI < 0.30: Warning (Moderate shift, monitor closely)
        PSI >= 0.30: Critical (Significant population drift, retraining recommended)
    """
    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0, {"error": "Empty array provided", "psi": 0.0, "status": "HEALTHY"}

    # Generate quantile bin edges from expected (reference) distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges = np.unique(bin_edges)  # Remove duplicate edges

    if len(bin_edges) < 2:
        # Uniform or single-value distribution
        return 0.0, {
            "psi": 0.0,
            "status": "HEALTHY",
            "bins": [],
            "expected_counts": [len(expected)],
            "actual_counts": [len(actual)]
        }

    # Extend boundary edges to cover full range
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate frequency counts in each bin
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to proportions
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Apply epsilon smoothing to prevent division by zero or ln(0)
    expected_pct = np.clip(expected_pct, epsilon, 1.0)
    actual_pct = np.clip(actual_pct, epsilon, 1.0)

    # Normalize proportions after clipping
    expected_pct = expected_pct / np.sum(expected_pct)
    actual_pct = actual_pct / np.sum(actual_pct)

    # Calculate PSI per bin and total
    bin_psi = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    total_psi = float(np.sum(bin_psi))

    # Determine status
    if total_psi >= settings.DRIFT_PSI_CRITICAL_THRESHOLD:
        status = "CRITICAL"
    elif total_psi >= settings.DRIFT_PSI_WARNING_THRESHOLD:
        status = "WARNING"
    else:
        status = "HEALTHY"

    breakdown = {
        "psi": round(total_psi, 4),
        "status": status,
        "bin_count": len(bin_edges) - 1,
        "expected_pct": [round(float(p), 4) for p in expected_pct],
        "actual_pct": [round(float(p), 4) for p in actual_pct],
        "bin_psi": [round(float(p), 4) for p in bin_psi]
    }

    return round(total_psi, 4), breakdown

def calculate_batch_psi(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    num_bins: int = 10
) -> Dict[str, Any]:
    """
    Calculates PSI for all specified numerical features across reference and current datasets.
    """
    if features is None:
        features = [col for col in reference_df.columns if col in current_df.columns and col != "Class"]

    results = {}
    psi_scores = []
    critical_features = []
    warning_features = []

    for feat in features:
        if feat not in current_df.columns or feat not in reference_df.columns:
            continue
        exp_arr = reference_df[feat].to_numpy(dtype=float)
        act_arr = current_df[feat].to_numpy(dtype=float)

        score, details = calculate_feature_psi(exp_arr, act_arr, num_bins=num_bins)
        results[feat] = details
        psi_scores.append(score)

        if details["status"] == "CRITICAL":
            critical_features.append(feat)
        elif details["status"] == "WARNING":
            warning_features.append(feat)

    mean_psi = float(np.mean(psi_scores)) if psi_scores else 0.0
    max_psi = float(np.max(psi_scores)) if psi_scores else 0.0

    if critical_features or max_psi >= settings.DRIFT_PSI_CRITICAL_THRESHOLD:
        batch_status = "CRITICAL"
    elif warning_features or max_psi >= settings.DRIFT_PSI_WARNING_THRESHOLD:
        batch_status = "WARNING"
    else:
        batch_status = "HEALTHY"

    return {
        "batch_status": batch_status,
        "mean_psi": round(mean_psi, 4),
        "max_psi": round(max_psi, 4),
        "total_features": len(features),
        "critical_features": critical_features,
        "warning_features": warning_features,
        "feature_psi": results
    }
