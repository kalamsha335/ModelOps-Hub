"""
ModelOps Hub - Two-Sample Kolmogorov-Smirnov (KS) Test
Non-parametric statistical hypothesis test to detect whether reference and current
feature distributions differ significantly.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy.stats import ks_2samp
from backend.app.core.config import settings

def calculate_feature_ks(
    expected: np.ndarray,
    actual: np.ndarray,
    alpha: Optional[float] = None
) -> Dict[str, Any]:
    """
    Performs Two-Sample Kolmogorov-Smirnov test on continuous feature distributions.
    
    Null Hypothesis (H0): Both reference and current samples are drawn from the same continuous distribution.
    Alternative Hypothesis (H1): Reference and current samples come from different distributions.
    
    If p-value < alpha (default: 0.05), reject H0 -> Drift Detected.
    """
    alpha = alpha or settings.DRIFT_KS_PVALUE_THRESHOLD

    # Filter NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "is_drifted": False,
            "error": "Empty array provided"
        }

    res = ks_2samp(expected, actual)
    statistic = float(res.statistic)
    p_value = float(res.pvalue)
    is_drifted = bool(p_value < alpha)

    return {
        "statistic": round(statistic, 4),
        "p_value": round(p_value, 6),
        "alpha": alpha,
        "is_drifted": is_drifted,
        "interpretation": "Distribution shift detected" if is_drifted else "No significant shift detected"
    }

def calculate_batch_ks(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    alpha: Optional[float] = None
) -> Dict[str, Any]:
    """
    Runs KS test across all specified continuous features.
    """
    alpha = alpha or settings.DRIFT_KS_PVALUE_THRESHOLD
    if features is None:
        features = [col for col in reference_df.columns if col in current_df.columns and col != "Class"]

    results = {}
    drifted_features = []

    for feat in features:
        if feat not in current_df.columns or feat not in reference_df.columns:
            continue
        exp_arr = reference_df[feat].to_numpy(dtype=float)
        act_arr = current_df[feat].to_numpy(dtype=float)

        res = calculate_feature_ks(exp_arr, act_arr, alpha=alpha)
        results[feat] = res
        if res["is_drifted"]:
            drifted_features.append(feat)

    share_drifted = len(drifted_features) / len(features) if features else 0.0

    return {
        "drifted_features_count": len(drifted_features),
        "total_features": len(features),
        "share_drifted": round(share_drifted, 4),
        "drifted_features": drifted_features,
        "feature_ks": results
    }

