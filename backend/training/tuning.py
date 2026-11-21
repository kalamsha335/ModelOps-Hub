"""
Hyperparameter Optimization and Stratified Cross-Validation with Optuna.
Executes Bayesian optimization with Tree-structured Parzen Estimator (TPE),
maximizing PR-AUC (Precision-Recall Area Under Curve) across Stratified K-Fold splits.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

# Silence verbose Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("modelops-hub.tuning")

def tune_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 5,
    n_splits: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Tuning for Logistic Regression using Stratified Cross-Validation.
    Uses 'lbfgs' quasi-Newton solver for fast, stable convergence on large tabular datasets.
    """
    logger.info(f"Starting Optuna hyperparameter optimization for Logistic Regression ({n_trials} trials, {n_splits}-fold CV)...")

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            model = LogisticRegression(
                C=C,
                solver="lbfgs",
                class_weight=class_weight,
                max_iter=300,
                random_state=random_state
            )
            model.fit(X_fold_train, y_fold_train)

            y_val_proba = model.predict_proba(X_fold_val)[:, 1]
            fold_pr_auc = average_precision_score(y_fold_val, y_val_proba)
            cv_scores.append(fold_pr_auc)

        return float(np.mean(cv_scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params["solver"] = "lbfgs"
    logger.info(
        f"Logistic Regression Tuning Finished. Best PR-AUC: {study.best_value:.4f} | "
        f"Params: {best_params}"
    )

    return {
        "algorithm": "LogisticRegression",
        "best_params": best_params,
        "best_cv_pr_auc": float(study.best_value),
        "total_trials": n_trials
    }

def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 5,
    n_splits: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Tuning for XGBoost Classifier using Stratified Cross-Validation.
    Optimizes depth, learning rate, and scale_pos_weight for imbalance.
    """
    logger.info(f"Starting Optuna hyperparameter optimization for XGBoost ({n_trials} trials, {n_splits}-fold CV)...")

    neg_count = int(len(y_train) - y_train.sum())
    pos_count = int(y_train.sum())
    baseline_scale_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 100, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight",
                max(1.0, baseline_scale_weight * 0.2),
                baseline_scale_weight * 1.1
            ),
            "eval_metric": "logloss",
            "random_state": random_state,
            "n_jobs": -1
        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            model = XGBClassifier(**params)
            model.fit(X_fold_train, y_fold_train)

            y_val_proba = model.predict_proba(X_fold_val)[:, 1]
            fold_pr_auc = average_precision_score(y_fold_val, y_val_proba)
            cv_scores.append(fold_pr_auc)

        return float(np.mean(cv_scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"eval_metric": "logloss", "random_state": random_state, "n_jobs": -1})

    logger.info(
        f"XGBoost Tuning Finished. Best PR-AUC: {study.best_value:.4f} | "
        f"Params: {best_params}"
    )

    return {
        "algorithm": "XGBClassifier",
        "best_params": best_params,
        "best_cv_pr_auc": float(study.best_value),
        "total_trials": n_trials
    }
