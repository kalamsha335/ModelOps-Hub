"""
ModelOps Hub - Dynamic Model Serving Service
Provides dynamic production model hot-reloading from MLflow Model Registry / Champion Artifacts,
leakage-free preprocessing, real-time prediction with calibrated fraud probabilities,
latency instrumentation, and prediction logging to PostgreSQL.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.database.session import SessionLocal
from backend.app.models.schemas_db import PredictionLog, ModelVersion
from backend.app.schemas.schemas_api import (
    TransactionFeatures, PredictionResponse, BatchPredictionResponse
)
from backend.app.registry.model_registry import registry_service

logger = logging.getLogger("modelops.services.serving")

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]

class ModelServingService:
    """
    High-performance serving service with automatic production hot-reloading.
    """

    def __init__(self):
        self._model = None
        self._preprocessor = None
        self._model_version: Optional[str] = None
        self._model_name: str = settings.MLFLOW_MODEL_NAME
        self._algorithm: str = "Unknown"
        self._last_loaded_at: Optional[float] = None
        self.reload_production_model()

    def reload_production_model(self, force: bool = False) -> bool:
        """
        Dynamically loads the active Production model from registry or champion artifacts.
        If a new production version was promoted, it hot-reloads seamlessly.
        """
        try:
            # Check registry for active Production model
            prod_info = registry_service.get_current_production_version(self._model_name)

            if prod_info and (force or prod_info["version"] != self._model_version):
                new_version = prod_info["version"]
                logger.info(f"Detected new production model version '{new_version}'. Hot-reloading...")

                # 1. Try loading from MLflow artifact path or champion folder
                champion_dir = Path(settings.ARTIFACTS_DIR) / "champion"
                model_file = champion_dir / "model.joblib"
                preprocessor_file = champion_dir / "preprocessor.joblib"

                if model_file.exists() and preprocessor_file.exists():
                    self._model = joblib.load(model_file)
                    self._preprocessor = joblib.load(preprocessor_file)
                    self._model_version = str(new_version)
                    self._algorithm = prod_info.get("algorithm", "XGBClassifier")
                    self._last_loaded_at = time.time()
                    logger.info(f"Loaded production model v{new_version} from champion artifacts successfully.")
                    return True

            # Fallback: Load champion artifact directly if not yet in memory
            if self._model is None:
                champion_dir = Path(settings.ARTIFACTS_DIR) / "champion"
                model_file = champion_dir / "model.joblib"
                preprocessor_file = champion_dir / "preprocessor.joblib"
                info_file = champion_dir / "champion_info.json"

                if model_file.exists() and preprocessor_file.exists():
                    self._model = joblib.load(model_file)
                    self._preprocessor = joblib.load(preprocessor_file)
                    algo_name = "XGBClassifier"
                    if info_file.exists():
                        try:
                            with open(info_file, "r") as f:
                                meta = json.load(f)
                                algo_name = meta.get("champion_algorithm", "XGBClassifier")
                        except Exception:
                            pass
                    self._model_version = prod_info["version"] if prod_info else "1"
                    self._algorithm = algo_name
                    self._last_loaded_at = time.time()
                    logger.info(f"Loaded fallback champion model v{self._model_version} ({self._algorithm}).")
                    return True

            return False
        except Exception as e:
            logger.error(f"Failed to reload production model: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Returns details about the currently active production serving model."""
        # Check if version has been updated in DB
        prod_info = registry_service.get_current_production_version(self._model_name)
        if prod_info and prod_info["version"] != self._model_version:
            self.reload_production_model(force=True)

        return {
            "model_name": self._model_name,
            "model_version": self._model_version or "uninitialized",
            "algorithm": self._algorithm,
            "status": "READY" if self._model is not None else "NOT_LOADED",
            "last_loaded_at": self._last_loaded_at,
            "features_expected": len(FEATURE_COLUMNS),
            "feature_names": FEATURE_COLUMNS
        }

    def predict_single(
        self,
        features: TransactionFeatures,
        db: Optional[Session] = None
    ) -> PredictionResponse:
        """
        Serves inference for a single credit card transaction:
        Preprocesses features, calculates probability, measures latency, and logs to database.
        """
        t0 = time.perf_counter()

        # Check if hot-reload needed
        prod_info = registry_service.get_current_production_version(self._model_name)
        if prod_info and prod_info["version"] != self._model_version:
            self.reload_production_model(force=True)

        if self._model is None or self._preprocessor is None:
            self.reload_production_model(force=True)
            if self._model is None:
                raise RuntimeError("Production model is not loaded. Please train or promote a model.")

        # Convert to DataFrame
        feat_dict = features.model_dump()
        input_df = pd.DataFrame([feat_dict])[FEATURE_COLUMNS]

        # Leakage-Free Preprocessing Transformation
        X_transformed = self._preprocessor.transform(input_df)

        # Inference
        probs = self._model.predict_proba(X_transformed)[0]
        fraud_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        pred_label_int = 1 if fraud_prob >= 0.5 else 0
        pred_label_str = "Fraudulent" if pred_label_int == 1 else "Legitimate"

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Log prediction to database
        pred_log_id = None
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            log_record = PredictionLog(
                model_name=self._model_name,
                model_version=self._model_version or "1",
                request_features=feat_dict,
                prediction=pred_label_int,
                prediction_label=pred_label_str,
                fraud_probability=round(fraud_prob, 5),
                latency_ms=latency_ms
            )
            db.add(log_record)
            db.commit()
            db.refresh(log_record)
            pred_log_id = log_record.id
        except Exception as e:
            logger.warning(f"Could not persist prediction log to DB: {e}")
            if db:
                db.rollback()
        finally:
            if close_db:
                db.close()

        return PredictionResponse(
            prediction=pred_label_int,
            prediction_label=pred_label_str,
            fraud_probability=round(fraud_prob, 5),
            model_name=self._model_name,
            model_version=self._model_version or "1",
            latency_ms=latency_ms,
            prediction_id=pred_log_id
        )

    def predict_batch(
        self,
        transactions: List[TransactionFeatures],
        db: Optional[Session] = None
    ) -> BatchPredictionResponse:
        """
        Serves vectorized batch prediction for high throughput.
        """
        t0 = time.perf_counter()

        if self._model is None or self._preprocessor is None:
            self.reload_production_model(force=True)
            if self._model is None:
                raise RuntimeError("Production model is not loaded.")

        # Batch DataFrame
        data_list = [t.model_dump() for t in transactions]
        batch_df = pd.DataFrame(data_list)[FEATURE_COLUMNS]

        # Vectorized transform & predict
        X_trans = self._preprocessor.transform(batch_df)
        probs = self._model.predict_proba(X_trans)

        total_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        avg_item_latency = round(total_latency_ms / len(transactions), 2)

        predictions_out = []
        log_records = []

        for i, p_row in enumerate(probs):
            f_prob = float(p_row[1]) if len(p_row) > 1 else float(p_row[0])
            pred_int = 1 if f_prob >= 0.5 else 0
            label_str = "Fraudulent" if pred_int == 1 else "Legitimate"

            predictions_out.append(PredictionResponse(
                prediction=pred_int,
                prediction_label=label_str,
                fraud_probability=round(f_prob, 5),
                model_name=self._model_name,
                model_version=self._model_version or "1",
                latency_ms=avg_item_latency
            ))

            log_records.append(PredictionLog(
                model_name=self._model_name,
                model_version=self._model_version or "1",
                request_features=data_list[i],
                prediction=pred_int,
                prediction_label=label_str,
                fraud_probability=round(f_prob, 5),
                latency_ms=avg_item_latency
            ))

        # Bulk log to DB
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            db.bulk_save_objects(log_records)
            db.commit()
        except Exception as e:
            logger.warning(f"Could not persist batch prediction logs: {e}")
            if db:
                db.rollback()
        finally:
            if close_db:
                db.close()

        return BatchPredictionResponse(
            predictions=predictions_out,
            total_processed=len(transactions),
            total_latency_ms=total_latency_ms,
            model_name=self._model_name,
            model_version=self._model_version or "1"
        )

# Module-level serving service
serving_service = ModelServingService()

