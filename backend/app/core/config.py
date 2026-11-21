from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseSettings):
    # Application Info
    APP_NAME: str = "ModelOps Hub"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "dev_secret_key_modelops_hub_2026"

    # PostgreSQL Database Settings
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "modelops_db"
    DATABASE_URL: Optional[str] = None
    SQLITE_FALLBACK_URL: str = f"sqlite:///{BASE_DIR / 'modelops.db'}"

    # MLflow Settings (Database backend required for modern MLflow 3+ & Model Registry)
    MLFLOW_TRACKING_URI: str = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
    MLFLOW_EXPERIMENT_NAME: str = "credit-card-fraud-detection"
    MLFLOW_MODEL_NAME: str = "fraud-detection-model"
    ARTIFACTS_DIR: str = str(BASE_DIR / "artifacts")

    # Data Directories
    DATA_DIR: str = str(BASE_DIR / "data")
    RAW_DATA_PATH: str = str(BASE_DIR / "data" / "raw" / "creditcard.csv")
    PROCESSED_DATA_DIR: str = str(BASE_DIR / "data" / "processed")
    REFERENCE_DATA_DIR: str = str(BASE_DIR / "data" / "reference")
    CURRENT_DATA_DIR: str = str(BASE_DIR / "data" / "current")

    # Drift Detection Thresholds (Configurable Operational Rules)
    DRIFT_PSI_WARNING_THRESHOLD: float = 0.20
    DRIFT_PSI_CRITICAL_THRESHOLD: float = 0.30
    DRIFT_KS_PVALUE_THRESHOLD: float = 0.05
    DRIFT_SHARE_DRIFTED_THRESHOLD: float = 0.25

    # Retraining Guardrails & Production Criteria
    RETRAINING_MIN_PR_AUC_THRESHOLD: float = 0.80
    RETRAINING_MIN_RECALL_THRESHOLD: float = 0.82
    MAX_RETRAINING_COOLDOWN_HOURS: int = 24

    # Prefect Orchestrator
    PREFECT_API_URL: str = "http://localhost:4200/api"

    # CORS
    FRONTEND_URL: str = "http://localhost:5173"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_effective_db_url(self) -> str:
        """
        Returns PostgreSQL connection string if available and specified,
        otherwise gracefully falls back to local SQLite database.
        """
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

settings = Settings()
