"""
Environment and Foundation Test Suite for ModelOps Hub (Phase 1)
Validates core package imports, database connection, model table creations, and health check.
"""
import pytest

def test_imports():
    import fastapi
    import sqlalchemy
    import mlflow
    import optuna
    import sklearn
    import xgboost
    import pandas
    import numpy

    assert fastapi.__version__ is not None
    assert sqlalchemy.__version__ is not None
    assert mlflow.__version__ is not None
    assert optuna.__version__ is not None
    assert sklearn.__version__ is not None
    assert xgboost.__version__ is not None
    assert pandas.__version__ is not None
    assert numpy.__version__ is not None

def test_database_and_models():
    from backend.app.database.session import engine, SessionLocal, Base
    from backend.app.models.schemas_db import (
        TrainingRun, ModelMetadata, ModelVersion, ModelDeployment,
        PredictionLog, DriftReport, DriftAlert, PerformanceMetric, RetrainingJob
    )

    # Ensure tables can be created on active engine
    Base.metadata.create_all(bind=engine)

    # Verify session creation and basic query
    session = SessionLocal()
    assert session is not None
    session.close()

def test_api_health_endpoint():
    from backend.app.main import health_check
    from backend.app.database.session import SessionLocal

    session = SessionLocal()
    try:
        response = health_check(db=session)
        assert response["status"] == "healthy"
        assert response["service"] == "ModelOps Hub"
        assert response["database"] == "connected"
    finally:
        session.close()
