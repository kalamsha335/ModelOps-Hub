"""
Test suite for MLflow Model Registry, Stage Governance, Validation Gates, and Rollbacks.
"""

import pytest
import os
from backend.app.core.config import settings
from backend.app.database.session import SessionLocal, engine, Base
from backend.app.registry.model_registry import ModelRegistryService, ModelValidationError
from backend.app.models.schemas_db import ModelMetadata, ModelVersion, ModelDeployment
import mlflow

@pytest.fixture(scope="module")
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    # Cleanup if necessary

@pytest.fixture(scope="module")
def registry():
    return ModelRegistryService()

@pytest.fixture(scope="module")
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

def test_get_or_create_registered_model(registry, db_session):
    model_name = "test-fraud-detector"
    res = registry.get_or_create_registered_model(model_name, description="Test Model", db=db_session)
    assert res["model_name"] == model_name
    
    # Check DB
    meta = db_session.query(ModelMetadata).filter(ModelMetadata.model_name == model_name).first()
    assert meta is not None
    assert meta.model_name == model_name
    assert meta.problem_type == "binary_classification"

def test_register_and_governance_lifecycle(registry, db_session):
    """
    End-to-end governance lifecycle test:
    1. Look up existing runs from Phase 2.
    2. Register the runs as versions 1 and 2.
    3. Test validation rejection on inferior model (Logistic Regression).
    4. Test validation pass and promotion on champion model (XGBoost).
    5. Test atomic rollback.
    """
    # 1. Fetch available runs from MLflow
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME)
    runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time ASC"])
    assert len(runs) >= 2, "Expected at least 2 runs from Phase 2 training"

    lr_run_id = runs.iloc[0]["run_id"]
    xgb_run_id = runs.iloc[1]["run_id"]

    model_name = settings.MLFLOW_MODEL_NAME

    # 2. Register Version 1 (LR)
    v1_info = registry.register_model_version(
        run_id=lr_run_id,
        model_name=model_name,
        description="Logistic Regression Baseline",
        db=db_session
    )
    assert v1_info["model_name"] == model_name
    assert v1_info["stage"] == "Development"
    v1_num = v1_info["version"]

    # 3. Test validation rejection on Version 1
    # Version 1 PR-AUC is ~0.6765, which is < 0.80
    val_v1 = registry.validate_candidate_for_production(
        model_name=model_name,
        version=v1_num,
        min_pr_auc=0.80,
        min_recall=0.75,
        db=db_session
    )
    assert val_v1["is_valid"] is False
    assert val_v1["status"] == "FAILED"
    assert "PR-AUC" in val_v1["reason"]

    # Attempt to promote Version 1 with enforce_validation=True -> Must raise ModelValidationError
    with pytest.raises(ModelValidationError):
        registry.promote_stage(
            model_name=model_name,
            version=v1_num,
            target_stage="Production",
            enforce_validation=True,
            db=db_session
        )

    # 4. Register Version 2 (XGBoost)
    v2_info = registry.register_model_version(
        run_id=xgb_run_id,
        model_name=model_name,
        description="XGBoost Classifier Champion",
        db=db_session
    )
    v2_num = v2_info["version"]

    # Validate Version 2 (PR-AUC is ~0.8245, Recall ~0.7973)
    val_v2 = registry.validate_candidate_for_production(
        model_name=model_name,
        version=v2_num,
        min_pr_auc=0.80,
        min_recall=0.75,
        db=db_session
    )
    assert val_v2["is_valid"] is True
    assert val_v2["status"] == "PASSED"

    # Promote Version 2 to Staging first
    staging_res = registry.promote_stage(
        model_name=model_name,
        version=v2_num,
        target_stage="Staging",
        reason="Promoting validated champion to Staging for integration testing",
        db=db_session
    )
    assert staging_res["new_stage"] == "Staging"
    assert staging_res["is_current_production"] is False

    # Promote Version 2 to Production
    prod_res = registry.promote_stage(
        model_name=model_name,
        version=v2_num,
        target_stage="Production",
        reason="Champion passed all validation criteria. Deploying to production serving.",
        db=db_session
    )
    assert prod_res["new_stage"] == "Production"
    assert prod_res["is_current_production"] is True

    # Verify active production model query
    current_prod = registry.get_current_production_version(model_name, db=db_session)
    assert current_prod is not None
    assert current_prod["version"] == v2_num
    assert current_prod["stage"] == "Production"

    # 5. Test manual promotion override of V1 to Staging
    registry.promote_stage(
        model_name=model_name,
        version=v1_num,
        target_stage="Staging",
        reason="Staging baseline for comparison",
        db=db_session
    )

    # 6. Test Rollback: Force V1 to production with override, then rollback to V2
    registry.promote_stage(
        model_name=model_name,
        version=v1_num,
        target_stage="Production",
        enforce_validation=False,
        reason="Emergency hotfix test",
        db=db_session
    )
    assert registry.get_current_production_version(model_name, db=db_session)["version"] == v1_num

    # Roll back to V2
    rollback_res = registry.rollback_production(
        model_name=model_name,
        target_version=v2_num,
        reason="Performance degradation detected; rolling back to validated XGBoost v2",
        db=db_session
    )
    assert rollback_res["active_production_version"] == v2_num
    assert registry.get_current_production_version(model_name, db=db_session)["version"] == v2_num

    # 7. Check audit history in deployments
    deployments = db_session.query(ModelDeployment).filter(ModelDeployment.model_name == model_name).all()
    assert len(deployments) >= 3

    # 8. Check listing functions
    models_list = registry.list_models(db=db_session)
    assert any(m["model_name"] == model_name for m in models_list)

    versions_list = registry.list_versions(model_name, db=db_session)
    assert len(versions_list) >= 2
