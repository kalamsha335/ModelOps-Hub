"""
ModelOps Hub - Registry Governance Endpoints
Provides model inspection, version tracking, validation gates, promotion, and atomic rollback.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.schemas.schemas_api import PromoteRequest, RollbackRequest
from backend.app.registry.model_registry import registry_service, ModelValidationError
from backend.app.services.serving import serving_service

router = APIRouter(prefix="/registry", tags=["Model Registry & Governance"])

@router.get("/models", status_code=status.HTTP_200_OK)
def list_registered_models(db: Session = Depends(get_db)):
    """Lists all registered machine learning models with active production pointers."""
    return registry_service.list_models(db=db)

@router.get("/models/{model_name}/versions", status_code=status.HTTP_200_OK)
def list_model_versions(model_name: str, db: Session = Depends(get_db)):
    """Lists all versions, lifecycle stages, metrics, and validation status for a registered model."""
    versions = registry_service.list_versions(model_name, db=db)
    if not versions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No versions found for model '{model_name}'"
        )
    return versions

@router.get("/models/{model_name}/versions/{version}/artifacts", status_code=status.HTTP_200_OK)
def get_model_version_artifacts(model_name: str, version: str):
    """Retrieves logged MLflow artifacts for a specific model version."""
    return registry_service.get_model_artifacts(model_name, version)

@router.post("/promote", status_code=status.HTTP_200_OK)
def promote_model_stage(payload: PromoteRequest, db: Session = Depends(get_db)):
    """
    Promotes a model version to a target stage (Staging, Production, Archived).
    Enforces quantitative validation gates (PR-AUC >= 0.80, Recall >= 0.75) for Production promotions.
    """
    try:
        res = registry_service.promote_stage(
            model_name=payload.model_name,
            version=payload.version,
            target_stage=payload.target_stage,
            reason=payload.reason,
            deployed_by="user_dashboard",
            enforce_validation=payload.enforce_validation,
            db=db
        )
        # Notify serving service to hot-reload if promoted to Production
        if payload.target_stage.capitalize() == "Production":
            serving_service.reload_production_model(force=True)
        return res
    except ModelValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/rollback", status_code=status.HTTP_200_OK)
def rollback_production_model(payload: RollbackRequest, db: Session = Depends(get_db)):
    """
    Atomically rolls back the production model pointer to a previous validated model version.
    """
    try:
        res = registry_service.rollback_production(
            model_name=payload.model_name,
            target_version=payload.target_version,
            reason=payload.reason,
            deployed_by="user_dashboard",
            db=db
        )
        # Hot-reload newly restored production model into serving service
        serving_service.reload_production_model(force=True)
        return res
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
