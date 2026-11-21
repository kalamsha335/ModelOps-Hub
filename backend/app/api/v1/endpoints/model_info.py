"""
ModelOps Hub - Model Information Endpoint
Returns active production model metadata, algorithm type, version, and input schema.
"""

from fastapi import APIRouter, status
from backend.app.services.serving import serving_service

router = APIRouter(prefix="", tags=["Model Info"])

@router.get("/model/info", status_code=status.HTTP_200_OK)
def get_model_info():
    """
    Returns information regarding the active serving model in production.
    """
    return serving_service.get_model_info()
