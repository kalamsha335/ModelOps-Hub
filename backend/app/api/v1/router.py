"""
ModelOps Hub - API v1 Router Aggregator
Mounts modular endpoint routers under /api/v1 prefix.
"""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import (
    predict, model_info, registry, monitoring, retraining, training
)

api_router = APIRouter()

# Prediction & Inference
api_router.include_router(predict.router)
api_router.include_router(model_info.router)

# Model Registry & Lifecycle Governance
api_router.include_router(registry.router)

# Drift & Performance Monitoring
api_router.include_router(monitoring.router)

# Orchestration & Retraining
api_router.include_router(retraining.router)

# Training Run History
api_router.include_router(training.router)

