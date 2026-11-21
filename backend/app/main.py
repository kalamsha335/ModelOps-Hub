"""
ModelOps Hub - FastAPI Main Application
Entry point for REST APIs, dynamic model serving, monitoring, registry governance,
and workflow orchestration.
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from datetime import datetime, timezone
from pathlib import Path
import logging
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.app.core.config import settings
from backend.app.database.session import get_db, engine, Base
from backend.app.api.v1.router import api_router
from backend.app.services.serving import serving_service

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("modelops-hub")

# Ensure database tables exist
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.APP_NAME,
    description="Automated Model Training, Registry, Drift Monitoring, and Retraining Platform for Tabular ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 endpoints
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.on_event("startup")
def on_startup():
    """Initializes and loads the production model upon service startup."""
    logger.info("Initializing ModelOps Hub API...")
    try:
        serving_service.reload_production_model()
        logger.info("Production serving model initialized.")
    except Exception as e:
        logger.warning(f"Could not load initial production model: {e}")

@app.get("/health", tags=["System Health"])
def health_check(db: Session = Depends(get_db)):
    """
    Verifies that the backend API, underlying database connection,
    and runtime environment are operational.
    """
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
        "database": db_status,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/reports/html/{report_id}", response_class=HTMLResponse, tags=["Monitoring"])
def get_evidently_html_report(report_id: str):
    """
    Renders generated Evidently AI HTML report for inline viewing in dashboard iframes.
    """
    reports_dir = Path(settings.ARTIFACTS_DIR) / "monitoring" / "reports"
    file_path = reports_dir / f"{report_id}.html"
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"HTML report '{report_id}.html' not found"
        )
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/", tags=["Root"])
def root():
    return {
        "message": f"Welcome to {settings.APP_NAME} API",
        "docs": "/docs",
        "health": "/health",
        "api_v1": settings.API_V1_STR,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
