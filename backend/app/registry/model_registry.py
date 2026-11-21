"""
ModelOps Hub - MLflow Model Registry & Stage Governance Service
Provides model registration, lifecycle stage transitions (Development -> Staging -> Production),
governance validation gates, atomic rollbacks, and dual-state synchronization with MLflow and PostgreSQL.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.database.session import SessionLocal
from backend.app.models.schemas_db import ModelMetadata, ModelVersion, ModelDeployment

logger = logging.getLogger("modelops.registry")

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class ModelValidationError(Exception):
    """Raised when a candidate model fails validation gates for production promotion."""
    pass

class ModelRegistryService:
    """
    Production-grade Model Registry & Governance orchestrator.
    Bridges MLflow Model Registry (artifacts, lineage, aliases) and SQL Database (audit trail, deployments, metrics).
    """

    VALID_STAGES = ["Development", "Staging", "Production", "Archived"]

    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def get_or_create_registered_model(
        self,
        model_name: str,
        description: Optional[str] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Ensures registered model exists in both MLflow Model Registry and application DB.
        """
        description = description or "Automated Credit Card Fraud Detection Model"

        # 1. Ensure in MLflow
        try:
            reg_model = self.client.get_registered_model(model_name)
        except Exception:
            try:
                reg_model = self.client.create_registered_model(
                    name=model_name,
                    description=description,
                    tags={"framework": "scikit-learn/xgboost", "task": "fraud-detection"}
                )
            except Exception as e:
                # May fail if created concurrently
                reg_model = self.client.get_registered_model(model_name)

        # 2. Ensure in Database
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            meta = db.query(ModelMetadata).filter(ModelMetadata.model_name == model_name).first()
            if not meta:
                meta = ModelMetadata(
                    model_name=model_name,
                    description=description,
                    problem_type="binary_classification",
                    target_column="Class"
                )
                db.add(meta)
                db.commit()
                db.refresh(meta)

            return {
                "model_name": model_name,
                "description": meta.description,
                "created_at": meta.created_at.isoformat() if meta.created_at else None
            }
        finally:
            if close_db:
                db.close()

    def register_model_version(
        self,
        run_id: str,
        model_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Registers an MLflow run's model artifact as a new version in the Model Registry.
        Sets initial stage to 'Development' and synchronizes with PostgreSQL.
        """
        model_name = model_name or settings.MLFLOW_MODEL_NAME
        self.get_or_create_registered_model(model_name, description, db=db)

        # Verify run exists and gather metadata
        run = self.client.get_run(run_id)
        run_metrics = run.data.metrics
        run_params = run.data.params
        algorithm = run.data.tags.get("algorithm", "UnknownModel")

        model_uri = f"runs:/{run_id}/model"
        logger.info(f"Registering model from run {run_id} as '{model_name}'...")

        # Register in MLflow
        try:
            mv = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id,
                tags=tags or {"algorithm": algorithm, "registered_by": "ModelRegistryService"},
                description=description or f"Trained {algorithm} model from run {run_id}"
            )
            version_str = str(mv.version)
        except Exception as e:
            # Fallback to register_model API
            reg = mlflow.register_model(model_uri=model_uri, name=model_name)
            version_str = str(reg.version)

        # Assign development alias and stage
        try:
            self.client.set_registered_model_alias(model_name, "development", version_str)
        except Exception as e:
            logger.warning(f"Could not set MLflow alias 'development': {e}")

        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version_str,
                stage="None",
                archive_existing_versions=False
            )
        except Exception:
            pass  # Stages are deprecated in newer MLflow in favor of aliases

        # Synchronize with Database
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            existing_db_ver = db.query(ModelVersion).filter(
                ModelVersion.model_name == model_name,
                ModelVersion.version == version_str
            ).first()

            if not existing_db_ver:
                db_ver = ModelVersion(
                    model_name=model_name,
                    version=version_str,
                    mlflow_run_id=run_id,
                    algorithm=algorithm,
                    stage="Development",
                    is_current_production=False,
                    validation_status="PENDING",
                    metrics=run_metrics,
                    artifacts_path=model_uri,
                    created_at=utc_now()
                )
                db.add(db_ver)
                db.commit()
                db.refresh(db_ver)
            else:
                db_ver = existing_db_ver

            logger.info(f"Registered model '{model_name}' version {version_str} successfully.")
            return {
                "model_name": model_name,
                "version": version_str,
                "run_id": run_id,
                "algorithm": algorithm,
                "stage": "Development",
                "validation_status": db_ver.validation_status,
                "metrics": run_metrics,
                "created_at": db_ver.created_at.isoformat()
            }
        finally:
            if close_db:
                db.close()

    def validate_candidate_for_production(
        self,
        model_name: str,
        version: str,
        min_pr_auc: Optional[float] = None,
        min_recall: Optional[float] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Enforces quantitative validation gates before a model can enter Production.
        Evaluates Precision-Recall AUC and Recall to safeguard against fraud detection regressions.
        """
        min_pr_auc = min_pr_auc if min_pr_auc is not None else settings.RETRAINING_MIN_PR_AUC_THRESHOLD
        min_recall = min_recall if min_recall is not None else settings.RETRAINING_MIN_RECALL_THRESHOLD

        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            db_ver = db.query(ModelVersion).filter(
                ModelVersion.model_name == model_name,
                ModelVersion.version == str(version)
            ).first()

            metrics = {}
            if db_ver and db_ver.metrics:
                metrics = db_ver.metrics
            elif db_ver and db_ver.mlflow_run_id:
                run = self.client.get_run(db_ver.mlflow_run_id)
                metrics = run.data.metrics
            else:
                # Query MLflow directly
                mv = self.client.get_model_version(model_name, str(version))
                if mv.run_id:
                    run = self.client.get_run(mv.run_id)
                    metrics = run.data.metrics

            pr_auc = metrics.get("pr_auc", 0.0)
            recall = metrics.get("recall", 0.0)
            f1 = metrics.get("f1_score", 0.0)

            passed_pr_auc = pr_auc >= min_pr_auc
            passed_recall = recall >= min_recall
            is_valid = passed_pr_auc and passed_recall

            reasons = []
            if not passed_pr_auc:
                reasons.append(f"PR-AUC ({pr_auc:.4f}) below threshold ({min_pr_auc:.4f})")
            if not passed_recall:
                reasons.append(f"Recall ({recall:.4f}) below threshold ({min_recall:.4f})")

            status = "PASSED" if is_valid else "FAILED"
            reason_text = "Validation passed all criteria." if is_valid else "; ".join(reasons)

            # Update DB validation status
            if db_ver:
                db_ver.validation_status = status
                db.commit()

            return {
                "model_name": model_name,
                "version": str(version),
                "is_valid": is_valid,
                "status": status,
                "reason": reason_text,
                "criteria": {
                    "min_pr_auc": min_pr_auc,
                    "min_recall": min_recall
                },
                "metrics": {
                    "pr_auc": pr_auc,
                    "recall": recall,
                    "f1_score": f1
                }
            }
        finally:
            if close_db:
                db.close()

    def promote_stage(
        self,
        model_name: str,
        version: str,
        target_stage: str,
        reason: Optional[str] = None,
        deployed_by: str = "system",
        enforce_validation: bool = True,
        min_pr_auc: Optional[float] = None,
        min_recall: Optional[float] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Promotes a model version to a target stage (Development, Staging, Production, Archived).
        If promoting to Production, strictly validates candidate metrics and performs safe transition.
        """
        target_stage = target_stage.capitalize()
        if target_stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid stage '{target_stage}'. Must be one of {self.VALID_STAGES}")

        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            # 1. Fetch DB record
            db_ver = db.query(ModelVersion).filter(
                ModelVersion.model_name == model_name,
                ModelVersion.version == str(version)
            ).first()

            if not db_ver:
                raise ValueError(f"Model '{model_name}' version '{version}' not found in registry.")

            # 2. Production Promotion Validation Gate
            if target_stage == "Production":
                if enforce_validation:
                    val_res = self.validate_candidate_for_production(
                        model_name=model_name,
                        version=version,
                        min_pr_auc=min_pr_auc,
                        min_recall=min_recall,
                        db=db
                    )
                    if not val_res["is_valid"]:
                        raise ModelValidationError(
                            f"Model promotion rejected. Validation failed: {val_res['reason']}"
                        )
                else:
                    db_ver.validation_status = "OVERRIDDEN"

            # 3. Handle Production Specific Transitions
            previous_prod_version = None
            if target_stage == "Production":
                # Find current production model to demote
                current_prod = db.query(ModelVersion).filter(
                    ModelVersion.model_name == model_name,
                    ModelVersion.is_current_production == True
                ).first()

                if current_prod and current_prod.version != str(version):
                    previous_prod_version = current_prod.version
                    current_prod.is_current_production = False
                    current_prod.stage = "Staging"  # Demoted to staging / fallback
                    logger.info(f"Demoted version {previous_prod_version} from Production to Staging.")

                db_ver.is_current_production = True
                db_ver.promoted_at = utc_now()

                # Set MLflow Aliases
                try:
                    self.client.set_registered_model_alias(model_name, "production", str(version))
                    self.client.set_registered_model_alias(model_name, "champion", str(version))
                    if previous_prod_version:
                        self.client.set_registered_model_alias(model_name, "previous_production", previous_prod_version)
                except Exception as e:
                    logger.warning(f"Failed to set MLflow production alias: {e}")

            elif target_stage == "Staging":
                if db_ver.is_current_production:
                    db_ver.is_current_production = False
                try:
                    self.client.set_registered_model_alias(model_name, "staging", str(version))
                except Exception as e:
                    logger.warning(f"Failed to set MLflow staging alias: {e}")

            # Legacy MLflow stage transition (if supported)
            try:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=str(version),
                    stage=target_stage,
                    archive_existing_versions=(target_stage == "Production")
                )
            except Exception:
                pass

            # Update DB version
            db_ver.stage = target_stage
            db_ver.promoted_at = utc_now()

            # Record Deployment Audit Record
            deployment = ModelDeployment(
                model_name=model_name,
                model_version=str(version),
                stage=target_stage,
                deployed_by=deployed_by,
                deployed_at=utc_now(),
                rollback_version=previous_prod_version,
                reason=reason or f"Promoted to {target_stage}",
                is_active=(target_stage == "Production")
            )
            db.add(deployment)
            db.commit()
            db.refresh(db_ver)

            logger.info(f"Successfully promoted '{model_name}' version {version} to stage '{target_stage}'.")
            return {
                "model_name": model_name,
                "version": str(version),
                "new_stage": target_stage,
                "is_current_production": db_ver.is_current_production,
                "previous_production_version": previous_prod_version,
                "validation_status": db_ver.validation_status,
                "promoted_at": db_ver.promoted_at.isoformat(),
                "reason": reason
            }
        finally:
            if close_db:
                db.close()

    def rollback_production(
        self,
        model_name: str,
        target_version: Optional[str] = None,
        reason: Optional[str] = None,
        deployed_by: str = "system",
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Rolls back the current Production model to a previous validated model version.
        If target_version is not specified, rolls back to the most recent previous production version.
        """
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            current_prod = db.query(ModelVersion).filter(
                ModelVersion.model_name == model_name,
                ModelVersion.is_current_production == True
            ).first()

            current_ver_str = current_prod.version if current_prod else None

            # Determine target rollback version
            if target_version is None:
                # Check last deployment rollback_version
                last_dep = db.query(ModelDeployment).filter(
                    ModelDeployment.model_name == model_name,
                    ModelDeployment.stage == "Production",
                    ModelDeployment.rollback_version.isnot(None)
                ).order_by(ModelDeployment.id.desc()).first()

                if last_dep and last_dep.rollback_version:
                    target_version = last_dep.rollback_version
                else:
                    # Find any previously passed version that is not current
                    prev_candidate = db.query(ModelVersion).filter(
                        ModelVersion.model_name == model_name,
                        ModelVersion.version != current_ver_str,
                        ModelVersion.validation_status.in_(["PASSED", "OVERRIDDEN"])
                    ).order_by(ModelVersion.id.desc()).first()

                    if not prev_candidate:
                        raise ValueError(f"Cannot rollback: No previous validated version found for '{model_name}'.")
                    target_version = prev_candidate.version

            # Validate target exists
            target_ver_obj = db.query(ModelVersion).filter(
                ModelVersion.model_name == model_name,
                ModelVersion.version == str(target_version)
            ).first()

            if not target_ver_obj:
                raise ValueError(f"Rollback target version '{target_version}' does not exist.")

            # Perform atomic switch
            if current_prod:
                current_prod.is_current_production = False
                current_prod.stage = "Staging"

            target_ver_obj.is_current_production = True
            target_ver_obj.stage = "Production"
            target_ver_obj.promoted_at = utc_now()

            # Update MLflow Alias
            try:
                self.client.set_registered_model_alias(model_name, "production", str(target_version))
                self.client.set_registered_model_alias(model_name, "champion", str(target_version))
                if current_ver_str:
                    self.client.set_registered_model_alias(model_name, "rolled_back", current_ver_str)
            except Exception as e:
                logger.warning(f"Failed to update MLflow aliases during rollback: {e}")

            # Legacy stage transition
            try:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=str(target_version),
                    stage="Production",
                    archive_existing_versions=True
                )
            except Exception:
                pass

            # Audit record
            deployment = ModelDeployment(
                model_name=model_name,
                model_version=str(target_version),
                stage="Production",
                deployed_by=deployed_by,
                deployed_at=utc_now(),
                rollback_version=current_ver_str,
                reason=reason or f"Emergency rollback from version {current_ver_str} to version {target_version}",
                is_active=True
            )
            db.add(deployment)
            db.commit()

            logger.info(f"Rolled back '{model_name}' production to version {target_version}.")
            return {
                "model_name": model_name,
                "active_production_version": str(target_version),
                "demoted_version": current_ver_str,
                "reason": deployment.reason,
                "rolled_back_at": deployment.deployed_at.isoformat()
            }
        finally:
            if close_db:
                db.close()

    def list_models(self, db: Optional[Session] = None) -> List[Dict[str, Any]]:
        """Lists all registered models with their active production versions."""
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            metas = db.query(ModelMetadata).all()
            result = []
            for m in metas:
                prod_ver = db.query(ModelVersion).filter(
                    ModelVersion.model_name == m.model_name,
                    ModelVersion.is_current_production == True
                ).first()

                vers_count = db.query(ModelVersion).filter(
                    ModelVersion.model_name == m.model_name
                ).count()

                result.append({
                    "model_name": m.model_name,
                    "description": m.description,
                    "problem_type": m.problem_type,
                    "target_column": m.target_column,
                    "current_production_version": prod_ver.version if prod_ver else None,
                    "total_versions": vers_count,
                    "updated_at": m.updated_at.isoformat() if m.updated_at else None
                })
            return result
        finally:
            if close_db:
                db.close()

    def list_versions(self, model_name: str, db: Optional[Session] = None) -> List[Dict[str, Any]]:
        """Lists all versions and metadata for a registered model."""
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            versions = db.query(ModelVersion).filter(
                ModelVersion.model_name == model_name
            ).order_by(ModelVersion.id.desc()).all()

            result = []
            for v in versions:
                result.append({
                    "version": v.version,
                    "model_name": v.model_name,
                    "algorithm": v.algorithm,
                    "stage": v.stage,
                    "is_current_production": v.is_current_production,
                    "validation_status": v.validation_status,
                    "metrics": v.metrics,
                    "mlflow_run_id": v.mlflow_run_id,
                    "artifacts_path": v.artifacts_path,
                    "promoted_at": v.promoted_at.isoformat() if v.promoted_at else None,
                    "created_at": v.created_at.isoformat() if v.created_at else None
                })
            return result
        finally:
            if close_db:
                db.close()

    def get_current_production_version(self, model_name: str, db: Optional[Session] = None) -> Optional[Dict[str, Any]]:
        """Retrieves the active production model version details."""
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            prod_ver = db.query(ModelVersion).filter(
                ModelVersion.model_name == model_name,
                ModelVersion.is_current_production == True
            ).first()

            if not prod_ver:
                return None

            return {
                "model_name": prod_ver.model_name,
                "version": prod_ver.version,
                "algorithm": prod_ver.algorithm,
                "stage": prod_ver.stage,
                "metrics": prod_ver.metrics,
                "mlflow_run_id": prod_ver.mlflow_run_id,
                "artifacts_path": prod_ver.artifacts_path,
                "promoted_at": prod_ver.promoted_at.isoformat() if prod_ver.promoted_at else None
            }
        finally:
            if close_db:
                db.close()

    def get_model_artifacts(self, model_name: str, version: str) -> Dict[str, Any]:
        """Lists logged artifacts from MLflow for a specific model version."""
        try:
            mv = self.client.get_model_version(model_name, str(version))
            artifacts = self.client.list_artifacts(mv.run_id) if mv.run_id else []
            artifact_list = [{"path": a.path, "is_dir": a.is_dir, "file_size": a.file_size} for a in artifacts]
            return {
                "model_name": model_name,
                "version": str(version),
                "run_id": mv.run_id,
                "source": mv.source,
                "artifacts": artifact_list
            }
        except Exception as e:
            logger.warning(f"Could not retrieve MLflow artifacts for {model_name} v{version}: {e}")
            return {
                "model_name": model_name,
                "version": str(version),
                "artifacts": [],
                "error": str(e)
            }

# Module-level singleton
registry_service = ModelRegistryService()
