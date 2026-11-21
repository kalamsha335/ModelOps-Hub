"""
ModelOps Hub - Data Drift Monitoring Service
Integrates Evidently AI, Population Stability Index (PSI), and Two-Sample KS Tests.
Tracks feature distribution shift, generates HTML/JSON reports, and logs alerts to PostgreSQL.
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.database.session import SessionLocal
from backend.app.models.schemas_db import DriftReport, DriftAlert
from backend.monitoring.psi import calculate_batch_psi
from backend.monitoring.ks_test import calculate_batch_ks

logger = logging.getLogger("modelops.monitoring.data_drift")

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class DataDriftDetector:
    """
    Comprehensive Data Drift detection orchestrator.
    Combines PSI, KS tests, and Evidently AI reports.
    """

    def __init__(self, reference_path: Optional[str] = None):
        self.reference_path = reference_path or str(Path(settings.REFERENCE_DATA_DIR) / "reference_data.parquet")
        self._reference_df: Optional[pd.DataFrame] = None

    def get_reference_data(self) -> pd.DataFrame:
        """Loads and caches reference baseline dataset."""
        if self._reference_df is None:
            if not os.path.exists(self.reference_path):
                raise FileNotFoundError(f"Reference dataset not found at {self.reference_path}")
            self._reference_df = pd.read_parquet(self.reference_path)
            logger.info(f"Loaded reference dataset with {len(self._reference_df)} rows and {len(self._reference_df.columns)} columns.")
        return self._reference_df

    def run_drift_analysis(
        self,
        current_df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
        generate_evidently_html: bool = True,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Executes complete drift analysis comparing reference data against current production data.
        Saves report and alerts into the application database.
        """
        if reference_df is None:
            reference_df = self.get_reference_data()

        # Identify numerical features to evaluate (exclude target column 'Class')
        features = [col for col in reference_df.columns if col in current_df.columns and col != "Class"]

        report_id = f"drift_rep_{uuid.uuid4().hex[:12]}"
        timestamp = utc_now()

        # 1. Run PSI Analysis
        psi_summary = calculate_batch_psi(reference_df, current_df, features=features)

        # 2. Run Kolmogorov-Smirnov Test
        ks_summary = calculate_batch_ks(reference_df, current_df, features=features)

        # 3. Aggregate feature-level metrics
        feature_metrics = {}
        drifted_features = set(ks_summary["drifted_features"])
        drifted_features.update(psi_summary["critical_features"])
        drifted_features.update(psi_summary["warning_features"])

        for feat in features:
            f_psi = psi_summary["feature_psi"].get(feat, {})
            f_ks = ks_summary["feature_ks"].get(feat, {})

            feature_metrics[feat] = {
                "psi": f_psi.get("psi", 0.0),
                "psi_status": f_psi.get("status", "HEALTHY"),
                "ks_statistic": f_ks.get("statistic", 0.0),
                "ks_pvalue": f_ks.get("p_value", 1.0),
                "ks_drifted": f_ks.get("is_drifted", False),
                "is_drifted": (feat in drifted_features)
            }

        # 4. Synthesize Overall Drift Score & Status
        drift_share = len(drifted_features) / len(features) if features else 0.0
        drift_score = round(float(np.clip(psi_summary["max_psi"] + drift_share, 0.0, 1.0)), 4)

        if psi_summary["batch_status"] == "CRITICAL" or drift_share >= settings.DRIFT_SHARE_DRIFTED_THRESHOLD:
            overall_status = "Critical"
        elif psi_summary["batch_status"] == "WARNING" or len(drifted_features) > 0:
            overall_status = "Warning"
        else:
            overall_status = "Healthy"

        # 5. Generate Evidently AI HTML Report (if requested)
        html_report_path = None
        if generate_evidently_html:
            try:
                # Disable telemetry for offline execution
                os.environ["EVIDENTLY_DISABLE_TELEMETRY"] = "1"
                os.environ["ITERATIVE_DO_NOT_TRACK"] = "1"
                from evidently.report import Report
                from evidently.metric_preset import DataDriftPreset

                reports_dir = Path(settings.ARTIFACTS_DIR) / "monitoring" / "reports"
                reports_dir.mkdir(parents=True, exist_ok=True)
                html_report_path = str(reports_dir / f"{report_id}.html")

                # Sample up to 5,000 rows for report generation speed if batch is large
                ref_sample = reference_df[features].sample(min(len(reference_df), 5000), random_state=42)
                curr_sample = current_df[features].sample(min(len(current_df), 5000), random_state=42)

                evidently_report = Report(metrics=[DataDriftPreset()])
                evidently_report.run(reference_data=ref_sample, current_data=curr_sample)
                evidently_report.save_html(html_report_path)
                logger.info(f"Generated Evidently HTML report at {html_report_path}")
            except Exception as e:
                logger.warning(f"Evidently HTML report generation skipped: {e}")
                html_report_path = None

        # 6. Save to Database
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            db_report = DriftReport(
                report_id=report_id,
                status=overall_status,
                drift_score=drift_score,
                drifted_features_count=len(drifted_features),
                total_features_count=len(features),
                feature_metrics=feature_metrics,
                evidently_report_html_path=html_report_path,
                created_at=timestamp
            )
            db.add(db_report)
            db.flush()  # Get db_report.id

            # Create Alert records for drifted features
            created_alerts = []
            for feat in drifted_features:
                metrics_info = feature_metrics[feat]
                is_crit = metrics_info["psi_status"] == "CRITICAL"
                sev = "CRITICAL" if is_crit else "WARNING"

                alert = DriftAlert(
                    drift_report_id=db_report.id,
                    severity=sev,
                    affected_feature=feat,
                    drift_metric="PSI" if is_crit else "KS_TEST",
                    metric_value=metrics_info["psi"] if is_crit else metrics_info["ks_statistic"],
                    threshold_value=settings.DRIFT_PSI_CRITICAL_THRESHOLD if is_crit else settings.DRIFT_KS_PVALUE_THRESHOLD,
                    recommended_action=(
                        "Trigger automated model retraining on recent distribution window"
                        if is_crit
                        else "Monitor feature trends; verify upstream data ingestion pipeline"
                    ),
                    status="ACTIVE",
                    detected_at=timestamp
                )
                db.add(alert)
                created_alerts.append({
                    "feature": feat,
                    "severity": sev,
                    "metric": alert.drift_metric,
                    "metric_value": alert.metric_value
                })

            db.commit()
            db.refresh(db_report)

            logger.info(f"Drift Analysis completed: Report ID {report_id}, Status: {overall_status}, Drift Score: {drift_score}")

            return {
                "report_id": report_id,
                "status": overall_status,
                "drift_score": drift_score,
                "drifted_features_count": len(drifted_features),
                "total_features_count": len(features),
                "drift_share": round(drift_share, 4),
                "mean_psi": psi_summary["mean_psi"],
                "max_psi": psi_summary["max_psi"],
                "drifted_features": list(drifted_features),
                "alerts_created": len(created_alerts),
                "evidently_html_path": html_report_path,
                "feature_metrics": feature_metrics,
                "created_at": timestamp.isoformat()
            }
        finally:
            if close_db:
                db.close()

    def get_latest_report(self, db: Optional[Session] = None) -> Optional[Dict[str, Any]]:
        """Retrieves the most recent drift report from the database."""
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            report = db.query(DriftReport).order_by(DriftReport.id.desc()).first()
            if not report:
                return None

            return {
                "id": report.id,
                "report_id": report.report_id,
                "status": report.status,
                "drift_score": report.drift_score,
                "drifted_features_count": report.drifted_features_count,
                "total_features_count": report.total_features_count,
                "feature_metrics": report.feature_metrics,
                "evidently_report_html_path": report.evidently_report_html_path,
                "created_at": report.created_at.isoformat() if report.created_at else None
            }
        finally:
            if close_db:
                db.close()

    def list_reports(self, limit: int = 20, db: Optional[Session] = None) -> List[Dict[str, Any]]:
        """Lists historical drift monitoring reports."""
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            reports = db.query(DriftReport).order_by(DriftReport.id.desc()).limit(limit).all()
            return [{
                "id": r.id,
                "report_id": r.report_id,
                "status": r.status,
                "drift_score": r.drift_score,
                "drifted_features_count": r.drifted_features_count,
                "total_features_count": r.total_features_count,
                "created_at": r.created_at.isoformat() if r.created_at else None
            } for r in reports]
        finally:
            if close_db:
                db.close()

    def list_active_alerts(self, db: Optional[Session] = None) -> List[Dict[str, Any]]:
        """Retrieves active alerts across all drift runs."""
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            alerts = db.query(DriftAlert).filter(DriftAlert.status == "ACTIVE").order_by(DriftAlert.id.desc()).all()
            return [{
                "id": a.id,
                "severity": a.severity,
                "affected_feature": a.affected_feature,
                "drift_metric": a.drift_metric,
                "metric_value": a.metric_value,
                "threshold_value": a.threshold_value,
                "recommended_action": a.recommended_action,
                "status": a.status,
                "detected_at": a.detected_at.isoformat() if a.detected_at else None
            } for a in alerts]
        finally:
            if close_db:
                db.close()

# Module-level instance
drift_detector = DataDriftDetector()
