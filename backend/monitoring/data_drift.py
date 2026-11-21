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

        # 5. Generate Statistical & Evidently AI HTML Report
        reports_dir = Path(settings.ARTIFACTS_DIR) / "monitoring" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        html_report_path = str(reports_dir / f"{report_id}.html")

        # 5a. Always create the high-performance executive HTML report with PSI & KS metrics
        self._generate_custom_html_report(
            html_report_path=html_report_path,
            report_id=report_id,
            timestamp=timestamp,
            overall_status=overall_status,
            drift_score=drift_score,
            drifted_features=drifted_features,
            features=features,
            feature_metrics=feature_metrics,
            sample_size=len(current_df)
        )

        # 5b. Generate Evidently AI 0.7 HTML Report (if requested)
        if generate_evidently_html:
            try:
                os.environ["EVIDENTLY_DISABLE_TELEMETRY"] = "1"
                os.environ["ITERATIVE_DO_NOT_TRACK"] = "1"
                
                # Sample 500 rows for rapid non-blocking evaluation
                sample_n = min(len(reference_df), len(current_df), 500)
                ref_sample = reference_df[features].sample(sample_n, random_state=42)
                curr_sample = current_df[features].sample(sample_n, random_state=42)

                try:
                    # Modern Evidently 0.7+ API
                    from evidently import Report
                    from evidently.presets import DataDriftPreset

                    evidently_report = Report(metrics=[DataDriftPreset()])
                    snapshot = evidently_report.run(reference_data=ref_sample, current_data=curr_sample)
                    snapshot.save_html(html_report_path)
                    logger.info(f"Generated Evidently 0.7 HTML report at {html_report_path}")
                except (ImportError, AttributeError):
                    # Legacy Evidently API fallback
                    from evidently.legacy.report import Report as LegacyReport
                    from evidently.legacy.metric_preset import DataDriftPreset as LegacyDataDriftPreset

                    legacy_report = LegacyReport(metrics=[LegacyDataDriftPreset()])
                    legacy_report.run(reference_data=ref_sample, current_data=curr_sample)
                    legacy_report.save_html(html_report_path)
                    logger.info(f"Generated Legacy Evidently HTML report at {html_report_path}")
            except Exception as e:
                logger.warning(f"Evidently HTML report generation skipped: {e}. Preserved executive report.")

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

    def _generate_custom_html_report(
        self,
        html_report_path: str,
        report_id: str,
        timestamp: datetime,
        overall_status: str,
        drift_score: float,
        drifted_features: set,
        features: list,
        feature_metrics: dict,
        sample_size: int
    ) -> None:
        """Generates an executive standalone HTML report with PSI and KS test statistics."""
        status_color = "#10b981" if overall_status == "Healthy" else ("#f59e0b" if overall_status == "Warning" else "#ef4444")
        status_bg = "#064e3b" if overall_status == "Healthy" else ("#78350f" if overall_status == "Warning" else "#7f1d1d")

        rows_html = ""
        for feat in sorted(features):
            m = feature_metrics.get(feat, {})
            psi = m.get("psi", 0.0)
            psi_stat = m.get("psi_status", "HEALTHY")
            ks_stat = m.get("ks_statistic", 0.0)
            ks_pval = m.get("ks_pvalue", 1.0)
            is_drift = m.get("is_drifted", False)

            row_badge = f'<span style="padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600; background: {"#7f1d1d" if psi_stat == "CRITICAL" else ("#78350f" if psi_stat == "WARNING" else "#064e3b")}; color: {"#fca5a5" if psi_stat == "CRITICAL" else ("#fcd34d" if psi_stat == "WARNING" else "#6ee7b7")}">{psi_stat}</span>'
            decision_badge = f'<span style="padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600; background: {"#7f1d1d" if is_drift else "#064e3b"}; color: {"#fca5a5" if is_drift else "#6ee7b7"}">{"DRIFT DETECTED" if is_drift else "STABLE"}</span>'

            rows_html += f"""
            <tr style="border-bottom: 1px solid #1e293b;">
                <td style="padding: 10px 16px; font-family: monospace; font-size: 13px; color: #f8fafc;">{feat}</td>
                <td style="padding: 10px 16px; font-family: monospace; font-size: 13px; color: #94a3b8;">{psi:.4f}</td>
                <td style="padding: 10px 16px;">{row_badge}</td>
                <td style="padding: 10px 16px; font-family: monospace; font-size: 13px; color: #94a3b8;">{ks_stat:.4f}</td>
                <td style="padding: 10px 16px; font-family: monospace; font-size: 13px; color: #94a3b8;">{ks_pval:.4e}</td>
                <td style="padding: 10px 16px;">{decision_badge}</td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ModelOps Hub - Data Drift Report {report_id}</title>
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; background: #090d16; color: #e2e8f0; margin: 0; padding: 24px; }}
        .card {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 20px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 16px; margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
        .stat-card {{ background: #131d33; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; }}
        .stat-label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
        .stat-value {{ font-size: 24px; font-weight: 700; color: #ffffff; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        th {{ background: #131d33; padding: 12px 16px; font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1 style="font-size: 22px; font-weight: 700; color: #ffffff; margin: 0;">Statistical Data Drift & Distribution Health Report</h1>
            <p style="font-size: 13px; color: #94a3b8; margin: 4px 0 0 0;">Report ID: <code>{report_id}</code> | Evaluated at: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </div>
        <div>
            <span style="display: inline-block; padding: 6px 16px; border-radius: 9999px; font-weight: 700; font-size: 14px; background: {status_bg}; color: {status_color}; border: 1px solid {status_color};">
                STATUS: {overall_status.upper()}
            </span>
        </div>
    </div>

    <div class="card">
        <h2 style="font-size: 16px; font-weight: 600; color: #f8fafc; margin-top: 0;">Executive Summary</h2>
        <div class="grid">
            <div class="stat-card">
                <div class="stat-label">Composite Drift Score</div>
                <div class="stat-value" style="color: {status_color};">{drift_score:.4f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Drifted Features</div>
                <div class="stat-value">{len(drifted_features)} / {len(features)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Production Batch Size</div>
                <div class="stat-value">{sample_size:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Evaluation Engine</div>
                <div class="stat-value" style="font-size: 16px; color: #38bdf8;">PSI + KS-Test + Evidently</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2 style="font-size: 16px; font-weight: 600; color: #f8fafc; margin-top: 0; margin-bottom: 16px;">Feature Distribution Shift Analysis (All 30 Features)</h2>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Population Stability Index (PSI)</th>
                        <th>PSI Health Status</th>
                        <th>KS Statistic</th>
                        <th>KS p-value</th>
                        <th>Distribution Verdict</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>"""
        with open(html_report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

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

