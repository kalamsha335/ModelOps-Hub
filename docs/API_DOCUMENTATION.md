# ModelOps Hub - REST API Specification

Comprehensive documentation for all REST API endpoints provided by the ModelOps Hub FastAPI serving layer.

- **Base URL (Local)**: `http://localhost:8000`
- **Interactive OpenAPI Docs**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **API Prefix**: `/api/v1`

---

## 1. System Health & Metadata

### `GET /health`
Verifies backend API responsiveness and database connectivity.

- **Status Code**: `200 OK`
- **Response**:
```json
{
  "status": "healthy",
  "service": "ModelOps Hub",
  "environment": "development",
  "database": "connected",
  "timestamp": "2026-11-21T18:12:00.000Z"
}
```

### `GET /api/v1/model/info`
Retrieves metadata, training run ID, algorithm, validation metrics, and loaded timestamp of the active in-memory production model.

- **Status Code**: `200 OK`
- **Response**:
```json
{
  "status": "SUCCESS",
  "model_name": "fraud-detection-model",
  "version": "1",
  "algorithm": "XGBClassifier",
  "stage": "Production",
  "metrics": {
    "pr_auc": 0.8245,
    "recall": 0.7973,
    "precision": 0.8676,
    "f1_score": 0.8310,
    "roc_auc": 0.9782
  },
  "mlflow_run_id": "bd3ca06697034d5583012807d86bdef7",
  "loaded_at": "2026-11-21T16:54:12.123456"
}
```

---

## 2. Real-Time Inference & Feedback

### `POST /api/v1/predict`
Validates an individual transaction payload, runs preprocessing pipeline, scores via the production model, and logs prediction to the audit database.

- **Request Body**:
```json
{
  "Time": 406.0,
  "V1": -2.312226,
  "V2": 1.951992,
  "V3": -1.609851,
  "V4": 3.997906,
  "V5": -0.522188,
  "V6": -1.426545,
  "V7": -2.537387,
  "V8": 1.391657,
  "V9": -2.770089,
  "V10": -2.772272,
  "V11": 3.202033,
  "V12": -2.899907,
  "V13": -0.595222,
  "V14": -4.289254,
  "V15": 0.389724,
  "V16": -1.140747,
  "V17": -2.830056,
  "V18": -0.016822,
  "V19": 0.416956,
  "V20": 0.126911,
  "V21": 0.517232,
  "V22": -0.035049,
  "V23": -0.465211,
  "V24": 0.320198,
  "V25": 0.044519,
  "V26": 0.177840,
  "V27": 0.261145,
  "V28": -0.143276,
  "Amount": 0.00
}
```
- **Response** (`200 OK`):
```json
{
  "log_id": "pred_4f89d1b092a1",
  "model_version": "1",
  "is_fraud": 1,
  "fraud_probability": 0.8942,
  "latency_ms": 2.41,
  "timestamp": "2026-11-21T18:15:20.102Z"
}
```

### `POST /api/v1/predict/batch`
High-throughput inference accepting a list of transactions (up to 1,000 items per request).

- **Request Body**:
```json
{
  "transactions": [
    { "Time": 0.0, "V1": -1.3598, "...": "...", "Amount": 149.62 },
    { "Time": 406.0, "V1": -2.3122, "...": "...", "Amount": 0.00 }
  ]
}
```
- **Response** (`200 OK`):
```json
{
  "total_records": 2,
  "predictions": [
    { "log_id": "pred_a1...", "is_fraud": 0, "fraud_probability": 0.0003, "latency_ms": 1.1 },
    { "log_id": "pred_b2...", "is_fraud": 1, "fraud_probability": 0.8942, "latency_ms": 1.1 }
  ],
  "batch_latency_ms": 3.8
}
```

### `POST /api/v1/feedback`
Ingests delayed ground-truth feedback (e.g., bank chargebacks 30–90 days later) associated with a previous prediction `log_id`.

- **Request Body**:
```json
{
  "log_id": "pred_4f89d1b092a1",
  "true_label": 1,
  "notes": "Confirmed bank chargeback"
}
```
- **Response** (`200 OK`):
```json
{
  "status": "SUCCESS",
  "log_id": "pred_4f89d1b092a1",
  "true_label": 1,
  "updated_at": "2026-11-21T18:16:00.000Z"
}
```

### `GET /api/v1/logs`
Returns paginated prediction inference audit logs with features, predictions, latency, and true labels.

- **Query Parameters**:
  - `limit` (default: 50, max: 200)
  - `offset` (default: 0)
- **Response** (`200 OK`):
```json
{
  "total_logs": 1420,
  "limit": 50,
  "offset": 0,
  "items": [
    {
      "log_id": "pred_4f89d1b092a1",
      "model_version": "1",
      "prediction": 1,
      "probability": 0.8942,
      "ground_truth": 1,
      "latency_ms": 2.41,
      "timestamp": "2026-11-21T18:15:20.102Z"
    }
  ]
}
```

---

## 3. Model Registry & Stage Governance

### `GET /api/v1/registry/models`
Lists all registered ML models in MLflow and application database.

- **Response** (`200 OK`): List of registered model objects.

### `GET /api/v1/registry/models/{model_name}/versions`
Returns version history, stages, parameters, and evaluation metrics for a specific model.

- **Response** (`200 OK`): List of version dictionaries with metrics, stage aliases, and creation dates.

### `POST /api/v1/registry/promote`
Promotes a model version across lifecycle stages (`Development` &rarr; `Staging` &rarr; `Production`) with validation gate enforcement.

- **Request Body**:
```json
{
  "model_name": "fraud-detection-model",
  "version": "2",
  "target_stage": "Production",
  "reason": "Promoting challenger after passing cross-validation gates"
}
```
- **Validation Gate Rules**:
  - Promotion to `Production` strictly requires $\text{PR-AUC} \ge 0.80$ and $\text{Recall} \ge 0.75$.
  - Attempting to promote an inferior model raises `400 Bad Request` with `ModelValidationError`.

### `POST /api/v1/registry/rollback`
Performs an atomic zero-downtime rollback to a previous production model version, logs audit event, and hot-reloads serving memory.

- **Request Body**:
```json
{
  "model_name": "fraud-detection-model",
  "target_version": "1",
  "reason": "Rollback due to upstream data schema change"
}
```
- **Response** (`200 OK`):
```json
{
  "status": "SUCCESS",
  "message": "Model 'fraud-detection-model' rolled back to version 1",
  "active_production_version": "1"
}
```

---

## 4. Statistical Drift Monitoring & Health

### `GET /api/v1/monitoring/status`
Returns high-level system health, active alert counts, latest drift score, and status.

- **Response** (`200 OK`):
```json
{
  "overall_health": "Healthy",
  "latest_drift_status": "Healthy",
  "latest_drift_score": 0.0825,
  "active_alerts_total": 0,
  "critical_alerts_count": 0,
  "latest_report_id": "drift_rep_bf0b451f7dd3",
  "last_checked_at": "2026-11-21T18:13:22.000Z"
}
```

### `GET /api/v1/monitoring/drift`
Returns detailed statistical drift metrics for all 30 features from the most recent run.

- **Response** (`200 OK`): Contains PSI scores, KS statistics, p-values, and HTML report path.

### `POST /api/v1/monitoring/analyze`
Triggers an immediate ad-hoc statistical drift evaluation comparing current production transactions against baseline reference data.

- **Request Body**:
```json
{
  "sample_size": 1000,
  "generate_evidently_html": true
}
```
- **Response** (`200 OK`):
```json
{
  "report_id": "drift_rep_bf0b451f7dd3",
  "status": "Healthy",
  "drift_score": 0.0825,
  "drifted_features_count": 0,
  "total_features_count": 30,
  "drift_share": 0.0,
  "mean_psi": 0.0121,
  "max_psi": 0.0418,
  "evidently_html_path": "artifacts/monitoring/reports/drift_rep_bf0b451f7dd3.html"
}
```

### `GET /api/v1/monitoring/reports`
Lists historical drift evaluation records.

### `GET /api/v1/monitoring/alerts`
Lists active drift alerts with severity (`WARNING` or `CRITICAL`) and recommended remediation actions.

### `POST /api/v1/monitoring/alerts/{alert_id}/resolve`
Acknowledges and resolves an active drift alert.

### `GET /api/v1/monitoring/performance`
Retrieves concept drift and performance tracking history calculated when true labels are ingested.

---

## 5. Automated Retraining Workflows

### `POST /api/v1/retraining/trigger`
Dispatches automated retraining with anti-regression guardrails.

- **Request Body**:
```json
{
  "trigger_reason": "CRITICAL_FEATURE_DRIFT",
  "force": false,
  "quick_run": false,
  "optuna_trials": 15
}
```
- **Anti-Regression Logic**:
  - Compares retrained candidate composite score against current production model ($0.6 \times \text{PR-AUC} + 0.4 \times \text{Recall}$).
  - If candidate is inferior, retraining is marked `SKIPPED_INFERIOR` and production model remains untouched.

### `GET /api/v1/retraining/history`
Returns execution records, trigger causes, candidate vs champion metrics, and decisions.

### `GET /api/v1/training/history`
Lists all historical MLflow training runs.

---

## 6. HTML Reports

### `GET /reports/html/{report_id}`
Renders generated Evidently AI HTML report for inline viewing in dashboard iframes.
- **Response**: `text/html`
