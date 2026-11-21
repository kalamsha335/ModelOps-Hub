# ModelOps Hub - System Architecture & Engineering Design

---

## 1. High-Level System Architecture

```mermaid
flowchart TD
    subgraph Data_Layer ["1. Data Ingestion & Storage"]
        RawData[("Raw Transaction Data\n(creditcard.csv)")]
        ValData["Data Validator\n(Pandas & Schema Checks)"]
        SplitData["Stratified Train/Val/Test Split\n& Feature Preprocessor"]
        RefStore[("Reference Dataset Store\n(reference_data.parquet)")]
        CurrStore[("Production Inbound Stream\n(prediction_logs)")]
        RawData --> ValData --> SplitData
        SplitData --> RefStore
    end

    subgraph Training_Pipeline ["2. Automated ML Training & Optimization"]
        OptunaTune["Optuna Hyperparameter Tuning\n(Stratified 5-Fold CV)"]
        ModelTrain["Model Training\n(Logistic Regression vs XGBoost)"]
        Evaluator["Comprehensive Model Evaluator\n(PR-AUC, Recall, ROC-AUC, F1)"]
        SplitData --> OptunaTune --> ModelTrain --> Evaluator
    end

    subgraph Registry_Layer ["3. MLflow Tracking & Model Registry"]
        MLflowTrack["MLflow Experiment Tracker\n(Parameters, Metrics, Artifacts)"]
        ModelReg["MLflow Model Registry\n(Aliases: @champion, @production, @staging)"]
        Guardrail["Production Promotion Guardrail\n(PR-AUC >= 0.80, Recall >= 0.75)"]
        Evaluator --> MLflowTrack --> ModelReg
        ModelReg --> Guardrail
    end

    subgraph Database_Layer ["4. Application Database (PostgreSQL / SQLite)"]
        PostgresDB[("Metadata & State Store\n- models\n- model_versions\n- model_deployments\n- prediction_logs\n- drift_reports\n- drift_alerts\n- retraining_jobs\n- training_runs")]
    end

    subgraph Serving_Layer ["5. FastAPI Model Serving Layer"]
        FastAPIApp["FastAPI REST Server (/api/v1)"]
        ModelLoader["Dynamic Production Model Loader\n(In-Memory Hot Reloading)"]
        InferenceEngine["Preprocessed Inference Engine"]
        Logger["Prediction Audit Logger\n(Sub-5ms Latency Tracking)"]

        Guardrail --> ModelLoader --> InferenceEngine
        FastAPIApp --> InferenceEngine --> Logger
        Logger --> CurrStore
        Logger --> PostgresDB
    end

    subgraph Monitoring_Layer ["6. Statistical & Concept Drift Monitoring"]
        EvidentlyEngine["Evidently AI 0.7 Snapshot Engine\n(Interactive Plotly Dashboards)"]
        StatEngine["Statistical Drift Detector\n(Decile PSI & Two-Sample KS Tests)"]
        PerfEngine["Concept Drift / Performance Degradation\n(Delayed Ground Truth vs Predictions)"]
        AlertEngine["Drift Alert Manager\n(Severity: WARNING / CRITICAL)"]

        RefStore --> EvidentlyEngine
        CurrStore --> EvidentlyEngine
        RefStore --> StatEngine
        CurrStore --> StatEngine
        CurrStore --> PerfEngine
        EvidentlyEngine --> AlertEngine
        StatEngine --> AlertEngine
        PerfEngine --> AlertEngine
        AlertEngine --> PostgresDB
    end

    subgraph Automation_Layer ["7. Prefect Orchestrator & Retraining Loop"]
        PrefectScheduler["Prefect Workflow Orchestration"]
        RetrainTrigger{"Trigger Decision Engine\n- Critical PSI >= 0.30\n- Recall < 0.75\n- Scheduled / Manual"}
        RetrainExec["Retraining Pipeline Execution"]
        SafetyCheck{"Anti-Regression Guardrail\nComposite: 0.6*PR-AUC + 0.4*Rec\nCandidate > Production?"}

        AlertEngine --> RetrainTrigger
        PrefectScheduler --> RetrainTrigger
        RetrainTrigger -->|Trigger Fire| RetrainExec
        RetrainExec --> SafetyCheck
        SafetyCheck -->|Passed| ModelReg
        SafetyCheck -->|Rejected (Inferior)| PostgresDB
    end

    subgraph Presentation_Layer ["8. ModelOps Console (React 18 + Vite + Tailwind)"]
        OverviewDash["Overview Dashboard"]
        RegistryUI["Model Registry & Stage Governance"]
        DriftUI["Data Drift & PSI Inspector"]
        ConceptUI["Concept Drift & Delayed Feedback"]
        AlertUI["Active Alerts Center"]
        RetrainUI["Retraining Workflow Control"]
        PlaygroundUI["Live Prediction Playground"]

        OverviewDash & RegistryUI & DriftUI & ConceptUI & AlertUI & RetrainUI & PlaygroundUI <--> FastAPIApp
    end
```

---

## 2. Sequence Diagrams

### A. Real-Time Inference & Delayed Feedback Loop

```mermaid
sequenceDiagram
    autonumber
    actor Client as Client / Payment Gateway
    participant API as FastAPI Serving Layer
    participant Serving as In-Memory Serving Service
    participant DB as Relational Database
    actor Analyst as Fraud Analyst / Bank

    Client->>API: POST /api/v1/predict (Transaction Features)
    API->>Serving: Score with @production Model
    Serving-->>API: Fraud Probability & Class (1.8ms)
    API->>DB: Log Transaction to prediction_logs
    API-->>Client: PredictionResponse (is_fraud, probability, log_id)

    Note over Client,Analyst: 30 to 90 Days Settlement Window (Chargebacks)

    Analyst->>API: POST /api/v1/feedback (log_id, true_label=1)
    API->>DB: Update ground_truth in prediction_logs
    API->>API: Recompute Confusion Matrix & PR-AUC
    opt If Production Recall < 0.75
        API->>DB: Log PERFORMANCE_DEGRADATION Alert
    end
    API-->>Analyst: Feedback Accepted
```

### B. Automated Retraining with Anti-Regression Guardrail

```mermaid
sequenceDiagram
    autonumber
    participant Mon as Monitoring Engine
    participant Prefect as Prefect Workflow Flow
    participant Train as Optuna Training Pipeline
    participant Guard as Anti-Regression Guardrail
    participant MLflow as MLflow Registry
    participant Serving as FastAPI Serving Process

    Mon->>Prefect: Event Trigger: CRITICAL_FEATURE_DRIFT (PSI >= 0.30)
    Prefect->>Prefect: Check Duplicate Retraining Locks
    Prefect->>Train: Execute Stratified 5-Fold HPO on New Ingestion Window
    Train->>MLflow: Log Candidate Parameters, Metrics, Artifacts
    Train-->>Prefect: Return Challenger Model & Holdout Metrics
    Prefect->>Guard: Evaluate Champion vs. Challenger
    
    alt Challenger Score > Champion Score
        Guard->>MLflow: Assign @production Alias to Challenger
        Guard->>Serving: Signal In-Memory Hot Reload
        Serving->>Serving: Atomic Model Pointer Swap (Zero Downtime)
        Guard-->>Prefect: Retraining Status: SUCCESS_PROMOTED
    else Challenger Score <= Champion Score
        Guard->>Guard: Reject Challenger (Prevent Performance Regression)
        Guard-->>Prefect: Retraining Status: SKIPPED_INFERIOR
    end
```

---

## 3. Database Entity-Relationship (ER) Model

```mermaid
erDiagram
    MODELS ||--o{ MODEL_VERSIONS : contains
    MODEL_VERSIONS ||--o{ MODEL_DEPLOYMENTS : deployed_as
    DRIFT_REPORTS ||--o{ DRIFT_ALERTS : generates
    MODEL_VERSIONS ||--o{ RETRAINING_JOBS : produced_by

    MODELS {
        int id PK
        string name UK
        string description
        datetime created_at
    }

    MODEL_VERSIONS {
        int id PK
        int model_id FK
        string version
        string stage
        string mlflow_run_id
        string algorithm
        json metrics
        json parameters
        string artifact_path
        datetime created_at
    }

    MODEL_DEPLOYMENTS {
        int id PK
        string model_name
        string model_version
        string stage
        string deployed_by
        string status
        text deployment_notes
        datetime deployed_at
    }

    PREDICTION_LOGS {
        int id PK
        string log_id UK
        string model_version
        json input_features
        int prediction
        float probability
        int ground_truth
        float latency_ms
        datetime created_at
    }

    DRIFT_REPORTS {
        int id PK
        string report_id UK
        string status
        float drift_score
        int drifted_features_count
        int total_features_count
        json feature_metrics
        string evidently_report_html_path
        datetime created_at
    }

    DRIFT_ALERTS {
        int id PK
        int drift_report_id FK
        string severity
        string affected_feature
        string drift_metric
        float metric_value
        float threshold_value
        string recommended_action
        string status
        datetime detected_at
    }

    RETRAINING_JOBS {
        int id PK
        string retraining_id UK
        string trigger_reason
        string status
        string previous_model_version
        string new_model_version
        json champion_metrics
        json challenger_metrics
        text decision_rationale
        datetime started_at
        datetime completed_at
    }

    TRAINING_RUNS {
        int id PK
        string run_id UK
        string experiment_name
        string model_type
        json parameters
        json metrics
        json confusion_matrix
        float training_duration_seconds
        datetime created_at
    }

    PERFORMANCE_METRICS {
        int id PK
        string model_version
        int evaluation_window_size
        float precision
        float recall
        float f1_score
        float pr_auc
        float roc_auc
        int false_positives
        int false_negatives
        datetime calculated_at
    }
```

---

## 4. Failure Recovery & High-Availability Modes

1. **Database Fallback**: Dual-engine SQLAlchemy driver tries PostgreSQL 15 on port 5432. If PostgreSQL is offline or unreachable, it smoothly falls back to SQLite (`modelops.db`), allowing 100% functionality without requiring external infrastructure.
2. **Offline Evidently Rendering**: If network access or fonts are unavailable, the monitoring service generates an executive HTML dashboard first, and overlays Evidently AI 0.7 snapshot data without blocking the API request thread.
3. **Hot Reload Memory Isolation**: Prediction serving loads model weights into an isolated memory slot. If model reloading fails during rollback or promotion, the previous valid model remains active without serving degradation.
