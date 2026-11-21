# ModelOps Hub - System Architecture & Engineering Design

## 1. High-Level System Architecture

```mermaid
flowchart TD
    subgraph Data_Layer ["1. Data Ingestion & Storage"]
        RawData[("Raw Transaction Data\n(creditcard.csv)")]
        ValData["Data Validator\n(Pandas & Schema Checks)"]
        SplitData["Stratified Train/Val/Test Split\n& Feature Preprocessor"]
        RefStore[("Reference Dataset Store")]
        CurrStore[("Production Inbound Data Store")]
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
        ModelReg["MLflow Model Registry\n(Logical Stages: Dev -> Staging -> Prod)"]
        Guardrail["Production Promotion Guardrail\n(PR-AUC >= 0.80, Recall >= 0.82)"]
        Evaluator --> MLflowTrack --> ModelReg
        ModelReg --> Guardrail
    end

    subgraph Database_Layer ["4. PostgreSQL Application State"]
        PostgresDB[("PostgreSQL Database\n- model_metadata\n- model_versions\n- prediction_logs\n- drift_reports\n- drift_alerts\n- retraining_jobs")]
    end

    subgraph Serving_Layer ["5. FastAPI Model Serving Layer"]
        FastAPIApp["FastAPI REST Server (/api/v1)"]
        ModelLoader["Dynamic Production Model Loader\n(Hot Reloading from Registry)"]
        InferenceEngine["Preprocessed Inference Engine"]
        Logger["Prediction Logger & Latency Monitor"]

        Guardrail --> ModelLoader --> InferenceEngine
        FastAPIApp --> InferenceEngine --> Logger
        Logger --> CurrStore
        Logger --> PostgresDB
    end

    subgraph Monitoring_Layer ["6. Drift & Degradation Monitoring"]
        EvidentlyEngine["Evidently AI Drift Engine"]
        StatEngine["Statistical Drift Detector\n(PSI & Kolmogorov-Smirnov Tests)"]
        PerfEngine["Concept Drift / Performance Degradation\n(Ground Truth vs Predictions)"]
        AlertEngine["Drift Alert Manager\n(Severity: Warning / Critical)"]

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
        RetrainTrigger{"Trigger Decision Engine\n- Critical PSI > 0.30\n- Perf Degradation\n- Scheduled / Manual"}
        RetrainExec["Retraining Pipeline Execution"]
        SafetyCheck{"Candidate vs Prod Validation\nIs Candidate >= Production?"}

        AlertEngine --> RetrainTrigger
        PrefectScheduler --> RetrainTrigger
        RetrainTrigger -->|Yes| RetrainExec
        RetrainExec --> SafetyCheck
        SafetyCheck -->|Passed| ModelReg
        SafetyCheck -->|Rejected (Inferior)| PostgresDB
    end

    subgraph Presentation_Layer ["8. ModelOps Console (React + Vite + Tailwind)"]
        OverviewDash["Overview Dashboard"]
        RegistryUI["Model Registry & Stage Manager"]
        TrainUI["Training & Tuning Analytics"]
        DriftUI["Evidently & Distribution Inspector"]
        AlertUI["Drift Alerts & Action Center"]
        RetrainUI["Retraining Workflow Control"]
        PlaygroundUI["Live Prediction Playground"]

        OverviewDash & RegistryUI & TrainUI & DriftUI & AlertUI & RetrainUI & PlaygroundUI <--> FastAPIApp
    end
```

---

## 2. Component Breakdown

### A. Data Layer
- **Ingestion & Validation**: Reads the ULB European Credit Card Fraud Detection dataset (284,807 transactions, 492 fraud cases, 0.172% positive class).
- **Leakage Prevention**: Standardizes `Amount` and `Time` using `RobustScaler` fit strictly on the training partition; PCA features `V1-V28` are preserved.
- **Data Partitioning**: Stratified 70/15/15 split into Train, Validation, and Test sets. Reference baseline data is persisted for drift detection.

### B. Machine Learning & Training Layer
- **Models**:
  1. Regularized Logistic Regression with class-weight balancing (`class_weight="balanced"`).
  2. XGBoost Classifier with `scale_pos_weight` tuned dynamically based on negative-to-positive ratio.
- **Optimization**: Optuna Bayesian hyperparameter search using 5-Fold Stratified Cross-Validation maximizing PR-AUC (Precision-Recall Area Under Curve).
- **Metrics Selection Priority**:
  1. **PR-AUC (Primary)**: Focuses directly on minority fraud detection performance without being inflated by overwhelming true negatives.
  2. **Recall (Fraud Detection Rate)**: Minimizes false negatives (undetected fraud).
  3. **Precision**: Controls false alerts that burden human fraud analysts.
  4. **ROC-AUC & F1-Score**: Secondary broad discriminative metrics.

### C. MLflow Model Registry & Stage Promotion
- **Storage**: PostgreSQL backend store with persistent local/cloud artifact storage.
- **Lifecycle Management**:
  - `Development`: Raw tuned models from experimental runs.
  - `Staging`: Candidate models passing automated unit and validation checks.
  - `Production`: Validated models active for live serving.
- **Rollback**: Instant atomic pointer swap back to previous production version if anomalies occur.

### D. Data Drift & Statistical Testing
- **Evidently AI**: Automated DataDriftPreset reports comparing baseline reference distributions to current live production transactions.
- **Population Stability Index (PSI)**:
  - $\text{PSI} = \sum \left( \text{Actual}\% - \text{Expected}\% \right) \times \ln\left( \frac{\text{Actual}\%}{\text{Expected}\%} \right)$
  - Thresholds: $< 0.10$ Stable, $0.10 - 0.20$ Moderate Drift (Warning), $> 0.20$ Significant Drift (Critical).
- **Kolmogorov-Smirnov (KS) Test**: Two-sample non-parametric test comparing cumulative distribution functions ($p\text{-value} < 0.05$ indicates distribution shift).

### E. Concept Drift vs. Data Drift
- **Data Drift**: $P(X)$ changes while $P(Y|X)$ remains constant (e.g., changes in transaction amounts or user spending patterns).
- **Concept Drift**: $P(Y|X)$ changes (e.g., fraudsters devise novel evasion mechanisms such that a legitimate-looking pattern is actually fraud).
- When ground-truth labels arrive with latency, performance metrics are recalculated; degradation triggers retrain workflows.

### F. Automated Retraining Guardrails
1. **Concurrency Lock**: Prevents duplicate overlapping retraining runs.
2. **Cold-down Interval**: Minimum 24 hours between automated retraining runs unless manually overridden.
3. **Challenger vs. Champion Validation**: Candidate model must outperform or meet the production model's PR-AUC and Recall thresholds before promotion. Inferior models are logged as `REJECTED_INFERIOR`.
