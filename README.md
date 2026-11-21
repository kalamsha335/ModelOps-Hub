# ModelOps Hub
### Automated Model Training, Registry, Drift Monitoring, and Retraining Platform for Tabular ML

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11%2B-0194E2.svg)](https://mlflow.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791.svg)](https://www.postgresql.org/)
[![Evidently AI](https://img.shields.io/badge/Evidently-AI-darkgreen.svg)](https://evidentlyai.com/)
[![Prefect](https://img.shields.io/badge/Prefect-2.16%2B-blueviolet.svg)](https://www.prefect.io/)

---

## 1. Problem Statement & Overview

In financial technology, deploying a fraud detection model is merely day one of the machine learning lifecycle. Over time:
1. **Data Drift ($P(X)$ changes)**: Seasonal spending spikes, shifting consumer demographics, or new payment methods alter input distributions.
2. **Concept Drift ($P(Y|X)$ changes)**: Fraudsters actively evolve adversarial patterns to circumvent existing decision boundaries.
3. **Silent Failure**: A static model degrades silently, resulting in undetected fraudulent transactions and severe financial losses.

**ModelOps Hub** is an end-to-end, enterprise-grade MLOps platform engineered to automate this entire lifecycle for tabular machine learning models using the European Credit Card Fraud Detection dataset:
- Automated multi-model training (Logistic Regression vs. XGBoost) with Optuna hyperparameter optimization.
- MLflow Tracking & Model Registry with strict governance (Development → Staging → Production).
- Continuous statistical drift detection (PSI, Kolmogorov-Smirnov, Evidently AI).
- Concept drift & performance monitoring with delayed ground-truth ingestion.
- Prefect-orchestrated automated retraining with Champion vs. Challenger guardrails.
- High-performance FastAPI prediction serving with dynamic hot-reloading and latency tracking.
- Interactive React + Tailwind CSS dashboard providing observability across every pipeline stage.

---

## 2. Technology Stack

| Layer | Technologies |
|---|---|
| **Backend & Serving** | Python 3.11+, FastAPI, Pydantic v2, Uvicorn, SQLAlchemy |
| **Machine Learning** | Scikit-learn, XGBoost, Pandas, NumPy, Optuna, Joblib, Scipy |
| **Experiment Tracking & Registry** | MLflow Tracking Server & MLflow Model Registry |
| **Monitoring & Drift** | Evidently AI, PSI (Population Stability Index), Kolmogorov-Smirnov Test |
| **Workflow Orchestration** | Prefect 2.16+ |
| **Database** | PostgreSQL 16 (with automated SQLite fallback for local development) |
| **Frontend UI** | React 18, Vite, Tailwind CSS, Lucide Icons, Recharts |
| **Infrastructure & DevOps** | Docker, Docker Compose, Git, REST APIs |

---

## 3. Dataset Setup

The platform uses the **ULB European Credit Card Fraud Detection** dataset (284,807 transactions, 492 frauds, 0.172% fraud rate).

### Download Instructions:
1. Visit the Kaggle dataset page: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`.
3. Place the file inside the project directory:
   ```bash
   data/raw/creditcard.csv
   ```
4. Verify the file exists:
   ```powershell
   Test-Path data/raw/creditcard.csv
   ```

*Note: The platform includes a schema validator and synthetic sample generator for testing environments, but real model training executes against the actual Kaggle dataset.*

---

## 4. Quick Start & Execution

### Option A: Local Development Setup

1. **Activate the Virtual Environment**:
   ```powershell
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Configure Environment Variables**:
   ```bash
   cp .env.example .env
   ```

4. **Run FastAPI Backend**:
   ```bash
   uvicorn backend.app.main:app --reload --port 8000
   ```
   Access API documentation: `http://localhost:8000/docs`

5. **Start Frontend (React + Vite)**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Access Dashboard: `http://localhost:5173`

### Option B: Docker Compose Full Stack

To start PostgreSQL, MLflow, FastAPI backend, Prefect, and the React frontend simultaneously:
```bash
docker-compose up --build
```

---

## 5. Architectural Highlights & Guardrails

- **Data Leakage Prevention**: Preprocessing scalers are fit strictly on the training partition and saved as serialized artifacts bundled with the model.
- **Evaluation Priority**: Due to severe class imbalance (0.172%), models are evaluated and selected primarily by **PR-AUC (Precision-Recall Curve Area)** and **Fraud Recall**, never raw accuracy.
- **Production Guardrails**: A retrained model is rejected (`SKIPPED_INFERIOR`) if its validation PR-AUC is below the threshold or lower than the existing active production champion.
- **Dynamic Hot Reloading**: FastAPI loads the production model directly from MLflow or the designated model artifact store without requiring server restarts.

---

## 6. Directory Structure

```text
ModelOps-Hub/
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints/  # FastAPI REST routes (predict, registry, monitoring, retraining)
│   │   ├── core/              # Configuration & Pydantic settings
│   │   ├── database/          # SQLAlchemy session & fallback SQLite/PostgreSQL
│   │   ├── models/            # SQLAlchemy DB models (9 tables)
│   │   ├── schemas/           # Pydantic request/response schemas
│   │   ├── services/          # Dynamic model serving & hot-reloading
│   │   └── main.py            # FastAPI entry point & health check
│   ├── training/              # Training, tuning & MLflow pipeline (Optuna, XGBoost, LR)
│   ├── monitoring/            # Statistical drift (PSI, KS, Evidently) & Concept drift
│   ├── workflows/             # Prefect orchestration workflows (retraining, monitoring)
│   ├── tests/                 # Automated test suite (26 passing tests)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/        # Navbar, Sidebar, status pills
│   │   ├── pages/             # Overview, Registry, Drift, Concept, Retraining, Playground
│   │   ├── services/          # Axios API client
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   ├── tailwind.config.js
│   ├── vite.config.js
│   └── Dockerfile
├── data/
│   ├── raw/                   # Kaggle creditcard.csv
│   ├── reference/             # 42,721 baseline validation rows (.parquet)
│   └── current/               # Incoming production inference stream
├── artifacts/
│   ├── champion/              # Champion XGBoost model & preprocessor binaries
│   └── monitoring/reports/    # Generated Evidently AI HTML reports
├── docker-compose.yml         # 5-container architecture (Postgres, MLflow, FastAPI, Prefect, React)
├── ARCHITECTURE.md            # Deep-dive system design & data flow diagrams
├── INTERVIEW_PREP.md          # 10-minute presentation outline & 10 Q&A for senior interviews
└── README.md
```

---

## 7. Quickstart Guide

### Option A: Local Development (Fastest)

1. **Activate Python Environment & Install Dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Execute Automated Machine Learning Pipeline**:
   ```bash
   python backend/training/train.py
   ```
   *Trains Logistic Regression & XGBoost, logs to MLflow, and saves the Champion model.*

3. **Start the FastAPI Backend**:
   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   *Interactive API Docs available at `http://localhost:8000/docs`.*

4. **Start the React Frontend Console**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   *Console UI available at `http://localhost:5173`.*

---

### Option B: Docker Compose (Full-Stack 5-Container Stack)

Run all services (PostgreSQL 16, MLflow 2.11, FastAPI, Prefect 2.16, React UI):

```bash
docker-compose up --build
```

- **React Frontend**: `http://localhost:5173`
- **FastAPI Serving & Docs**: `http://localhost:8000/docs`
- **MLflow Tracking & Registry**: `http://localhost:5000`
- **Prefect Orchestrator**: `http://localhost:4200`
- **PostgreSQL Database**: `localhost:5432`

---

## 8. Verified Test Suite Matrix (26/26 Passing)

To execute the entire automated test suite:

```bash
python -m pytest backend/tests/ -v
```

| Test Module | Coverage Scope | Status |
|---|---|---|
| `test_env.py` | Package imports, DB connectivity, schema creation, `/health` | **3/3 PASSED** |
| `test_training.py` | Schema checks, EDA, leakage-free scaling, stratified splits, PR-AUC evaluation | **7/7 PASSED** |
| `test_registry.py` | Model registration, quantitative validation gates, staging/prod promotion, atomic rollback | **2/2 PASSED** |
| `test_monitoring.py` | PSI calculation, KS distribution tests, Evidently reports, concept drift with true labels | **4/4 PASSED** |
| `test_workflows.py` | Concurrency guardrails, Champion vs. Challenger inferior rejection, automated retraining | **3/3 PASSED** |
| `test_api.py` | Serving endpoints (`/predict`, `/predict/batch`, `/feedback`, `/logs`, `/monitoring`, `/registry`) | **7/7 PASSED** |
| **Total** | **End-to-End Enterprise Verification** | **26/26 PASSED (100%)** |

---

## 9. Technical Interview & Mentoring Resources

- **[`ARCHITECTURE.md`](./ARCHITECTURE.md)**: Full architecture specification, Mermaid diagrams, database schemas, and failure recovery modes.
- **[`INTERVIEW_PREP.md`](./INTERVIEW_PREP.md)**: 10-minute slide-by-slide presentation script, architectural justifications, code-level trade-offs, and 10 battle-tested interview questions with answers for 2026 CS graduates.


