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
│   │   ├── api/v1/endpoints/  # FastAPI REST routes
│   │   ├── core/              # Configuration & settings
│   │   ├── database/          # SQLAlchemy session & engine
│   │   ├── models/            # SQLAlchemy DB models
│   │   ├── schemas/           # Pydantic validation schemas
│   │   ├── services/          # Business logic services
│   │   └── main.py            # FastAPI entry point
│   ├── training/              # Training, tuning & MLflow pipeline
│   ├── monitoring/            # Drift (PSI, KS, Evidently) engines
│   ├── workflows/             # Prefect orchestration workflows
│   ├── tests/                 # Pytest automated test suites
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   ├── pages/             # Dashboard views
│   │   ├── layouts/           # Sidebar & top navigation
│   │   ├── services/          # Axios/fetch API clients
│   │   └── App.jsx
│   ├── package.json
│   └── Dockerfile
├── data/
│   ├── raw/                   # Kaggle creditcard.csv
│   ├── processed/             # Cleaned & scaled sets
│   ├── reference/             # Baseline training distribution
│   └── current/               # Production inference batches
├── mlruns/                    # MLflow local tracking
├── artifacts/                 # Model & scaler binaries
├── docker-compose.yml
├── ARCHITECTURE.md
└── README.md
```

