# Automated Retraining & Anti-Regression Guardrails

ModelOps Hub orchestrates retraining workflows using **Prefect 2**, combining event-driven triggers with rigorous Champion vs. Challenger guardrails to prevent model degradation in production.

---

## 1. Retraining Triggers

Retraining can be initiated through four operational channels:

```
┌────────────────────────────────────────────────────────┐
│                   RETRAINING TRIGGERS                  │
├────────────────────┬───────────────────────────────────┤
│ CRITICAL_DRIFT     │ Feature PSI ≥ 0.30 or share ≥ 25% │
│ DEGRADATION        │ True label Recall < 0.75          │
│ SCHEDULED          │ Periodic Prefect cron schedule    │
│ MANUAL             │ Operator ad-hoc trigger via UI/API│
└────────────────────┴───────────────────────────────────┘
```

---

## 2. Anti-Regression Guardrail (Champion vs. Challenger)

To guarantee that an automated retraining job never deploys an inferior model into production, ModelOps Hub enforces an automated mathematical guardrail:

```
                  ┌───────────────────────────────┐
                  │    Retraining Job Completes   │
                  │ (Trains Challenger on New Data│
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │  Evaluate Champion vs.        │
                  │  Challenger on Holdout Test   │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ Calculate Composite Scores:   │
                  │ S = 0.60 * PR-AUC + 0.40 * Rec│
                  └───────────────┬───────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   │
       S_Challenger > S_Champion           S_Challenger ≤ S_Champion
                │                                   │
                ▼                                   ▼
    ┌───────────────────────┐           ┌───────────────────────┐
    │  PROMOTED TO PROD     │           │   SKIPPED_INFERIOR    │
    │  - MLflow updated     │           │  - Retain Champion    │
    │  - DB audit logged    │           │  - Audit log rejection│
    │  - Serving hot-reload │           │  - Zero downtime      │
    └───────────────────────┘           └───────────────────────┘
```

### Composite Score Formula
Fraud detection requires a balanced trade-off between identifying fraudulent transactions (Recall) and minimizing customer friction (PR-AUC / Precision):

$$\text{Composite Score } S = 0.60 \times \text{PR-AUC} + 0.40 \times \text{Recall}$$

### Duplicate Job Safeguard
If an active retraining workflow is already running, subsequent requests are automatically throttled to prevent redundant resource consumption.

---

## 3. Retraining Execution Record Schema

Every retraining execution records full lineage in the `retraining_jobs` database table:

| Field | Type | Description |
| :--- | :--- | :--- |
| `retraining_id` | String | e.g. `retrain_a1b2c3d4e5f6` |
| `trigger_reason` | String | Trigger cause |
| `status` | String | `SUCCESS_PROMOTED`, `SKIPPED_INFERIOR`, `FAILED` |
| `previous_model_version` | String | Active production version at run start |
| `new_model_version` | String | Candidate challenger version |
| `champion_metrics` | JSON | Baseline metrics at evaluation |
| `challenger_metrics` | JSON | Retrained candidate metrics |
| `decision_rationale` | Text | Mathematical justification for promotion or rejection |
| `started_at` / `completed_at` | DateTime | Execution duration tracking |
