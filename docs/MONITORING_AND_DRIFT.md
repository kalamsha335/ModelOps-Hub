# Statistical Data Drift & Concept Drift Monitoring

ModelOps Hub provides a mathematically rigorous monitoring engine that distinguishes between input data distribution shifts and true model concept degradation.

---

## 1. Data Drift vs. Concept Drift

In tabular machine learning systems, statistical shifts occur across three distinct layers:

```
Input Features P(X)  ──[Data Drift]──>  Model Predictions P(Ŷ)
        │                                      │
   [Concept Drift]                        [Prediction Drift]
        ▼                                      ▼
Ground Truth Labels P(Y|X) <──[Performance Drift]── Model Performance (F1, PR-AUC)
```

1. **Data Drift ($P(X)$)**: Shift in the joint distribution of incoming features $X$. The relationship between $X$ and $Y$ may remain identical.
2. **Prediction Drift ($P(\hat{Y})$)**: Shift in the model's output probability distribution.
3. **Concept Drift ($P(Y|X)$)**: Shift in the fundamental relationship between input features and target labels (e.g., fraudsters changing transaction techniques to mimic legitimate users).
4. **Delayed Feedback Reality**: In credit card fraud detection, true ground-truth chargebacks take **30 to 90 days** to settle through acquiring banks. The system tracks statistical data drift in real-time and awaits delayed labels before declaring true concept degradation.

---

## 2. Statistical Drift Metrics & Formulas

### Population Stability Index (PSI)
Quantifies distribution changes between baseline reference data $B$ and current production data $T$:

$$\text{PSI} = \sum_{k=1}^{K} \left( \% T_k - \% B_k \right) \times \ln\left( \frac{\% T_k}{\% B_k} \right)$$

- **Decile Binning**: Features are partitioned into $K = 10$ quantile bins derived from baseline distributions.
- **Laplace Smoothing**: Epsilon smoothing ($\epsilon = 10^{-4}$) is applied to avoid division by zero when empty bins occur in small production batches.
- **Configurable Operational Thresholds**:
  - $\text{PSI} < 0.20$: **Healthy** (No significant distribution shift).
  - $0.20 \le \text{PSI} < 0.30$: **Warning** (Moderate shift; pipeline inspection recommended).
  - $\text{PSI} \ge 0.30$: **Critical** (Severe distribution shift; candidate for automated retraining).

### Two-Sample Kolmogorov-Smirnov (KS) Test
A non-parametric test comparing cumulative empirical distribution functions $F_1(x)$ and $F_2(x)$:

$$D = \sup_{x} |F_1(x) - F_2(x)|$$

- Null Hypothesis $H_0$: Current and reference data are drawn from identical continuous distributions.
- Significance Level: $\alpha = 0.05$.
- A feature is flagged as drifted if $p\text{-value} < 0.05$.

---

## 3. Evidently AI 0.7 Snapshot Engine

ModelOps Hub integrates **Evidently AI 0.7+** to generate interactive HTML/Plotly diagnostic reports:
- Generates histograms, CDF plots, quantile-quantile curves, and drift summaries for all 30 features (`Time`, `V1`–`V28`, `Amount`).
- Utilizes the modern `Snapshot` architecture (`snapshot.save_html()`).
- Embedded directly into the React console via `/reports/html/{report_id}`.

---

## 4. Delayed Feedback & Performance Degradation

When bank chargebacks settle, administrators or automated ETL streams ingest ground-truth labels via:

```http
POST /api/v1/feedback
Content-Type: application/json

{
  "log_id": "pred_4f89d1b092a1",
  "true_label": 1,
  "notes": "Bank chargeback confirmed by merchant bank"
}
```

The system recalculates historical metrics on labeled production transactions:
- **Precision, Recall, F1-Score**
- **Precision-Recall AUC (PR-AUC)**
- **Receiver Operating Characteristic AUC (ROC-AUC)**
- **False Positive & False Negative Counts**

If production $\text{PR-AUC}$ falls below $0.80$ or $\text{Recall}$ falls below $0.75$, an automated retraining trigger is dispatched with reason `PERFORMANCE_DEGRADATION`.
