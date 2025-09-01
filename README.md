
# Enterprise-Scale Fraud Detection — Databricks + PySpark + LightGBM

A **production-ready, scalable fraud detection pipeline** built on **Databricks** and **PySpark**, leveraging **LightGBM** for predictive modeling, **Optuna** for tuning, and **SHAP** for explainability. Designed for enterprise environments where **audit compliance, accuracy, and scalability** are critical.

---

## 🚀 Highlights
- **Big Data ETL:** Feature engineering and transformation using **PySpark**.
- **High-performance modeling:** **LightGBM** with class imbalance handling.
- **Explainable AI:** Integrated **SHAP plots** for transparent, audit-ready insights.
- **Hyperparameter tuning:** Automated optimization with **Optuna**.
- **Deployment-ready:** FastAPI microservice included for easy serving.

**Performance (Balanced Dataset):**
| Metric | Score |
|--------|-------|
| **AUC** | **0.986** |
| **Accuracy** | **96%** |

---

## 🗺 Architecture

### **Mermaid Flow (renders in GitHub)**
```mermaid
flowchart LR
    A[Raw Transactions CSV] --> B[PySpark ETL & Feature Engineering]
    B --> C[Vectorization]
    C --> D[LightGBM Training]
    D --> E[Model Evaluation (AUC, Accuracy, Confusion Matrix)]
    D --> F[SHAP Explainability]
    D --> G[Model Export (fraud_model.pkl)]
    G --> H[FastAPI Microservice Deployment]

