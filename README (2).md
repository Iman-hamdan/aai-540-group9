# Machine Learning Operations (AAI-540-02)
# group9 
# Iman Hamdan
# Michael Skirvin
# 🧠 Hard Drive Failure Prediction – Production ML Pipeline

End-to-end MLOps system for predicting rare hard drive failures using SMART telemetry data.

Built with:

- AWS S3
- SageMaker
- XGBoost
- Feature Engineering
- CI/CD
- Real-time inference
- Model monitoring

---

# 🚀 What This Project Does

Predicts hard drive failures **before they happen** using telemetry metrics.

The dataset is extremely imbalanced (~0.1% failures), so this system is optimized for:

- High Recall
- Precision-Recall AUC
- Not Accuracy (misleading for rare events)

---

# 🏗️ Architecture Overview

```
Raw SMART Data (S3)
        ↓
Feature Engineering (Rolling Aggregations)
        ↓
Parquet Dataset (S3)
        ↓
Model Training
   ├── Logistic Regression (Baseline)
   └── XGBoost (Advanced)
        ↓
Model Comparison
        ↓
Model Registry
        ↓
SageMaker Endpoint
        ↓
Monitoring + Alerts
```

---

# 📂 Project Structure

```
.
├── 01_feature_engineering.ipynb
├── 02_baseline_model.ipynb
├── 03_xgboost_model.ipynb
├── 04_model_comparison.ipynb
├── deploy/
│   └── inference.py
├── .github/workflows/
│   └── deploy.yml
├── dashboard/
│   └── streamlit_app.py
└── README.md
```

---

# 🔧 Feature Engineering

- Rolling averages
- Failure label generation
- Removal of serial identifiers
- Clean train/test split
- Stratified sampling

Output:
```
engineered_data_sample.parquet
```

---

# 🤖 Models Built

## 1️⃣ Logistic Regression (Baseline)

- Class-weight balanced
- Standard scaling
- Recall-focused evaluation

---

## 2️⃣ XGBoost (Production Candidate)

- scale_pos_weight for imbalance
- PR-AUC optimization
- Threshold tuning
- Better recall tradeoff

---

# 📊 Model Comparison

| Model | ROC-AUC | PR-AUC | Recall | Precision |
|-------|---------|--------|--------|-----------|
| Logistic Regression | 0.84 | 0.18 | 0.52 | 0.07 |
| XGBoost | 0.93 | 0.41 | 0.78 | 0.19 |

XGBoost selected for deployment.

---

# 📦 Deployment

The selected model is deployed as a **SageMaker real-time endpoint**.

Instance Type:
```
ml.m5.large
```

---

## 🔌 Endpoint Request Example

### Python

```python
import boto3
import json

runtime = boto3.client("sagemaker-runtime")

payload = {
    "instances": [
        {
            "smart_5_raw": 12,
            "smart_9_raw": 14567,
            "smart_187_raw": 0
        }
    ]
}

response = runtime.invoke_endpoint(
    EndpointName="hard-drive-failure-endpoint",
    ContentType="application/json",
    Body=json.dumps(payload)
)

print(response["Body"].read().decode())
```

---

# 🗂 Model Registry

Models are versioned in SageMaker Model Registry:

- Version control
- Approval workflow
- Safe production promotion
- Rollback capability

This prevents accidental deployment of unverified models.

---

# ⚙ CI/CD Pipeline

GitHub Actions pipeline:

- Lint
- Unit tests
- Model packaging
- S3 upload
- Endpoint update

Fully automated deployment.

---

# 📊 Monitoring Strategy

### Data Drift
- Feature distribution shifts
- Missing values

### Model Performance
- PR-AUC tracking
- Recall changes

### System Health
- Latency
- Error rate
- Invocation failures

Alerts configured via CloudWatch.

---

# 💰 AWS Cost Estimate

| Component | Estimated Monthly |
|-----------|-------------------|
| Endpoint (8h/day) | ~$29 |
| Full-time endpoint | ~$86 |
| Storage | <$5 |

Recommendation: Delete endpoint when not in use.

---

# 🎯 Why This Project Matters

- Demonstrates rare-event ML handling
- Shows imbalance-aware modeling
- Production deployment experience
- CI/CD integration
- Monitoring awareness
- Real-world MLOps thinking
