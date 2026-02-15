# AAI-540 Group 9 — Hard Drive Failure Prediction

End-to-end MLOps pipeline for predicting hard drive failures using Backblaze telemetry data and Amazon product review sentiment. The project spans data ingestion, feature engineering, model training and deployment, and automated production monitoring with retraining.

## Pipeline Overview

Run the notebooks in order. Each notebook reads configuration from a shared `.env` file and writes back any new resource identifiers (bucket names, Feature Group names, endpoint names, S3 URIs, etc.) so downstream notebooks can pick them up automatically.

```
load_data.ipynb → feature_engineering.ipynb → base_model.ipynb → model_monitoring_cicd.ipynb
```

---

## 1. Data Ingestion — `load_data.ipynb`

Sets up the AWS infrastructure and loads raw data into S3.

| Step | What it does |
|------|-------------|
| **Initialize S3 Bucket** | Creates (or reuses) an S3 bucket and persists the name in `.env` |
| **Create Folder Structure** | Builds a standard MLOps directory layout in S3 (`raw/`, `curated/`, `features/`, `artifacts/`, `monitoring/`, `inference/`) |
| **Download Backblaze Data** | Scrapes the Backblaze data-release page for ZIP URLs, streams each ZIP into `raw/backblaze/zips/` |
| **Convert CSV → Parquet** | Extracts CSVs from each ZIP, converts to Snappy-compressed Parquet with Hive-style date partitioning under `curated/` |
| **Load Amazon Reviews** | Reads Amazon product-review Parquet files into `raw/reviews/` |
| **Join-Key Analysis** | Inspects both datasets, extracts manufacturer and model information, and documents the join strategy for the next notebook |

**Outputs:** Raw ZIPs and curated Parquet files in S3; `BUCKET_NAME` saved to `.env`.

---

## 2. Feature Engineering — `feature_engineering.ipynb`

Transforms raw data into model-ready features and stores them in SageMaker Feature Store.

| Step | What it does |
|------|-------------|
| **Read Backblaze Parquet** | Loads partitioned Parquet files from S3 using PyArrow |
| **Read & Process Reviews** | Loads Amazon review Parquet files; extracts manufacturer/model hints for join keys |
| **Compute Review Sentiment** | Calculates per-manufacturer percentage of 1-star and 2-star reviews (`pct_one_star`, `pct_two_star`) |
| **Join Datasets** | Merges review-sentiment features with Backblaze telemetry on manufacturer |
| **Select Feature Set** | Keeps 7 features: `pct_one_star`, `pct_two_star`, `smart_5_raw`, `smart_187_raw`, `smart_188_raw`, `smart_197_raw`, `smart_198_raw`, plus the `failure` target |
| **Data Quality & Cleaning** | Drops records missing review features, fills SMART NaNs with 0, removes post-failure records (leakage prevention), deduplicates to one record per serial number |
| **Train/Val/Test/Prod Split** | Splits 40/10/10/40 by serial number to prevent data leakage |
| **Save to Feature Store** | Writes each split to a SageMaker Feature Group with record IDs and event timestamps |

**Outputs:** Four Feature Groups (`-train`, `-validation`, `-test`, `-production`) in SageMaker Feature Store; Feature Group names saved to `.env`.

---

## 3. Model Training & Deployment — `base_model.ipynb`

Trains two models, deploys the best one, and creates monitoring baselines.

| Step | What it does |
|------|-------------|
| **Load Features** | Reads train/val/test Feature Groups from Feature Store |
| **Baseline Model (Logistic Regression)** | Trains on 5 SMART attributes only; tunes decision threshold for ≥ 70 % recall with minimum false positives |
| **LightGBM Model** | Trains on all 7 features with `scale_pos_weight` (sqrt-dampened class imbalance), early stopping on validation AUC, and threshold tuning for ≥ 70 % recall |
| **Model Comparison** | Side-by-side confusion matrices and metric summary (recall, precision, FPR, ROC-AUC, average precision) |
| **Deploy to SageMaker** | Packages model artifacts (`lgb_model.txt`, `model_metadata.json`, `inference.py`), uploads to S3, deploys as a real-time SageMaker endpoint with Data Capture enabled |
| **Model Registry** | Registers the model as a versioned Model Package with evaluation metrics |
| **Model Card** | Creates a SageMaker Model Card with intended uses, training details, and ethics considerations |
| **Generate Baselines** | Produces data-quality and model-quality baseline artifacts (recall floor, precision floor, FP ceiling, ROC-AUC floor, drift z-score bounds) and uploads to S3 |

**Outputs:** Live SageMaker endpoint, Model Registry entry, Model Card, baseline artifacts in S3; endpoint name, model S3 URI, baseline URIs, and constraints URI saved to `.env`.

---

## 4. Model Monitoring & CI/CD — `model_monitoring_cicd.ipynb`

Implements automated production monitoring, quality gates, infrastructure alarms, and a SageMaker Pipeline that retrains and redeploys the model when performance degrades.

### Override Parameters

Two binary flags at the top of the notebook let you force or suppress retraining regardless of quality-gate results:

| Flag | Default | Effect |
|------|---------|--------|
| `FORCE_RETRAIN` | `False` | Forces retraining even if all gates pass |
| `FORCE_SKIP_RETRAIN` | `False` | Suppresses retraining even if gates fail |

If both are `True`, `FORCE_SKIP_RETRAIN` wins (safe default). These overrides are also passed as pipeline parameters (`ForceRetrain`, `ForceSkipRetrain`).

### Notebook Sections

| Step | What it does |
|------|-------------|
| **Load Constraints & Baselines** | Retrieves monitoring constraints JSON and baseline artifacts produced by `base_model.ipynb` |
| **Quality Gate Functions** | Defines reusable gate functions: recall degradation (hard), metric regression (soft), data drift (hard) |
| **Load Production Data** | Reads the production Feature Group as unseen evaluation data |
| **Generate Predictions** | Loads the LightGBM model locally, predicts on production data using the tuned threshold |
| **Run Quality Gates** | Executes all gates: recall ≥ 70 %, secondary metric regression (≥ 2 of 3 triggers retrain), null-rate and z-score drift detection; prints a full verdict report |
| **Detailed Evaluation Report** | Classification report, confusion matrix, and precision-recall curve on production data |
| **Retraining Signal** | Writes a retraining-trigger JSON to S3; optionally publishes an SNS notification |
| **Save Monitoring Results** | Persists timestamped monitoring JSON as an audit trail |
| **SageMaker Model Monitor (Optional)** | Configures scheduled data-quality and model-quality monitoring on the live endpoint |
| **CloudWatch Alarms** | Creates 8 infrastructure alarms: 5xx/4xx error rates, p99 latency, zero-traffic canary, CPU/memory/disk utilization, overhead latency |
| **CloudWatch Dashboard** | Consolidated dashboard with invocation counts, latency graphs, utilization gauges, and alarm-status widgets |
| **SageMaker Pipeline DAG** | Registers a fully automated pipeline: `EvaluateProductionData` → `CheckRetraining` → `RetrainAndDeploy` (or `SaveHealthyReport`). The retrain step trains LightGBM, packages artifacts, deploys to the endpoint, and registers the new model in Model Registry — no manual intervention required |
| **Cleanup** | Deletes CloudWatch alarms, dashboard, Model Monitor schedules, the pipeline, and S3 monitoring artifacts |

### Pipeline DAG

```
┌──────────────────────────┐
│  EvaluateProductionData  │  Feature Store → inference → quality gates
└────────────┬─────────────┘
             │
┌────────────▼─────────────┐
│     CheckRetraining      │  status == "RETRAIN" ?
└──────┬───────────┬───────┘
  [YES]│           │[NO]
┌──────▼──────┐ ┌──▼─────────────┐
│  Retrain &  │ │  SaveHealthy   │
│  Deploy     │ │    Report      │
│  + Registry │ │                │
└─────────────┘ └────────────────┘
```

**Outputs:** CloudWatch alarms and dashboard, SageMaker Pipeline (`hdd-failure-monitoring`), audit-trail JSONs in S3.

---

## Project Structure

```
aai-540-group9/
├── load_data.ipynb              # 1. S3 setup & raw data ingestion
├── feature_engineering.ipynb    # 2. Feature engineering & Feature Store
├── base_model.ipynb             # 3. Model training, deployment & baselines
├── model_monitoring_cicd.ipynb  # 4. Monitoring, quality gates & retraining pipeline
├── inference.py                 # Custom inference script for SageMaker endpoint
├── requirements.txt             # Python dependencies
├── lgb_model.txt                # Trained LightGBM model (text format)
├── model_metadata.json          # Model threshold & feature config
├── .env                         # Shared configuration (auto-generated)
└── README.md
```

---

## Prerequisites

- AWS account with SageMaker, S3, CloudWatch, and Athena access
- IAM role with sufficient permissions (e.g., `LabRole`)
- Python 3.10+ with packages listed in `requirements.txt`
- SageMaker Studio or a SageMaker notebook instance

## Quick Start

```bash
# 1. Set AWS credentials
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...

# 2. Run notebooks in order
#    load_data → feature_engineering → base_model → model_monitoring_cicd
```