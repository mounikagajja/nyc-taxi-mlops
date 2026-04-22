# NYC Taxi Trip Duration - End-to-End MLOps Pipeline

[![CI](https://github.com/mounikagajja/nyc-taxi-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/mounikagajja/nyc-taxi-mlops/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![MLflow](https://img.shields.io/badge/mlflow-3.x-orange.svg)]()

> End-to-end MLOps pipeline predicting NYC taxi trip duration - built on Azure Databricks, Delta Lake, MLflow 3, FastAPI, and GitHub Actions CI/CD.

## What This Project Does

Predicts how long a NYC taxi trip will take given pickup/dropoff location, time of day, and trip distance. Same problem Uber and Lyft solve for ETA estimation.

## Architecture

Raw Parquet → Delta Lake → Spark Features → LightGBM + MLflow → FastAPI → KS Drift Monitor → Retrain Trigger

## Tech Stack

| Layer | Technology |
|---|---|
| Data storage | Delta Lake 4.x (ACID, time travel, partitioning) |
| Processing | Apache Spark on Databricks Free Edition |
| Experiment tracking | MLflow 3 with Unity Catalog |
| Model | LightGBM with SHAP explainability |
| Serving | FastAPI + Docker |
| Drift detection | KS test (scipy) with MLflow logging |
| CI/CD | GitHub Actions (lint + test on every push) |

## Model Results

| Metric | Value |
|---|---|
| RMSE | 299.6s (5.0 min) |
| MAE | 178.5s (3.0 min) |
| Training rows | 2,195,185 |
| Test rows | 548,797 |

## Quickstart

    git clone https://github.com/mounikagajja/nyc-taxi-mlops.git
    cd nyc-taxi-mlops
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    pytest tests/ -v

## Project Structure

    src/
    ingestion/         - Raw Parquet to Delta table
    features/          - Spark feature engineering
    training/          - LightGBM + MLflow tracking
    serving/           - FastAPI REST endpoint
    monitoring/        - KS drift detection

## Key Design Decisions

- **LightGBM over deep learning** - tabular regression with interpretability. SHAP values explain each prediction.
- **Delta Lake partitioned by pickup_date** - enables partition pruning for time-based retraining windows.
- **MLflow 3 aliases** - uses champion/challenger pattern instead of deprecated Production stage.
- **KS test for drift** - detects distributional shift between reference and current data. Retrain triggered if drift share exceeds 20%.

## Monitoring

Drift detection compares January 2024 (reference) vs February 2024 (current) data. KS test runs per feature. All drift scores logged to MLflow for full audit trail.

## Dataset

NYC Taxi and Limousine Commission (TLC) Yellow Taxi Trip Records - January and February 2024.

Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
