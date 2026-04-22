# NYC Taxi Trip Duration — End-to-End MLOps Pipeline

[![CI](https://github.com/mounikagajja/nyc-taxi-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/mounikagajja/nyc-taxi-mlops/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![MLflow](https://img.shields.io/badge/mlflow-3.x-orange.svg)]()

> End-to-end MLOps pipeline predicting NYC taxi trip duration — built on Azure Databricks, Delta Lake, MLflow 3, FastAPI, and GitHub Actions CI/CD.

## What This Project Does
Predicts how long a NYC taxi trip will take (in seconds) given pickup/dropoff location, time of day, and trip distance. This is the same problem Uber and Lyft solve for ETA estimation.

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
```bash
git clone https://github.com/mounikagajja/nyc-taxi-mlops.git
cd nyc-taxi-mlops
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -v
```

## Project Structure