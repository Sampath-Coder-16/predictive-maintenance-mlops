# Predictive Maintenance ML System

## Overview
Production-grade machine learning system to predict equipment failure using time-series sensor data.

## Features
- Time-series feature engineering (rolling, lag)
- Imbalanced classification (PR-AUC)
- LightGBM model
- SHAP explainability
- Flask API
- Docker deployment

## Dataset
NASA CMAPSS Turbofan Engine Dataset

## Project Structure
predictive-maintenance-mlops/
│
├── README.md
├── requirements.txt
├── Dockerfile
│
├── data/
│   └── sample_data.csv   (optional small sample)
│
├── src/
│   ├── train.py
│   ├── features.py
│   └── evaluate.py
│
├── api/
│   └── app.py
│
├── artifacts/
│   └── model.pkl   (uploaded file)
│
├── notebooks/
│   └── ZAALIMA_PROJECT.ipynb    ( notebook)
│
├── outputs/
│   ├── pr_auc_score.png
│   ├── shap_summary.png
│   └── model_results.txt
│
└── images/
    ├── architecture.png
    ├── workflow.png
