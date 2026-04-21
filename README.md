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
README.md
requirements.txt
Dockerfile

src/train.py
api/app.py

artifacts/model.pkl
outputs/results.txt





 Results
Model: LightGBM (LGBMClassifier)
PR-AUC: 0.955 (Initial) / 0.956 (Optimized)
ROC-AUC: 0.99
Dataset: NASA CMAPSS (FD001)
Split Strategy: Group-based splitting by engine_id (ensures no data leakage from the same engine between train/test).
Feature Engineering Applied:
Rolling Mean: 5-cycle window mean for all sensors to capture short-term degradation trends.
Lag Features: 1-cycle lag to capture the immediate change in sensor values.
Targeting: Binary classification for failure within 30 cycles.


Run
pip install -r requirements.txt  
python src/train.py  
python api/app.py
    ├── architecture.png
    ├── workflow.png
