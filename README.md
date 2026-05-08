# Predictive Maintenance ML System

## Overview

A production-grade Machine Learning system designed to predict equipment failure using multivariate time-series sensor data from NASA turbofan engines.

The project focuses on predictive maintenance by identifying potential failures within the next 30 operational cycles, enabling proactive maintenance decisions and reducing unexpected downtime.

The system integrates:

* Advanced feature engineering
* Robust model validation
* Hyperparameter optimization
* Explainable AI (SHAP)
* API deployment
* Docker-based containerization

---

# Key Features

## Time-Series Feature Engineering

Implemented temporal degradation modeling techniques:

* Rolling mean features (5-cycle window)
* Lag-1 features

These features capture short-term degradation patterns and sequential sensor behavior.

---

## Imbalanced Classification Optimization

The dataset exhibits class imbalance, so evaluation was optimized using:

* Precision-Recall AUC (PR-AUC)

This provides more reliable performance measurement than accuracy for predictive maintenance scenarios.

---

## LightGBM-Based Prediction Engine

Used:

### LightGBM (LGBMClassifier)

Benefits:

* Fast training
* High efficiency
* Excellent performance on tabular sensor data
* Strong handling of nonlinear relationships

---

## Explainable AI with SHAP

Integrated SHAP explainability to:

* Interpret model predictions
* Identify critical sensor features
* Improve transparency and trustworthiness

Top influential features included:

* cycle
* sensor_11
* sensor_12
* sensor_4
* sensor_20

---

## REST API Deployment

Built a Flask-based API for:

* Real-time inference
* External application integration
* Production deployment readiness

---

## Dockerized Deployment

Containerized the complete application using Docker to ensure:

* Environment consistency
* Portability
* Scalable deployment

---

# Dataset

## NASA CMAPSS Turbofan Engine Dataset

The project uses the NASA CMAPSS dataset containing:

* Engine operational cycles
* Sensor measurements
* Operational settings
* Simulated degradation behavior

### Objective

Predict whether an engine is likely to fail within the next 30 cycles.

---

# Project Structure

```text id="s6kgm2"
Predictive-Maintenance-ML-System/
│
├── README.md
├── requirements.txt
├── Dockerfile
│
├── src/
│   └── train.py
│
├── api/
│   └── app.py
│
├── artifacts/
│   └── model.pkl
│
├── outputs/
│   ├── results.txt
│   ├── architecture.png
│   └── workflow.png
│
└── data/
    └── nasa.zip
```

---

# Technical Workflow

## 1. Data Preprocessing

* Loaded NASA CMAPSS FD001 dataset
* Calculated Remaining Useful Life (RUL)
* Generated binary target labels
* Removed invalid and NaN rows

---

## 2. Feature Engineering

### Rolling Mean Features

Generated:

* 5-cycle moving averages

Purpose:

* Capture local degradation trends

---

### Lag Features

Generated:

* 1-cycle lag values

Purpose:

* Capture immediate temporal changes in sensor behavior

---

## 3. Train-Test Splitting

Used:

### Group-based splitting by `engine_id`

This prevents:

* Data leakage
* Same engine appearing in both train and test sets

Ensuring realistic evaluation.

---

## 4. Model Training

### Algorithm

* LightGBM Classifier

### Hyperparameter Optimization

Used:

* RandomizedSearchCV
* Cross-validation tuning

Optimized for:

* PR-AUC

---

# Model Results

| Metric           | Score |
| ---------------- | ----- |
| Initial PR-AUC   | 0.955 |
| Optimized PR-AUC | 0.956 |
| ROC-AUC          | 0.99  |

---

# Validation Strategy

## GroupKFold Cross-Validation

Implemented GroupKFold validation grouped by `engine_id` to:

* Prevent temporal leakage
* Improve robustness
* Simulate real-world deployment conditions

Cross-validation scores remained highly consistent across folds, indicating strong generalization performance.

---

# Explainability

## SHAP Analysis

SHAP values were used to analyze:

* Feature importance
* Positive and negative prediction impact
* Sensor contribution patterns

This improved model interpretability and debugging capability.

---

# Deployment

## Flask API

Run API locally:

```bash id="wn9k3t"
python api/app.py
```

API supports:

* Real-time predictions
* JSON input/output
* Integration with external systems

---

# Docker Deployment

## Build Docker Image

```bash id="wldh4o"
docker build -t predictive-maintenance .
```

## Run Container

```bash id="lsjlwm"
docker run -p 5000:5000 predictive-maintenance
```

---

# Installation

## Install Dependencies

```bash id="5hvjlwm"
pip install -r requirements.txt
```

---

# Training the Model

```bash id="kij92y"
python src/train.py
```

---

# Running the API

```bash id="utn6qk"
python api/app.py
```

---

# Future Improvements

Potential enhancements include:

* Real-time streaming inference
* Cloud deployment (AWS/GCP/Azure)
* CI/CD pipeline integration
* Automated retraining
* Deep learning approaches (LSTM/Transformer)
* Live monitoring dashboard

---

# Conclusion

This project demonstrates a complete end-to-end predictive maintenance pipeline combining:

* Time-series machine learning
* Feature engineering
* Explainable AI
* MLOps principles
* Deployment engineering

The final optimized LightGBM system achieved strong predictive performance while maintaining deployment readiness and interpretability for industrial applications.
