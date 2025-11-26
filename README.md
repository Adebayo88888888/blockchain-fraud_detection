# 🚨 Ethereum Fraud Detection – On-Chain ML Classifier  

Machine Learning Model • FastAPI • Docker • AWS Elastic Beanstalk

---

## 📌 Overview

This project implements an end-to-end **Ethereum fraud detection system** powered by machine learning.  
It analyzes wallet behavior and transaction-based features to predict whether activity is **fraudulent** or **legitimate** using a trained **XGBoost classifier**.

The entire solution is deployed as a Dockerized **FastAPI** application running on **AWS Elastic Beanstalk**.

---

## 🔍 Problem Statement

Fraudulent activity on public blockchains is rising and increasingly complex.  
Manual identification is inconsistent and not scalable.

This project provides an automated fraud detection pipeline that:

- Flags suspicious Ethereum wallet behavior  
- Assigns real-time fraud probability scores  
- Enables monitoring dashboards and API integrations  
- Supports investigators, analysts, and automated risk systems  

---

## 🧩 Dataset & Feature Description

Each row in the dataset represents summarized historical wallet behavior derived from Ethereum transactions.

| Feature Type | Feature Name                       | Description |
|--------------|------------------------------------|-------------|
| Numerical    | `Hour`                             | Hour of day when the transaction occurred |
| Numerical    | `total_received`                   | Sum of incoming ETH value |
| Numerical    | `mean_value_received`              | Average value per received transaction |
| Numerical    | `time_diff_first_last_received`    | Time between first and last incoming transaction |
| Numerical    | `total_tx_sent`                    | Total outgoing transactions |
| Numerical    | `total_tx_sent_unique`             | Unique outgoing transaction targets |
| Binary       | `has_activity`                     | Indicates whether wallet shows active behavior |
| Target       | `Fraud`                            | 1 = fraud, 0 = legitimate |

These engineered features feed directly into the ML model.

---

## 📊 Key Insights from EDA

- Wallets with **many unique outgoing transactions** tend to exhibit higher fraud probability.  
- Fraudulent behavior often displays **non-linear patterns**, especially regarding value transfers.  
- Sudden bursts of activity (large `time_diff_first_last_received`) correspond strongly with fraud.  
- Fraud is more common during **off-peak hours**, making temporal features useful.

---

## Model Summary

- **Model**: XGBoost Classifier  
- **Output**:  
  - `fraud_probability` (0–1)  
  - `is_fraud` (True/False)  
  - Risk category: *Low, Medium, High Risk*


💻 Tech Stack

  * Modeling: XGBoost, pandas, scikit-learn

  * Backend: FastAPI

  * Serialization: joblib

  * Environment: Pipenv

  * Containerization: Docker

  * Cloud Deployment: AWS Elastic Beanstalk


🏁 Conclusion

This Ethereum fraud detection platform provides a fully operational, cloud-deployed machine learning pipeline.
By analyzing on-chain behavioral patterns, it delivers real-time fraud scoring suitable for:

- Risk monitoring dashboards

- Compliance automation

- dApp integrations

- Security analytics platforms

This system is a step toward creating safer, more transparent blockchain ecosystems powered by data-driven intelligence.
