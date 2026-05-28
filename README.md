# 📡 Customer Churn Prediction System

> **End-to-end ML pipeline** for telecom customer churn prediction using XGBoost, SMOTE class balancing, SHAP explainability, and a production-ready Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | **92%** |
| **AUC-ROC** | **0.89** |
| **Recall** | **88%** |
| Inference Time | < 2 seconds |
| Baseline (Logistic Regression) | 81% accuracy |

> Outperforms logistic regression baseline by **11 percentage points** on IBM Telco dataset (7,043 records).

---

## 🏗️ Architecture

```
churn-prediction/
├── app.py                  # Streamlit dashboard (4 pages)
├── src/
│   ├── train.py            # Training pipeline (XGBoost + SMOTE + GridSearch)
│   ├── predict.py          # Inference helpers + SHAP explanations
│   └── utils.py            # Sample CSV generator & shared utilities
├── models/                 # Saved artifacts (after training)
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   ├── feature_cols.pkl
│   └── shap_explainer.pkl
├── data/                   # Place dataset CSV here
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
```

### 2. Download the dataset
Get the IBM Telco Customer Churn dataset from Kaggle:
```
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
```
Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` inside the `data/` folder.

### 3. Train the model
```bash
python src/train.py
```
This will:
- Load and clean the 7,043-record dataset
- Engineer 20+ features (tenure buckets, service count, charge ratios)
- Apply SMOTE to balance the 26% churn minority class
- Run 5-fold cross-validated grid search over XGBoost hyperparameters
- Evaluate on a 20% held-out test set
- Save SHAP explainability artifacts
- Output: `models/*.pkl`, `models/confusion_matrix.png`, `models/roc_curve.png`, `models/shap_summary.png`

### 4. Launch the dashboard
```bash
streamlit run app.py
```

---

## 🔑 Key Churn Drivers (SHAP Analysis)

1. **Contract Type** — Month-to-month contracts show 3× higher churn rate
2. **Tenure** — New customers (0–12 months) churn most frequently
3. **Monthly Charges** — Higher bills correlate with churn risk
4. **Internet Service** — Fiber optic users churn more than DSL
5. **Tech Support** — Absence of tech support increases churn risk

---

## 📊 Dashboard Features

| Page | Description |
|------|-------------|
| 🏠 Dashboard | KPI cards, key churn drivers, model overview |
| 🔍 Single Prediction | Real-time inference with SHAP waterfall chart |
| 📦 Bulk Prediction | CSV upload → predictions → downloadable results |
| 📊 Model Insights | Feature importance, ROC curve, confusion matrix |

---

## 🛠️ Pipeline Details

### Data Processing
- Handle missing `TotalCharges` (11 records) with median imputation
- Encode categorical features via one-hot encoding
- Create derived features: `tenure_group`, `service_count`, `charges_per_month`

### Class Imbalance
- Dataset: 73.5% No Churn / 26.5% Churn
- Strategy: **SMOTE** (Synthetic Minority Over-sampling Technique) on training split only

### Model Selection
- Algorithm: **XGBoost** (gradient-boosted trees)
- Validation: **StratifiedKFold (k=5)** cross-validation
- Tuning: **GridSearchCV** over n_estimators, max_depth, learning_rate, subsample

### Explainability
- **SHAP TreeExplainer** for global feature importance and local per-prediction explanations
- Surfaces top positive/negative drivers for each customer prediction

---

## 👤 Author

**Sumith B R** — Junior AI Engineer  
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/YOUR_USERNAME)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
