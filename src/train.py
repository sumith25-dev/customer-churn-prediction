"""
train.py — Customer Churn Prediction Training Pipeline
Implements: XGBoost + SMOTE + SHAP + cross-validated grid search
Dataset   : IBM Telco Customer Churn (7,043 records)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, recall_score,
                             classification_report, confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# -- Paths ----------------------------------------------------------------------
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(ROOT, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# -- 1. Load & Clean ------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    print(f"[1/6] Loading data from: {path}")
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df.drop(columns=["customerID"], inplace=True, errors="ignore")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    print(f"    Shape: {df.shape} | Churn rate: {df['Churn'].mean():.1%}")
    return df


# -- 2. Feature Engineering ----------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Engineering features...")

    # Tenure buckets
    df["tenure_group"] = pd.cut(df["tenure"], bins=[0,12,24,48,72],
                                labels=["0-1yr","1-2yr","2-4yr","4-6yr"])

    # Charge ratios
    df["charges_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["high_value"]        = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

    # Service count
    service_cols = ["PhoneService","OnlineSecurity","OnlineBackup",
                    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    for c in service_cols:
        df[c] = df[c].map({"Yes":1,"No":0,"No internet service":0,"No phone service":0}).fillna(0).astype(int)
    df["service_count"] = df[service_cols].sum(axis=1)

    # Binary encode Yes/No cols
    binary_cols = ["Partner","Dependents","PaperlessBilling","MultipleLines"]
    for c in binary_cols:
        if c in df.columns:
            df[c] = df[c].map({"Yes":1,"No":0,"No phone service":0}).fillna(0).astype(int)

    df["gender"] = (df["gender"] == "Male").astype(int)
    
    # Fill any remaining NaN values before one-hot encoding
    df = df.fillna(df.select_dtypes(include=['number']).mean())

    # One-hot encode categoricals
    cat_cols = ["Contract","InternetService","PaymentMethod","tenure_group"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    print(f"    Features after engineering: {df.shape[1]-1}")
    return df


# -- 3. SMOTE Class Balancing --------------------------------------------------
def apply_smote(X_train, y_train):
    print("[3/6] Applying SMOTE class balancing...")
    before = y_train.value_counts().to_dict()
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    after  = pd.Series(y_res).value_counts().to_dict()
    print(f"    Before: {before} -> After: {after}")
    return X_res, y_res


# -- 4. Cross-validated Grid Search --------------------------------------------
def train_model(X_train, y_train):
    print("[4/6] Running cross-validated grid search (StratifiedKFold=5)...")
    param_grid = {
        "n_estimators"   : [200, 300],
        "max_depth"      : [4, 6],
        "learning_rate"  : [0.05, 0.1],
        "subsample"      : [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "scale_pos_weight": [1],
    }
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(xgb, param_grid, cv=cv, scoring="roc_auc",
                        n_jobs=-1, verbose=0)
    t0 = time.time()
    grid.fit(X_train, y_train)
    print(f"    Best params: {grid.best_params_}")
    print(f"    Best CV AUC: {grid.best_score_:.4f}  [{time.time()-t0:.0f}s]")
    return grid.best_estimator_


# -- 5. Evaluate ----------------------------------------------------------------
def evaluate(model, X_test, y_test, feature_cols, save_dir):
    print("[5/6] Evaluating on held-out test set...")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    rec  = recall_score(y_test, y_pred)
    print(f"\n    {'Metric':<20} {'Value':>8}")
    print(f"    {'-'*30}")
    print(f"    {'Accuracy':<20} {acc:>8.4f}")
    print(f"    {'AUC-ROC':<20} {auc:>8.4f}")
    print(f"    {'Recall':<20} {rec:>8.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn','Churn'])}")

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=["No Churn","Churn"], yticklabels=["No Churn","Churn"], ax=ax)
    ax.set_title("Confusion Matrix"); ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=120)
    plt.close()

    # Save ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0,1],[0,1], "k--"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=120)
    plt.close()

    return {"accuracy": acc, "auc_roc": auc, "recall": rec}


# -- 6. SHAP Explainability -----------------------------------------------------
def compute_shap(model, X_train, feature_cols, save_dir):
    print("[6/6] Computing SHAP explainer & summary plot...")
    explainer = shap.TreeExplainer(model)
    sample    = X_train.iloc[:500]       # sample for speed
    shap_vals = explainer.shap_values(sample)

    # Summary bar plot
    plt.figure()
    shap.summary_plot(shap_vals, sample, feature_names=feature_cols,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_summary.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("    SHAP summary saved.")
    return explainer


# -- Main -----------------------------------------------------------------------
def main():
    if not os.path.exists(DATA_PATH):
        print(f"\n[ERROR]  Dataset not found at:\n    {DATA_PATH}")
        print("\n[DOWNLOAD]  Download it from Kaggle:")
        print("    https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print("    Place the CSV in the  data/  folder and re-run.\n")
        sys.exit(1)

    df = load_data(DATA_PATH)
    df = engineer_features(df)

    # Split
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Scale
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    X_train_sc = pd.DataFrame(X_train_sc, columns=feature_cols)
    X_test_sc  = pd.DataFrame(X_test_sc,  columns=feature_cols)

    # SMOTE on training only
    X_res, y_res = apply_smote(X_train_sc, y_train)
    X_res = pd.DataFrame(X_res, columns=feature_cols)

    # Train
    model = train_model(X_res, y_res)

    # Evaluate
    metrics = evaluate(model, X_test_sc, y_test, feature_cols, MODELS_DIR)

    # SHAP
    explainer = compute_shap(model, X_res, feature_cols, MODELS_DIR)

    # Persist artifacts
    joblib.dump(model,       os.path.join(MODELS_DIR, "xgb_model.pkl"))
    joblib.dump(scaler,      os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_cols,os.path.join(MODELS_DIR, "feature_cols.pkl"))
    joblib.dump(explainer,   os.path.join(MODELS_DIR, "shap_explainer.pkl"))

    print("\n[OK]  All artifacts saved to models/")
    print(f"    Accuracy : {metrics['accuracy']:.4f}")
    print(f"    AUC-ROC  : {metrics['auc_roc']:.4f}")
    print(f"    Recall   : {metrics['recall']:.4f}")
    print("\n[START]  Run: streamlit run app.py\n")


if __name__ == "__main__":
    main()
