"""
predict.py — Inference helpers for the Streamlit app
"""
import os
import numpy as np
import pandas as pd
import joblib
import shap

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")


# ── Load saved artifacts ───────────────────────────────────────────────────────
def load_model_artifacts():
    model        = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
    scaler       = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
    explainer    = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.pkl"))
    return model, scaler, feature_cols, explainer


# ── Encode raw form dict → model-ready DataFrame ──────────────────────────────
def _encode_input(raw: dict, feature_cols: list) -> pd.DataFrame:
    """Mirror the feature engineering in train.py for a single record."""
    d = {}
    d["tenure"]           = raw.get("tenure", 0)
    d["MonthlyCharges"]   = raw.get("MonthlyCharges", 0)
    d["TotalCharges"]     = raw.get("TotalCharges", 0)
    d["SeniorCitizen"]    = raw.get("SeniorCitizen", 0)
    d["gender"]           = 1 if raw.get("gender","Male")=="Male" else 0
    d["charges_per_month"]= d["TotalCharges"] / (d["tenure"] + 1)
    d["high_value"]       = 1 if d["MonthlyCharges"] > 65 else 0

    # Binary cols
    for col in ["Partner","Dependents","PaperlessBilling","MultipleLines",
                "PhoneService","OnlineSecurity","OnlineBackup",
                "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]:
        val = raw.get(col, "No")
        d[col] = 1 if val in ("Yes", 1) else 0

    # Service count
    svc_cols = ["PhoneService","OnlineSecurity","OnlineBackup",
                "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    d["service_count"] = sum(d.get(c, 0) for c in svc_cols)

    # Tenure group (one-hot)
    t = d["tenure"]
    d["tenure_group_0-1yr"] = 1 if t <= 12 else 0
    d["tenure_group_1-2yr"] = 1 if 12 < t <= 24 else 0
    d["tenure_group_2-4yr"] = 1 if 24 < t <= 48 else 0
    d["tenure_group_4-6yr"] = 1 if t > 48 else 0

    # Contract (one-hot)
    contract = raw.get("Contract","Month-to-month")
    d["Contract_Month-to-month"] = 1 if contract == "Month-to-month" else 0
    d["Contract_One year"]       = 1 if contract == "One year" else 0
    d["Contract_Two year"]       = 1 if contract == "Two year" else 0

    # Internet Service (one-hot)
    inet = raw.get("InternetService","Fiber optic")
    d["InternetService_DSL"]         = 1 if inet == "DSL" else 0
    d["InternetService_Fiber optic"] = 1 if inet == "Fiber optic" else 0
    d["InternetService_No"]          = 1 if inet == "No" else 0

    # Payment Method (one-hot)
    pay = raw.get("PaymentMethod","Electronic check")
    d["PaymentMethod_Bank transfer (automatic)"] = 1 if "Bank" in pay else 0
    d["PaymentMethod_Credit card (automatic)"]   = 1 if "Credit" in pay else 0
    d["PaymentMethod_Electronic check"]          = 1 if "Electronic" in pay else 0
    d["PaymentMethod_Mailed check"]              = 1 if "Mailed" in pay else 0

    # Build row aligned to training features (fill missing with 0)
    row = {col: d.get(col, 0) for col in feature_cols}
    return pd.DataFrame([row], columns=feature_cols)


# ── Single prediction ─────────────────────────────────────────────────────────
def predict_single(raw: dict, model, scaler, feature_cols: list, explainer) -> dict:
    df_input = _encode_input(raw, feature_cols)
    df_scaled = pd.DataFrame(scaler.transform(df_input), columns=feature_cols)

    prob = float(model.predict_proba(df_scaled)[0, 1])
    risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"

    # SHAP values for this record
    sv = explainer.shap_values(df_scaled)
    shap_pairs = sorted(zip(feature_cols, sv[0]), key=lambda x: abs(x[1]), reverse=True)

    return {
        "probability" : prob,
        "risk_level"  : risk,
        "shap_values" : sv,
        "top_features": shap_pairs[:10],
    }


# ── Bulk prediction ────────────────────────────────────────────────────────────
def predict_bulk(df: pd.DataFrame, model, scaler, feature_cols: list) -> pd.DataFrame:
    """Accepts a raw Telco-format DataFrame, returns predictions DataFrame."""
    # Light preprocessing to match training
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    if "Churn" in df.columns:
        df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Encode each row
    rows = []
    for _, row in df.iterrows():
        raw = row.to_dict()
        rows.append(_encode_input(raw, feature_cols).iloc[0])

    X = pd.DataFrame(rows, columns=feature_cols).fillna(0)
    X_sc = pd.DataFrame(scaler.transform(X), columns=feature_cols)

    probs = model.predict_proba(X_sc)[:, 1]
    preds = (probs > 0.5).astype(int)
    risk  = ["HIGH" if p > 0.7 else "MEDIUM" if p > 0.4 else "LOW" for p in probs]

    result = df.copy()
    result["Churn_Probability"] = probs.round(4)
    result["Churn_Prediction"]  = preds
    result["Risk_Level"]        = risk
    return result
