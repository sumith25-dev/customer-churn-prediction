"""utils.py — Shared helpers"""
import io
import pandas as pd
import numpy as np


def generate_sample_csv(n: int = 50) -> bytes:
    """Generate a synthetic Telco-like CSV for demo bulk predictions."""
    rng = np.random.default_rng(42)

    contracts       = ["Month-to-month", "One year", "Two year"]
    internet        = ["DSL", "Fiber optic", "No"]
    payment         = ["Electronic check", "Mailed check",
                       "Bank transfer (automatic)", "Credit card (automatic)"]
    yn              = ["Yes", "No"]
    yn_inet         = ["Yes", "No", "No internet service"]

    rows = []
    for i in range(n):
        tenure   = int(rng.integers(0, 73))
        monthly  = round(float(rng.uniform(18, 120)), 2)
        total    = round(monthly * tenure + float(rng.uniform(-50, 50)), 2)
        total    = max(0, total)
        rows.append({
            "customerID"      : f"DEMO-{i+1:04d}",
            "gender"          : rng.choice(["Male","Female"]),
            "SeniorCitizen"   : int(rng.choice([0, 1])),
            "Partner"         : rng.choice(yn),
            "Dependents"      : rng.choice(yn),
            "tenure"          : tenure,
            "PhoneService"    : rng.choice(yn),
            "MultipleLines"   : rng.choice(["Yes","No","No phone service"]),
            "InternetService" : rng.choice(internet),
            "OnlineSecurity"  : rng.choice(yn_inet),
            "OnlineBackup"    : rng.choice(yn_inet),
            "DeviceProtection": rng.choice(yn_inet),
            "TechSupport"     : rng.choice(yn_inet),
            "StreamingTV"     : rng.choice(yn_inet),
            "StreamingMovies" : rng.choice(yn_inet),
            "Contract"        : rng.choice(contracts),
            "PaperlessBilling": rng.choice(yn),
            "PaymentMethod"   : rng.choice(payment),
            "MonthlyCharges"  : monthly,
            "TotalCharges"    : total,
            "Churn"           : rng.choice(yn),
        })

    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue().encode()
