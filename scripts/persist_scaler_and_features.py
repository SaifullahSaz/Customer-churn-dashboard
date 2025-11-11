import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

MODEL_P = os.path.join("models", "best_model.pkl")
DATA_P = "cleaned_telco_customer_churn.csv"

if not os.path.exists(MODEL_P):
    print(f"{MODEL_P} not found. Aborting.")
    raise SystemExit(1)

with open(MODEL_P, "rb") as f:
    data = pickle.load(f)

if isinstance(data, dict):
    model = data.get("model") or data.get("estimator") or data.get("pipeline")
    features = data.get("features") or data.get("feature_list")
else:
    model = data
    features = None

if features is None:
    print("feature_list not found in artifact; cannot continue safely.")
    raise SystemExit(1)

if data.get("scaler") is not None:
    print("Scaler already present; nothing to do.")
    raise SystemExit(0)

if not os.path.exists(DATA_P):
    print(f"Training data {DATA_P} not found; please provide it in repo root.")
    raise SystemExit(1)

# Load cleaned training data and prepare numeric columns
df = pd.read_csv(DATA_P)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

num_cols = [c for c in ['tenure', 'MonthlyCharges', 'TotalCharges'] if c in df.columns]
if not num_cols:
    print('No numeric columns found to fit scaler; aborting')
    raise SystemExit(1)

scaler = StandardScaler()
scaler.fit(df[num_cols])

# Build artifact dict and overwrite pickle
artifact = {
    "model": model,
    "features": features,
    "scaler": scaler,
}

with open(MODEL_P, "wb") as f:
    pickle.dump(artifact, f)

print(f"Persisted scaler and feature_list into {MODEL_P}")
