import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
import logging

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Basic configuration if the consuming app hasn't configured logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Module-level cache for loaded artifacts
MODEL = None
FEATURE_LIST = None
SCALER = None


def _load_artifacts():
    """Load model, feature list and scaler from `models/best_model.pkl` if present.

    Returns (model, feature_list, scaler_or_None)
    Caches results in module-level variables MODEL, FEATURE_LIST, SCALER.
    """
    global MODEL, FEATURE_LIST, SCALER
    if MODEL is not None or FEATURE_LIST is not None or SCALER is not None:
        return MODEL, FEATURE_LIST, SCALER

    p = "models/best_model.pkl"
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        # propagate as a clear error
        logger.exception("Failed to load model artifacts")
        raise Exception(f"Could not load '{p}': {e}")

    if isinstance(data, dict):
        MODEL = data.get("model") or data.get("estimator") or data.get("pipeline")
        FEATURE_LIST = data.get("features") or data.get("feature_list")
        SCALER = data.get("scaler") or data.get("preprocessor")
        logger.info("Loaded artifacts from %s: model=%s, features=%s, scaler=%s", p, type(MODEL).__name__ if MODEL is not None else None, 'present' if FEATURE_LIST else None, 'present' if SCALER else None)
    else:
        # Unknown structure: try best-effort
        MODEL = data
        FEATURE_LIST = None
        SCALER = None

    logger.debug("Artifact load complete: MODEL=%s FEATURE_LIST=%s SCALER=%s", type(MODEL).__name__ if MODEL is not None else None, FEATURE_LIST if FEATURE_LIST is not None else None, type(SCALER).__name__ if SCALER is not None else None)

    return MODEL, FEATURE_LIST, SCALER


def load_model():
    """Backward-compatible loader: returns (model, feature_list).

    Internally uses `_load_artifacts()` which may also populate `SCALER`.
    """
    model, features, scaler = _load_artifacts()
    return model, features


def preprocess_data(df):
    """Cleans and preprocesses uploaded dataset for prediction."""
    
    # Drop customerID if exists
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Scale numerical columns (use same columns as in training if scaler available)
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    present_num_cols = [c for c in num_cols if c in df.columns]
    if present_num_cols:
        # Try to use a persisted scaler if available
        try:
            # Ensure artifacts loaded
            _load_artifacts()
        except Exception:
            # No persisted artifacts available; fall back to fitting a local scaler
            warnings.warn("Persisted scaler not found; scaling using a locally-fitted StandardScaler.")

        if SCALER is not None:
            try:
                df[present_num_cols] = SCALER.transform(df[present_num_cols])
            except Exception:
                # If the scaler can't transform due to shape mismatch, fall back
                warnings.warn("Saved scaler couldn't transform data; falling back to local scaling.")
                local_scaler = StandardScaler()
                df[present_num_cols] = local_scaler.fit_transform(df[present_num_cols])
        else:
            local_scaler = StandardScaler()
            df[present_num_cols] = local_scaler.fit_transform(df[present_num_cols])
    
    # Derived features
    if 'tenure' in df.columns and 'TotalCharges' in df.columns:
        df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df


def predict_df(df, model=None):
    """Take an input dataframe (raw uploaded), preprocess, align features with the
    training feature list, run the model and return the original dataframe with
    added columns: 'Churn_Probability', 'Churn_Binary', 'Predicted_Churn'.

    Note: this function will call `load_model()` to obtain both the model and
    the expected `feature_list` saved at training time.
    """
    # Keep a copy of original dataframe so we don't remove identifying cols like customerID
    original = df.copy()

    # Load model and expected features
    try:
        model_obj, feature_list = load_model()
    except Exception:
        raise

    # Preprocess a copy for prediction (preprocess_data may drop customerID)
    processed = preprocess_data(df.copy())

    # Ensure all expected features exist; add missing with zeros
    if feature_list is not None:
        for col in feature_list:
            if col not in processed.columns:
                processed[col] = 0

        # Reorder columns to match training
        processed = processed[feature_list]

    # Predict probabilities (try predict_proba, fall back to predict)
    try:
        probs = model_obj.predict_proba(processed)[:, 1]
    except Exception:
        # Not all models implement predict_proba; fall back to predict
        preds = model_obj.predict(processed)
        # If predictions are 0/1 class labels, map them to 0/1 probabilities
        probs = preds.astype(float)

    # Attach predictions to the original dataframe
    original["Churn_Probability"] = probs
    original["Churn_Binary"] = (original["Churn_Probability"] >= 0.5).astype(int)
    original["Predicted_Churn"] = original["Churn_Binary"].map({0: "No Churn", 1: "Churn"})

    return original


