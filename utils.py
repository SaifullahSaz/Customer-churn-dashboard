import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import pickle

# Load the same scaler used during training
scaler = StandardScaler()

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

    # Scale numerical columns (use same columns as in training)
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in num_cols:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[[col]])
    
    # Derived features
    if 'tenure' in df.columns and 'TotalCharges' in df.columns:
        df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df

# def load_model():
#     """Loads the trained ML model."""
#     return joblib.load("best_model.pkl")

# Load both model and feature list
    # model, feature_list = joblib.load("best_model.pkl")
    # return model, feature_list

def load_model():
    try:
        with open("models/best_model.pkl", "rb") as file:
            model_data = pickle.load(file)
        model = model_data["model"]
        feature_list = model_data["features"]
        return model, feature_list
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

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
    except Exception as e:
        raise

    # Preprocess a copy for prediction (preprocess_data may drop customerID)
    processed = preprocess_data(df.copy())

    # Ensure all expected features exist; add missing with zeros
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


