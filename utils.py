import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

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

def load_model():
    """Loads the trained ML model."""
    return joblib.load("best_model.pkl")
