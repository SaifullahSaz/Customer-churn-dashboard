import pandas as pd
import pytest
from utils import preprocess_data, predict_df


def make_sample_df():
    return pd.DataFrame({
        'customerID': ['c1','c2'],
        'tenure': [1, 12],
        'MonthlyCharges': [29.85, 56.95],
        'TotalCharges': [29.85, 678.8],
        'gender': ['Male','Female']
    })


def test_preprocess_basic():
    df = make_sample_df()
    processed = preprocess_data(df.copy())
    # after preprocessing, numeric columns should exist and no NaNs
    assert 'MonthlyCharges' in processed.columns
    assert processed.isna().sum().sum() == 0


class DummyModel:
    def predict_proba(self, X):
        import numpy as np
        # return a 2-column array, probability of class 0 and 1
        return np.vstack([(X.index % 2 == 0).astype(float), (X.index % 2 == 1).astype(float)]).T


def test_predict_df_monkeypatch(monkeypatch):
    df = make_sample_df()

    def fake_load_model():
        # fake feature list to include numeric columns
        return DummyModel(), ['tenure', 'MonthlyCharges', 'TotalCharges']

    monkeypatch.setattr('utils.load_model', fake_load_model)

    result = predict_df(df.copy())
    assert 'Churn_Probability' in result.columns
    assert 'Predicted_Churn' in result.columns
