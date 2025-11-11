#!/usr/bin/env python3
"""scripts/infer_column_types.py

Infer suitable data types for each column in the project's Telco churn CSV.
It prefers `cleaned_telco_customer_churn.csv` if present, otherwise the raw
`WA_Fn-UseC_-Telco-Customer-Churn.csv` file.

Outputs a human readable list: column name, suggested type, pandas dtype,
unique count, sample values.
"""
from __future__ import annotations
import os
import sys
import pandas as pd
import numpy as np
import re


def choose_file():
    opts = [
        "cleaned_telco_customer_churn.csv",
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "telco_customer_churn.csv",
    ]
    for p in opts:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No known CSV found in workspace. Looked for: %s" % ",".join(opts))


def is_boolean_like(series: pd.Series) -> bool:
    vals = set([str(x).strip().lower() for x in series.dropna().unique()[:50]])
    bool_like = vals <= {"yes", "no", "true", "false", "y", "n", "0", "1"}
    if bool_like and len(vals) <= 4:
        return True
    return False


def is_datetime_like(series: pd.Series) -> bool:
    # try to coerce a small sample to datetime
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
        frac = parsed.notna().mean()
        return frac >= 0.8
    except Exception:
        return False


def suggest_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        return "numeric"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if is_boolean_like(series):
        return "boolean (Yes/No)"
    if is_datetime_like(series):
        return "datetime"
    if pd.api.types.is_categorical_dtype(series):
        return "categorical"
    if pd.api.types.is_object_dtype(series):
        nunique = series.nunique(dropna=True)
        if nunique <= 50 or (nunique / max(1, len(series)) < 0.05):
            return "categorical"
        # If many unique but short strings, still treat as text
        avg_len = series.dropna().astype(str).map(len).mean()
        if avg_len <= 30 and nunique < 1000:
            return "categorical/text"
        return "text"
    return str(series.dtype)


def main():
    path = choose_file()
    print("Using file:", path)
    df = pd.read_csv(path)

    rows = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        suggested = suggest_type(s)
        nunique = int(s.nunique(dropna=True))
        nnull = int(s.isna().sum())
        sample_vals = s.dropna().astype(str).unique()[:5].tolist()
        rows.append((col, suggested, dtype, nunique, nnull, sample_vals))

    # Print results
    print("\nColumn suggestions:\n")
    for col, suggested, dtype, nunique, nnull, sample_vals in rows:
        print(f"- {col}")
        print(f"    suggested: {suggested}")
        print(f"    pandas dtype: {dtype}")
        print(f"    unique_values: {nunique}, null_count: {nnull}")
        print(f"    sample: {sample_vals}")
        print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(2)
