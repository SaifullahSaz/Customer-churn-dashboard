# Customer Churn Dashboard

This repository contains a Streamlit dashboard for predicting telecom customer churn.

Quick start

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the app:

```powershell
streamlit run app.py
```

3. Use the left Pages menu to visit Upload & Predict and Analysis.

Notes

- Add your Supabase credentials in `.streamlit/secrets.toml` if you want to use the Supabase upload feature.
- The app expects a trained model file at `models/best_model.pkl`. This file should be a pickle containing at least `model` and `features` keys; optionally include `scaler` to ensure preprocessing uses the same transformation as training.

Running tests

```powershell
pytest -q
```

Deployment

- A `Dockerfile` is provided for containerized deployment.

Next steps

- Pin dependency versions and add CI (done).
- Improve model artifact handling if your training pipeline outputs a different structure.
