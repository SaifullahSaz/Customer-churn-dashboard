# Customer Churn Dashboard

This repository contains a small end-to-end churn prediction project: a training notebook that fits several candidate models on the Telco Customer Churn dataset, saves the best model and artifacts, and a multipage Streamlit app to run predictions, inspect model evaluation results, and optionally persist predictions to Supabase.

This README covers the project purpose, repository layout, how to run the app locally, how to retrain the model, and deployment notes.

## What this project does
- Trains multiple classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost) on the Telco Customer Churn dataset.
- Selects and saves the best model and training artifacts (features list, optional scaler) to `models/best_model.pkl`.
- Exports a metrics table (`models/metrics.json`) summarizing model performance (Accuracy, Precision, Recall, F1, ROC-AUC).
- Provides a multipage Streamlit app (pages/) with pages for Home, Predict, Analysis and Evaluation.
- Integrates optional persistence of predictions to Supabase (configured via Streamlit secrets or environment variables).

## Repository structure (important files)
- `model_training.ipynb` - notebook with the data preparation, model training and evaluation flow.
- `executed_model_training.ipynb` - executed copy produced by CI / local runs (contains outputs).
- `models/best_model.pkl` - persisted model artifact used at inference time (may contain model, feature_list, scaler).
- `models/metrics.json` - machine-readable metrics table extracted from the training notebook.
- `utils.py` - shared preprocessing and prediction helpers used by Streamlit pages.
- `pages/` - Streamlit pages (1_Home, 2_Predict, 3_Analysis, 4_Evaluation).
- `scripts/extract_metrics.py` - helper script that extracts `models/metrics.json` from the executed notebook.
- `requirements.txt` - Python dependencies used by Streamlit and training.

## Quickstart — run locally (Windows PowerShell)
1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked by Policy, run PowerShell as Admin and: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app (root uses pages/ automatic multipage support):

```powershell
streamlit run app.py
# or: streamlit run pages/1_Home.py
```

Notes:
- If `models/best_model.pkl` or `models/metrics.json` are missing the app will still run, but pages that require artifacts will show friendly error messages explaining how to produce them.

## Reproduce training and produce artifacts
If you need to re-run training and produce fresh artifacts and the metrics JSON:

1. Ensure the training dependencies are installed (the requirements above already include the main ML libs). Install notebook execution tools:

```powershell
pip install papermill nbconvert ipykernel seaborn
```

2. Register a kernel (if needed) and run the notebook with papermill (this will create `executed_model_training.ipynb` with outputs):

```powershell
python -m ipykernel install --user --name python3 --display-name python3
python -m papermill model_training.ipynb executed_model_training.ipynb
```

3. Extract the metrics JSON (a small helper is provided):

```powershell
python .\scripts\extract_metrics.py
# This will write models/metrics.json if metrics can be parsed from the executed notebook
```

4. Ensure `models/best_model.pkl` contains the model and optionally `feature_list` and `scaler`. The training notebook saves the best model during the training flow.

## Streamlit pages overview
- `1_Home` — Landing page and quick project overview.
- `2_Predict` — Upload a CSV to run batch predictions. Configure threshold, view/download results, and optionally push results to Supabase.
- `3_Analysis` — Interactive charts and filters for the dataset or prediction results (cohort views, churn-rate by group, distributions).
- `4_Evaluation` — Shows training metrics (prefers `models/metrics.json` if available), loads persisted model summary and lets you upload a labeled test CSV to compute evaluation metrics and plot an ROC curve.

The pages share `utils.py` for preprocessing, artifact loading, and prediction so behavior is consistent across the app.

## Supabase integration
If you want the app to persist predictions to a Supabase table, configure secrets either via Streamlit Cloud UI or in a local `.streamlit/secrets.toml` file (DO NOT commit secrets to git). Example secrets format:

```toml
# .streamlit/secrets.toml (example, do not commit)
[supabase]
url = "https://xyzcompany.supabase.co"
key = "public-or-service-role-key"
table = "predictions"
```

Supabase table schema (recommended minimum):
- `customer_id` TEXT (or primary key)
- `churn_probability` FLOAT
- `predicted_churn` BOOL / INT
- `model_version` TEXT
- `run_at` TIMESTAMP
- `metadata` JSONB (optional)

Implementation notes:
- Use minimally-scoped keys and never store admin credentials in the repo.
- The app performs upserts (or inserts) and reports per-row success/failure counts.

## Deployment notes
- Streamlit Cloud: point the app to the branch you want deployed. Ensure `requirements.txt` contains all runtime deps used by pages.
- If the app throws ImportError on Streamlit Cloud, confirm the deployed branch contains the same `utils.py` and artifacts referenced by the pages.
- For containerized deployment, add a Dockerfile that installs Python deps and runs `streamlit run app.py`.

## Tests & CI
- Add unit tests (recommended) for `utils.preprocess_data` and `utils.predict_df` using `pytest`.
- Add a GitHub Actions workflow to install deps and run tests + linters on PRs.

## Recommended next improvements
- Persist scaler and `feature_list` together with the saved model (if not already) and make `preprocess_data` exclusively use the persisted scaler (avoid fitting on uploaded data).
- Add more unit tests and a CI pipeline.
- Add a SHAP explainability view in `pages/3_Analysis` to surface the top drivers of churn.

## Contributing
- Create a feature branch, add tests for new behavior, open a PR against `main` (or the branch used for deployment). Keep artifacts out of feature branch commits unless they are small and necessary for the app.

## License
This repository follows the license in the project root (if present). If you need a license, add one (MIT/Apache-2.0 are common choices).

If you'd like, I can add this README to the repository now and also create a small `docs/` page with a model card and a troubleshooting section for Streamlit Cloud deployments.
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
