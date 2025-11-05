import streamlit as st
import pandas as pd
import io
import os
import matplotlib.pyplot as plt

from utils import load_metrics_from_notebook, load_model, preprocess_data


def show_model_summary(model):
    try:
        params = model.get_params()
    except Exception:
        params = None
    st.subheader("Loaded model")
    st.write(type(model).__name__)
    if params:
        st.write("Model parameters:")
        st.json(params)


st.title("Evaluation & Saved Metrics")

st.markdown("This page shows the metrics table produced during training and lets you inspect the saved model. You can also upload a labelled test CSV to compute evaluation metrics and an ROC curve.")

# 1) Prefer a persisted metrics JSON file if it exists, otherwise try to extract from the notebook
metrics_path = os.path.join("models", "metrics.json")
metrics_df = None
if os.path.exists(metrics_path):
    try:
        metrics_df = pd.read_json(metrics_path)
        st.success(f"Loaded metrics from `{metrics_path}`")
    except Exception as e:
        st.error(f"Found `{metrics_path}` but failed to read it: {e}")

if metrics_df is None:
    metrics_df = load_metrics_from_notebook()
    if metrics_df is None:
        st.warning("Could not extract metrics from `model_training.ipynb`. If you have a saved metrics file (e.g., models/metrics.json) consider adding it.")
    else:
        st.subheader("Metrics table extracted from notebook")
        st.dataframe(metrics_df.set_index("Model"))
else:
    st.subheader("Persisted metrics")
    st.dataframe(metrics_df.set_index("Model"))

# 2) Try to load the persisted model
try:
    model, feature_list = load_model()
    st.success("Loaded persisted model from models/best_model.pkl")
    show_model_summary(model)
    if feature_list is not None:
        st.write(f"Saved feature list ({len(feature_list)} features).")
except Exception as e:
    st.error(f"Could not load persisted model: {e}")
    model = None
    feature_list = None

# 3) Option: upload a labeled test CSV to compute evaluation metrics and ROC
st.sidebar.header("Ad-hoc evaluation")
uploaded = st.sidebar.file_uploader("Upload a labelled CSV (must include a 'Churn' ground-truth column)", type=["csv"])
if uploaded is not None:
    try:
        test_df = pd.read_csv(io.BytesIO(uploaded.read()))
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")
        test_df = None

    if test_df is not None:
        if 'Churn' not in test_df.columns:
            st.sidebar.error("Uploaded CSV must include a 'Churn' column with 0/1 labels.")
        elif model is None:
            st.sidebar.error("No model loaded to score the uploaded dataset.")
        else:
            st.sidebar.info("Preprocessing uploaded data and scoring...")
            processed = preprocess_data(test_df.copy())

            # If feature_list is present, align
            if feature_list is not None:
                for col in feature_list:
                    if col not in processed.columns:
                        processed[col] = 0
                processed = processed[feature_list]

            # Predict
            try:
                y_proba = model.predict_proba(processed)[:, 1]
            except Exception:
                y_pred = model.predict(processed)
                y_proba = y_pred.astype(float)

            y_true = test_df['Churn'].astype(int).values

            # Compute metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

            y_pred_label = (y_proba >= 0.5).astype(int)
            results = {
                'Accuracy': accuracy_score(y_true, y_pred_label),
                'Precision': precision_score(y_true, y_pred_label, zero_division=0),
                'Recall': recall_score(y_true, y_pred_label, zero_division=0),
                'F1': f1_score(y_true, y_pred_label, zero_division=0),
                'ROC-AUC': roc_auc_score(y_true, y_proba)
            }

            st.subheader("Evaluation on uploaded test set")
            st.json(results)

            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"ROC (AUC = {results['ROC-AUC']:.3f})")
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            st.pyplot(fig)

            # Show top at-risk customers
            out_df = test_df.copy()
            out_df['Churn_Probability'] = y_proba
            st.subheader('Top 10 highest predicted churn probabilities')
            st.dataframe(out_df.sort_values('Churn_Probability', ascending=False).head(10))

st.markdown("---")
st.caption("If you want a permanent metrics file, run your training notebook to save metrics as JSON into `models/metrics.json` and this page will load it automatically.")
