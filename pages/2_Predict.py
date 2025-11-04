import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from utils import predict_df, load_model

st.title("📥 Upload & Predict — Churn")

# Configuration
MAX_UPLOAD_MB = 5
REQUIRED_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

def _validate_uploaded_file(uploaded):
    # size check (Streamlit's UploadedFile has size in bytes)
    try:
        size = uploaded.size
    except Exception:
        size = None
    if size and size > MAX_UPLOAD_MB * 1024 * 1024:
        return False, f"File too large ({size/1024/1024:.1f} MB). Max allowed is {MAX_UPLOAD_MB} MB." 
    return True, None


if uploaded_file:
    ok, msg = _validate_uploaded_file(uploaded_file)
    if not ok:
        st.error(msg)
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.success("File uploaded successfully!")
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Check model availability and feature expectations
    try:
        model_obj, feature_list = load_model()
    except Exception as e:
        st.error(f"Model artifacts not available: {e}")
        st.stop()

    # Basic required column check (best-effort). If feature_list is available we trust alignment logic,
    # but we still ensure key raw columns exist to avoid obvious errors.
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Uploaded CSV is missing required columns: {missing}. Please include these columns and re-upload.")
        st.stop()

    # Run prediction (utils.predict_df will load model and align features)
    try:
        with st.spinner("Running predictions..."):
            results = predict_df(df)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("Prediction Summary")
    churn_count = results['Predicted_Churn'].value_counts()
    st.write(churn_count)

    fig = px.pie(
        names=churn_count.index,
        values=churn_count.values,
        title="Churn Prediction Distribution",
    )
    st.plotly_chart(fig)

    st.subheader("Top At-Risk Customers")
    st.dataframe(results.sort_values("Churn_Probability", ascending=False).head(10))

    # Optional Save to Supabase
    st.subheader("Save results to Supabase")
    if st.button("Upload Predictions"):
        try:
            url = st.secrets["SUPABASE_URL"]
            key = st.secrets["SUPABASE_KEY"]
            supabase: Client = create_client(url, key)
        except Exception as e:
            st.error(f"Supabase credentials missing or invalid: {e}")
            st.stop()

        data_to_upload = []
        for _, row in results.iterrows():
            record = {
                "customer_id": str(row.get("customerID", "")),
                "predicted_churn": float(row.get("Churn_Probability", 0.0)),
                "monthly_charges": float(row.get("MonthlyCharges", 0.0)) if row.get("MonthlyCharges") is not None else None,
                "tenure": int(row.get("tenure", 0)) if row.get("tenure") is not None else None,
            }
            data_to_upload.append(record)

        try:
            supabase.table("predictions").insert(data_to_upload).execute()
            st.success("Predictions uploaded to Supabase.")
        except Exception as e:
            st.error(f"Upload failed: {e}")
