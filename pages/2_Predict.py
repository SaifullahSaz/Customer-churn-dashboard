import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from utils import predict_df

st.title("📥 Upload & Predict — Churn")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Run prediction (utils.predict_df will load model and align features)
    try:
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
