import streamlit as st
import pandas as pd
import plotly.express as px
from utils import predict_df

st.title("📊 Analysis")
st.write("Filter predictions and explore key metrics.")

uploaded_file = st.file_uploader("(Optional) Upload CSV with predictions (or upload raw CSV and let app predict)", type=["csv"] , key="analysis_upload")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # If predictions aren't present, attempt to predict
    if 'Churn_Probability' not in df.columns:
        try:
            df = predict_df(df)
        except Exception as e:
            st.error(f"Could not produce predictions: {e}")
            st.stop()

    st.sidebar.header("Filters")
    min_tenure, max_tenure = int(df['tenure'].min()), int(df['tenure'].max())
    tenure_range = st.sidebar.slider("Tenure range", min_value=min_tenure, max_value=max_tenure, value=(min_tenure, max_tenure))

    charges_min, charges_max = float(df['MonthlyCharges'].min()), float(df['MonthlyCharges'].max())
    charges_range = st.sidebar.slider("Monthly Charges", min_value=charges_min, max_value=charges_max, value=(charges_min, charges_max))

    filtered = df[(df['tenure'] >= tenure_range[0]) & (df['tenure'] <= tenure_range[1]) & (df['MonthlyCharges'] >= charges_range[0]) & (df['MonthlyCharges'] <= charges_range[1])]

    st.subheader("Churn Probability Distribution")
    fig = px.histogram(filtered, x='Churn_Probability', nbins=30, title='Churn Probability Distribution')
    st.plotly_chart(fig)

    st.subheader("Churn by Monthly Charges")
    fig2 = px.box(filtered, x='Predicted_Churn', y='MonthlyCharges', title='Monthly Charges by Predicted Churn')
    st.plotly_chart(fig2)

    st.subheader("Top At-Risk Customers")
    st.dataframe(filtered.sort_values('Churn_Probability', ascending=False).head(20))
else:
    st.info("Upload a CSV (with or without predictions) to explore analysis. If your CSV lacks predictions, the app will compute them.")
