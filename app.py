
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import preprocess_data, load_model
from supabase import create_client, Client


# --- Initialize Supabase ---
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(url, key)

# Checking 
st.write("Supabase URL:", url)

# --- Load Model ---
model = load_model()

# --- Streamlit Layout ---
st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")

st.title("📊 Telecom Customer Churn Prediction Dashboard")
st.write("Upload your customer dataset to predict churn and visualize key insights.")

uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head())

    # Load model and feature list
    model, feature_list = load_model()

    # Preprocess
    processed_df = preprocess_data(df)

    # Align columns to match training
    for col in feature_list:
        if col not in processed_df.columns:
            processed_df[col] = 0  # add missing columns
    processed_df = processed_df[feature_list]  # same order

    # Predict
    predictions = model.predict(processed_df)
    df['Predicted_Churn'] = predictions

    # Display results
    st.subheader("📈 Prediction Summary")
    churn_count = df['Predicted_Churn'].value_counts()
    st.write(churn_count)

    fig = px.pie(
        churn_count, 
        values=churn_count.values, 
        names=['No Churn', 'Churn'], 
        title="Churn Prediction Distribution"
    )
    st.plotly_chart(fig)

    # Save to Supabase
    st.subheader("💾 Save results to Supabase?")
    if st.button("Upload Predictions"):
        data_to_upload = df.to_dict(orient="records")
        supabase.table("predictions").insert(data_to_upload).execute()
        st.success("✅ Predictions uploaded to Supabase!")

    # Optional analysis
    st.subheader("📊 Key Insights")
    if 'MonthlyCharges' in df.columns:
        st.write("Monthly Charges vs Predicted Churn:")
        fig2 = px.histogram(df, x="MonthlyCharges", color="Predicted_Churn", barmode="group")
        st.plotly_chart(fig2)
