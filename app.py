
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

    # After preprocessing uploaded data:
    pred_probs = model.predict_proba(processed_df)[:, 1]  # probability of churn
    df["Churn_Probability"] = pred_probs

    # Optional threshold for reference
    df["Churn_Binary"] = (df["Churn_Probability"] >= 0.5).astype(int)
    df["Predicted_Churn"] = df["Churn_Binary"].map({0: "No Churn", 1: "Churn"})

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
        # Ensure model produces churn probabilities
        try:
            churn_probabilities = model.predict_proba(processed_df)[:, 1]  # probability of churn = class 1
            df["Churn_Probability"] = churn_probabilities
        except Exception as e:
            st.error(f"⚠️ Could not compute churn probabilities: {e}")
            st.stop()

        # Map only relevant columns for Supabase upload
        data_to_upload = []
        for _, row in df.iterrows():
            record = {
                "customer_id": str(row.get("customerID", "")),  # adjust column name if different
                "predicted_churn": float(row.get("Churn_Probability", 0.0)),  # now probabilistic
                "monthly_charges": float(row.get("MonthlyCharges", 0.0)),
                "tenure": int(row.get("tenure", 0)),
            }
            data_to_upload.append(record)

        try:
            supabase.table("predictions").insert(data_to_upload).execute()
            st.success("✅ Probabilistic churn predictions uploaded to Supabase!")
        except Exception as e:
            st.error(f"⚠️ Supabase upload failed: {e}")

    # Optional analysis
    st.subheader("📊 Key Insights")
    if 'MonthlyCharges' in df.columns:
        st.write("Monthly Charges vs Predicted Churn:")
        fig2 = px.histogram(df, x="MonthlyCharges", color="Predicted_Churn", barmode="group")
        st.plotly_chart(fig2)

    import matplotlib.pyplot as plt

    st.subheader("📈 Churn Probability Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["Churn_Probability"], bins=20, color="skyblue", edgecolor="black")
    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)

    st.subheader("⚠️ Top At-Risk Customers")
    st.dataframe(df.sort_values("Churn_Probability", ascending=False).head(10))

