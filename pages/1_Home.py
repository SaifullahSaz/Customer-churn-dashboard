import streamlit as st

st.title("🏠 Home — Customer Churn Dashboard")
st.write("Welcome to the Customer Churn Prediction dashboard.")
st.write("Use the Pages menu (left) to go to 'Upload & Predict' to upload a CSV and run predictions.")
st.markdown("---")
st.write("This project keeps preprocessing and model-loading logic in `utils.py`. The prediction page will call `utils.predict_df()` to run predictions.")
