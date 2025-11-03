
import streamlit as st

# Minimal home/launcher script. The main prediction UI is moved into a Streamlit
# pages file at `pages/2_Predict.py`. This file provides a friendly entrypoint.

st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")
st.write(
    "This app predicts customer churn. Use the Pages menu (left) to go to 'Upload & Predict'."
)

st.markdown("---")

st.subheader("Quick checks")
try:
    # Try to show model availability without loading heavy objects
    from utils import load_model
    model_info = load_model()
    st.success("Model file found and loadable.")
except Exception as e:
    st.warning(f"Model could not be loaded from disk: {e}")

st.write("If you don't see the 'Upload & Predict' page in the left Pages menu, create a `pages/` folder alongside `app.py` and add page scripts.")

