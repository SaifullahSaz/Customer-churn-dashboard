import streamlit as st
import pandas as pd
from utils import fetch_table_from_supabase, preprocess_data, predict_df, upsert_predictions_to_supabase, _get_supabase_client

st.title("Predict — Upload or Load from Supabase")

source = st.radio("Input source", ["Upload CSV", "Load from Supabase"])

if source == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
else:
    # Build simple filter UI — adapt to your table columns
    st.write("Load rows from Supabase")
    table = st.secrets["supabase"].get("table", "predictions")
    contract = st.selectbox("Contract type (optional)", options=["", "Month-to-month", "One year", "Two year"])
    # other filters...
    if st.button("Load from Supabase"):
        try:
            filters = {}
            if contract:
                filters["Contract"] = contract
            df = fetch_table_from_supabase(table, filters=filters, limit=1000)
            if df.empty:
                st.warning("No rows returned for these filters.")
            else:
                st.success(f"Loaded {len(df)} rows from Supabase")
                st.dataframe(df.head(10))
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            df = None

# If df exists, run the same pipeline as the upload path
if 'df' in locals() and df is not None and not df.empty:
    st.info("Preprocessing and scoring...")
    processed = preprocess_data(df.copy())
    result_df = predict_df(df.copy())  # or use model directly on processed, depending on your code
    st.dataframe(result_df.head(20))
    
    # Upsert predictions to Supabase
    if st.button("Save predictions to Supabase"):
        try:
            upsert_predictions_to_supabase(table, result_df)
            st.success("Predictions upserted to Supabase")
        except Exception as e:
            st.error(f"Failed to upsert predictions: {e}")
    
    # Optionally: offer download, etc.

def upsert_predictions_to_supabase(table_name: str, df: pd.DataFrame, key_col: str = "customerID"):
    supabase = _get_supabase_client()
    # Convert to list of dicts
    records = df.to_dict(orient="records")
    # Upsert in chunks
    chunk_size = 500
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i+chunk_size]
        res = supabase.table(table_name).upsert(chunk, on_conflict=key_col).execute()
        if res.error:
            raise Exception(f"Upsert failed: {res.error}")

# Debugging: Test Supabase connection and data fetching
if st.button("Test Supabase connection"):
    try:
        test_df = fetch_table_from_supabase("predictions", filters=None, limit=50)
        st.success("Supabase connection successful")
        st.dataframe(test_df)
    except Exception as e:
        st.error(f"Supabase connection failed: {e}")

