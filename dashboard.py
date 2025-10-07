import streamlit as st
import pandas as pd
from supabase import create_client
import joblib

# Load dataset
# Describe properties, size, dimension, datatype, distribution of the dataset
import pandas as pd

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Convert 'TotalCharges' to numeric, forcing errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


print("Dataset Properties:")
print(f"Dataset shape: {df.shape}") # rows, columns
print(f"Dataset size: {df.size}") # Total number of elements
print(f"Dataset datatypes: {df.dtypes}") # Data types of each column
print(f"Dataset info: {df.info()}") # General information about the dataset
print(f"Dataset summary: {df.describe()}") # Summary statistics for numerical columns

# Check for missing values and duplicates
null_count_name = df['TotalCharges'].isnull().sum()
print(f"Number of null values in 'TotalCharges' column: {null_count_name}")

#Drop rows with null values in 'TotalCharges' column
df = df.dropna(subset=['TotalCharges'])

# Check for missing values and duplicates

missing = df.isnull().sum().sum()
print(f"missing values: {missing}") # Total number of missing values

duplicaates = df.duplicated().sum()
print(f"duplicates: {duplicaates}") # Total number of duplicate rows

# Count of unique values per column
print(df.nunique())

# Display unique values for each column (optional)
for col in df.columns:
    print(f"{col}: {df[col].unique()}")

# Save the cleaned dataset
df.to_csv('cleaned_telco_customer_churn.csv', index=False)

# Save the model using joblib
import joblib

model = "cleaned_telco_customer_churn.csv" # Replace with your actual model object"

joblib.dump(model, "model.pkl")


# Connect to Supabase
url = "https://azysqaiasoalzkywmffa.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF6eXNxYWlhc29hbHpreXdtZmZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg4NDQyNDIsImV4cCI6MjA3NDQyMDI0Mn0.VS2oU5kqjoZN07Prv8XVk2kHK47uat9Wo3l8ELL3dow"
supabase = create_client(url, key)

st.title("Customer Churn Prediction Dashboard")

model = joblib.load("model.pkl")
st.write(model)


# Upload new dataset
uploaded_file = st.file_uploader("Upload your churn dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", df.head())
    # Optionally insert into Supabase
    #supabase.table("customers").insert(df.to_dict(orient="records")).execute()

# Fetch from Supabase
if st.button("Load from Supabase"):
    data = supabase.table("TelcoCustDataset").select("*").execute()
    df = pd.DataFrame(data.data)
    st.write(df.head())
