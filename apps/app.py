import streamlit as st
import pandas as pd
import joblib

st.title("Customer Churn Prediction App")

model = joblib.load("churn_model.pkl")
scaler = joblib.load("churn_scaler.pkl")
features = joblib.load("features.pkl")

st.sidebar.header("Customer Inputs")

input_data = {}
for col in features:
    input_data[col] = st.sidebar.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])

scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]
prob = model.predict_proba(scaled_input)[0][1]

if prediction == 1:
    st.error(f"⚠️ Customer likely to churn (Probability: {prob:.2f})")
else:
    st.success(f"✅ Customer likely to stay (Probability: {1-prob:.2f})")
