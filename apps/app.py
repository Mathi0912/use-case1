import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📉 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn")
np.random.seed(42)

train_df = pd.DataFrame({
    "Login_Frequency": np.random.randint(1, 50, 500),
    "Session_Duration_Avg": np.random.uniform(1, 60, 500),
    "Cart_Abandonment_Rate": np.random.uniform(0, 100, 500),
    "Customer_Service_Calls": np.random.randint(0, 10, 500)
})

train_df["Churned"] = (
    (train_df["Login_Frequency"] < 10) &
    (train_df["Session_Duration_Avg"] < 10) &
    (train_df["Cart_Abandonment_Rate"] > 60)
).astype(int)
X = train_df.drop("Churned", axis=1)
y = train_df["Churned"]
model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
model.fit(X, y)
st.sidebar.header("Customer Inputs")

login = st.sidebar.number_input("Login Frequency", 0, 100, 10)
session = st.sidebar.number_input("Avg Session Duration (mins)", 0.0, 120.0, 15.0)
cart = st.sidebar.slider("Cart Abandonment Rate (%)", 0.0, 100.0, 30.0)
support = st.sidebar.number_input("Customer Service Calls", 0, 20, 1)

input_df = pd.DataFrame({
    "Login_Frequency": [login],
    "Session_Duration_Avg": [session],
    "Cart_Abandonment_Rate": [cart],
    "Customer_Service_Calls": [support]
})
if st.button("Predict Churn"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠️ Likely to churn (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Likely to stay (Probability: {1 - prob:.2f})")
