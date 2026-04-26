import streamlit as st
import pickle
import numpy as np

# Load model

model = pickle.load(open("model.pkl", "rb"))

# Try loading scaler

try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
except:
    scaler = None

st.title("Customer Churn Prediction")

# ---- INPUTS ----

credit_score = st.number_input("Credit Score")
age = st.number_input("Age")
tenure = st.number_input("Tenure")
balance = st.number_input("Balance")
products = st.number_input("Number of Products")
has_card = st.selectbox("Has Credit Card", ["No", "Yes"])
active = st.selectbox("Is Active Member", ["No", "Yes"])
salary = st.number_input("Estimated Salary")

# Convert categorical

has_card = 1 if has_card == "Yes" else 0
active = 1 if active == "Yes" else 0

# ---- PREDICT ----

if st.button("Predict"):
    data = np.array([[credit_score, age, tenure, balance,
    products, has_card, active, salary]])

if scaler:
    data = scaler.transform(data)

result = model.predict(data)

if result[0] == 1:
    st.error("Customer likely to leave ❌")
else:
    st.success("Customer will stay ✅")
