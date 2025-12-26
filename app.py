import streamlit as st
import pandas as pd
from joblib import load

model = load("outputs/loan_pipeline.pkl")

st.title("🏦 Home Loan Approval Prediction")

raw_input = pd.DataFrame([{
    "Gender": st.selectbox("Gender", ["Male", "Female"]),
    "Married": st.selectbox("Married", ["Yes", "No"]),
    "Dependents": st.selectbox("Dependents", [0, 1, 2, 3]),
    "Education": st.selectbox("Education", ["Graduate", "Not Graduate"]),
    "Self_Employed": st.selectbox("Self Employed", ["Yes", "No"]),
    "ApplicantIncome": st.number_input("Applicant Income", min_value=0, value=15000),
    "CoapplicantIncome": st.number_input("Coapplicant Income", min_value=0, value=5000),
    "LoanAmount": st.number_input("Loan Amount (in thousands)", min_value=0, value=75),
    "Loan_Amount_Term": st.number_input("Loan Amount Term", min_value=0, value=360),
    "Credit_History": st.selectbox("Credit History", [1.0, 0.0]),
    "Property_Area": st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
}])

if st.button("Predict"):
    prob = model.predict_proba(raw_input)[0][1]

    if prob >= 0.50:
        st.success(f"Approved ✅ (Prob: {prob:.2f})")
    else:
        st.error(f"Rejected ❌ (Prob: {prob:.2f})")
