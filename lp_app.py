import streamlit as st
import numpy as np
from joblib import load

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# -----------------------------
# Load Model & Scaler
# -----------------------------
@st.cache_resource
def load_files():
    model = load("loan_model.joblib")
    scaler = load("loan_scaler.joblib")
    return model, scaler

model, scaler = load_files()

# -----------------------------
# Title
# -----------------------------
st.title("🏦 Loan Approval Prediction System")
st.markdown("Machine Learning Based Loan Approval System")

st.divider()

# -----------------------------
# Input Section
# -----------------------------
st.header("📋 Applicant Details")

Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)

LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0)
Loan_Amount_Term = st.selectbox("Loan Term (months)", [360, 240, 180, 120, 60])
Credit_History = st.selectbox("Credit History", [1.0, 0.0])

# -----------------------------
# Encoding Function
# -----------------------------
def encode_input():
    
    gender = 1 if Gender == "Male" else 0
    married = 1 if Married == "Yes" else 0
    edu = 0 if Education == "Graduate" else 1
    self_emp = 1 if Self_Employed == "Yes" else 0

    if Dependents == "3+":
        dep = 3
    else:
        dep = int(Dependents)

    if Property_Area == "Urban":
        prop = 2
    elif Property_Area == "Semiurban":
        prop = 1
    else:
        prop = 0

    features = [
        gender,
        married,
        dep,
        edu,
        self_emp,
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        prop
    ]

    return np.array(features).reshape(1, -1)

# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button("🔍 Check Loan Status", use_container_width=True):

    # Basic Validation
    if ApplicantIncome == 0 and CoapplicantIncome == 0:
        st.error("❌ Income cannot be zero.")
    
    elif LoanAmount == 0:
        st.error("❌ Loan amount must be greater than zero.")
    
    else:
        input_data = encode_input()
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        st.divider()

        if prediction[0] == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("Developed by Diya | ML Internship Project")
