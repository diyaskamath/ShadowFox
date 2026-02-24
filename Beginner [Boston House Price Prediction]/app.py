import streamlit as st
import numpy as np
from joblib import load

model = load("boston_house_model.joblib")

st.title("Boston House Price Prediction")

st.write("Enter the house details below:")

CRIM = st.number_input("Crime Rate (CRIM)", value=3.6)
ZN = st.number_input("Residential Zoning (ZN)", value=11.3)
INDUS = st.number_input("Industrial Area (INDUS)", value=11.1)
CHAS = st.selectbox("Near Charles River (CHAS)", [0, 1])
NOX = st.number_input("NOX Level", value=0.55)
RM = st.number_input("Average Rooms (RM)", value=6.2)
AGE = st.number_input("House Age (AGE)", value=68.5)
DIS = st.number_input("Distance to Employment (DIS)", value=3.8)
RAD = st.number_input("Highway Access (RAD)", value=9)
TAX = st.number_input("Tax Rate (TAX)", value=408)
PTRATIO = st.number_input("Pupil Teacher Ratio", value=18.4)
B = st.number_input("B value", value=356.6)
LSTAT = st.number_input("Lower Status %", value=12.6)

if st.button("Predict Price"):
    data = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS,
                      RAD, TAX, PTRATIO, B, LSTAT]])
    prediction = model.predict(data)
    st.success(f"Predicted House Price: ${prediction[0]:.2f}K")
