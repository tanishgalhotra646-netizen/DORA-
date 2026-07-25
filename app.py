import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Load Model and Scaler
# ----------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient's details below to predict diabetes.")

# ----------------------------
# User Inputs
# ----------------------------

age = st.number_input("Age", min_value=1, max_value=120, value=30)

hypertension = st.selectbox("Hypertension", [0, 1])

heart_disease = st.selectbox("Heart Disease", [0, 1])

bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)

hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)

glucose = st.number_input(
    "Blood Glucose Level",
    min_value=50,
    max_value=400,
    value=100
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

smoking = st.selectbox(
    "Smoking History",
    [
        "never",
        "former",
        "current",
        "ever",
        "not current"
    ]
)

# ----------------------------
# One-Hot Encoding
# ----------------------------

gender_male = 1 if gender == "Male" else 0
gender_other = 1 if gender == "Other" else 0

smoking_current = 1 if smoking == "current" else 0
smoking_ever = 1 if smoking == "ever" else 0
smoking_former = 1 if smoking == "former" else 0
smoking_never = 1 if smoking == "never" else 0
smoking_not_current = 1 if smoking == "not current" else 0

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict"):

    input_data = pd.DataFrame([[
        age,
        hypertension,
        heart_disease,
        bmi,
        hba1c,
        glucose,
        gender_male,
        gender_other,
        smoking_current,
        smoking_ever,
        smoking_former,
        smoking_never,
        smoking_not_current
    ]], columns=[
        "Age",
        "Hypertension",
        "HeartDisease",
        "BMI",
        "HbA1cLevel",
        "BloodGlucoseLevel",
        "Gender_Male",
        "Gender_Other",
        "SmokingHistory_current",
        "SmokingHistory_ever",
        "SmokingHistory_former",
        "SmokingHistory_never",
        "SmokingHistory_not current"
    ])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ The model predicts that the patient is likely to have Diabetes.")
    else:
        st.success("✅ The model predicts that the patient is unlikely to have Diabetes.")