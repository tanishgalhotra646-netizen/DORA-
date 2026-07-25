import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ============================
# Sidebar
# ============================

st.sidebar.title("🩺 Patient Information")
st.sidebar.markdown("Enter the patient's health details.")

age = st.sidebar.number_input("Age", 1, 120, 30)
hypertension = st.sidebar.selectbox("Hypertension", [0,1], format_func=lambda x:"Yes" if x else "No")
heart_disease = st.sidebar.selectbox("Heart Disease", [0,1], format_func=lambda x:"Yes" if x else "No")
bmi = st.sidebar.number_input("BMI", 10.0, 70.0, 25.0)
hba1c = st.sidebar.number_input("HbA1c Level", 3.0, 15.0, 5.5)
glucose = st.sidebar.number_input("Blood Glucose Level", 50, 400, 100)

gender = st.sidebar.selectbox(
    "Gender",
    ["Female","Male","Other"]
)

smoking = st.sidebar.selectbox(
    "Smoking History",
    ["never","former","current","ever","not current"]
)

predict = st.sidebar.button("🔍 Predict")

# ============================
# Main Page
# ============================

st.title("🩺 Diabetes Prediction System")

st.image(
    "https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=1200",
    use_container_width=True
)

st.header("📖 What is Diabetes?")

st.write("""
Diabetes is a chronic condition that affects how your body converts food into energy.

When you have diabetes, your body either:
- Doesn't produce enough insulin.
- Cannot effectively use the insulin it produces.

This results in elevated blood glucose levels, which can lead to serious health complications if not managed properly.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚠️ Common Symptoms")
    st.markdown("""
- Frequent urination
- Increased thirst
- Fatigue
- Blurred vision
- Slow wound healing
- Weight loss
""")

with col2:
    st.subheader("💡 Prevention Tips")
    st.markdown("""
- Eat a balanced diet
- Exercise regularly
- Maintain a healthy weight
- Avoid smoking
- Monitor blood sugar levels
- Get regular medical checkups
""")

st.divider()

# ============================
# Prediction
# ============================

if predict:

    gender_male = 1 if gender=="Male" else 0
    gender_other = 1 if gender=="Other" else 0

    smoking_current = 1 if smoking=="current" else 0
    smoking_ever = 1 if smoking=="ever" else 0
    smoking_former = 1 if smoking=="former" else 0
    smoking_never = 1 if smoking=="never" else 0
    smoking_not_current = 1 if smoking=="not current" else 0

    data = pd.DataFrame([[
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

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    st.header("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High likelihood of Diabetes")
    else:
        st.success("✅ Low likelihood of Diabetes")

    st.metric(
        "Probability of Diabetes",
        f"{probability*100:.2f}%"
    )

st.divider()

st.caption("Developed using Streamlit • Gradient Boosting Classifier • Scikit-learn")
