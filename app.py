import streamlit as st
import pandas as pd
import pickle

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# =====================================
# Load Model
# =====================================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =====================================
# Sidebar
# =====================================
st.sidebar.title("🩺 Patient Details")
st.sidebar.markdown("### Enter Patient Information")

age = st.sidebar.number_input("Age", 1, 120, 30)

hypertension = st.sidebar.selectbox(
    "Hypertension",
    [0, 1],
    format_func=lambda x: "Yes" if x else "No"
)

heart_disease = st.sidebar.selectbox(
    "Heart Disease",
    [0, 1],
    format_func=lambda x: "Yes" if x else "No"
)

bmi = st.sidebar.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=25.0
)

hba1c = st.sidebar.number_input(
    "HbA1c Level",
    min_value=3.0,
    max_value=15.0,
    value=5.5
)

glucose = st.sidebar.number_input(
    "Blood Glucose Level",
    min_value=50,
    max_value=400,
    value=100
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

smoking = st.sidebar.selectbox(
    "Smoking History",
    [
        "never",
        "former",
        "current",
        "ever",
        "not current"
    ]
)

predict = st.sidebar.button("🔍 Predict")

# =====================================
# Home Page
# =====================================

st.title("🩺 Diabetes Prediction System")
st.subheader("Machine Learning-based Early Diabetes Risk Assessment")

st.markdown("---")

st.header("📖 About Diabetes")

st.write("""
Diabetes is a chronic disease that occurs when the body either doesn't produce enough insulin or cannot effectively use the insulin it produces.

Early diagnosis can significantly reduce the risk of severe complications including kidney disease, heart disease, stroke, and vision problems.
""")

colA, colB = st.columns(2)

with colA:

    st.subheader("⚠️ Common Symptoms")

    st.markdown("""
- Frequent urination
- Increased thirst
- Increased hunger
- Fatigue
- Blurred vision
- Slow wound healing
""")

with colB:

    st.subheader("🥗 Prevention Tips")

    st.markdown("""
- Eat a healthy diet
- Exercise regularly
- Maintain a healthy weight
- Quit smoking
- Drink enough water
- Regular health checkups
""")

st.markdown("---")

# =====================================
# Current Input Dashboard
# =====================================

st.header("📊 Current Patient Data")

c1, c2, c3 = st.columns(3)

c1.metric("Age", age)
c2.metric("BMI", bmi)
c3.metric("Blood Glucose", glucose)

c1.metric("HbA1c", hba1c)
c2.metric("Hypertension", "Yes" if hypertension else "No")
c3.metric("Heart Disease", "Yes" if heart_disease else "No")

st.markdown("---")

# =====================================
# Prediction
# =====================================

if predict:

    gender_male = 1 if gender == "Male" else 0
    gender_other = 1 if gender == "Other" else 0

    smoking_current = 1 if smoking == "current" else 0
    smoking_ever = 1 if smoking == "ever" else 0
    smoking_former = 1 if smoking == "former" else 0
    smoking_never = 1 if smoking == "never" else 0
    smoking_not_current = 1 if smoking == "not current" else 0

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
    probability = model.predict_proba(input_scaled)[0][1]

    st.header("🩺 Prediction Result")

    if prediction == 1:

        st.error("⚠️ High Risk of Diabetes")

    else:

        st.success("✅ Low Risk of Diabetes")

    st.subheader("Risk Probability")

    st.progress(float(probability))

    st.write(f"### {probability*100:.2f}%")

    if probability < 0.30:

        st.success("🟢 Low Risk")

    elif probability < 0.70:

        st.warning("🟡 Moderate Risk")

    else:

        st.error("🔴 High Risk")

    st.markdown("---")

    st.subheader("📋 Patient Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Age",
            "Gender",
            "BMI",
            "Blood Glucose",
            "HbA1c",
            "Smoking History"
        ],
        "Value": [
            age,
            gender,
            bmi,
            glucose,
            hba1c,
            smoking
        ]
    })

    st.table(summary)

    st.markdown("---")

    st.subheader("💡 Lifestyle Recommendations")

    if prediction == 1:

        st.info("""
✔ Follow a balanced diet.

✔ Exercise at least 30 minutes daily.

✔ Reduce sugar intake.

✔ Maintain a healthy weight.

✔ Stay hydrated.

✔ Schedule regular medical checkups.
""")

    else:

        st.success("""
✔ Continue a healthy lifestyle.

✔ Maintain regular exercise.

✔ Eat a balanced diet.

✔ Monitor blood glucose periodically.
""")

st.markdown("---")

with st.expander("🤖 Model Information"):

    st.write("""
**Model Used:** Gradient Boosting Classifier

**Machine Learning Library:** Scikit-learn

**Input Features:** 13

This model predicts the likelihood of diabetes using patient demographic and clinical data.
""")

st.markdown("---")

st.caption(
"""
Developed using ❤️ Streamlit | Scikit-learn | Gradient Boosting Classifier
"""
)
