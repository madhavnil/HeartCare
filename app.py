
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="HeartCare",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ HeartCare")
st.subheader("Heart Disease Risk Prediction System")
st.write("A patient health information and screening platform.")

st.info(
    "This is an academic screening project, not a medical diagnosis."
)

# Patient details
st.header("Patient Information")

patient_id = st.text_input("Patient ID")
name = st.text_input("Patient Name")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    sex = st.selectbox("Sex", ["Male", "Female"])

with col2:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina",
         "Non-anginal Pain", "Asymptomatic"]
    )
    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)", min_value=0, value=120
    )

col3, col4 = st.columns(2)

with col3:
    cholesterol = st.number_input(
        "Cholesterol (mg/dl)", min_value=0, value=200
    )

with col4:
    max_heart_rate = st.number_input(
        "Maximum Heart Rate", min_value=0, value=150
    )

st.subheader("Additional Health Information")

st_depression = st.number_input(
    "ST Depression (Oldpeak)", min_value=0.0,
    max_value=20.0, value=0.0, step=0.1
)

if st.button("Submit Patient Information"):
    if not patient_id or not name:
        st.warning("Please enter Patient ID and Patient Name.")
    else:
        st.success("Patient information entered successfully.")

        st.write("### Patient Summary")
        st.write("Patient ID:", patient_id)
        st.write("Patient Name:", name)
        st.write("Age:", age)
        st.write("Sex:", sex)
        st.write("Chest Pain:", chest_pain)
        st.write("Resting BP:", resting_bp)
        st.write("Cholesterol:", cholesterol)
        st.write("Maximum Heart Rate:", max_heart_rate)
        st.write("ST Depression:", st_depression)

st.divider()

st.header("Previous Patient Records")

search_id = st.text_input("Enter Patient ID to search previous records")

if st.button("Search Previous Record"):
    st.info(
        "Previous-record database will be connected in the next step."
    )

st.divider()

st.header("Heart Disease Prediction")

st.warning(
    "The machine-learning prediction model will be connected "
    "after the dataset and model are prepared."
)

