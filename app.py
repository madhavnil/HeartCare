
import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="HeartCare",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ HeartCare")
st.caption("Heart Disease Prediction and Patient History System")

st.warning(
    "Academic prototype only. This prediction is not a medical "
    "diagnosis. Use fictional patient details for testing."
)

# --------------------------------------------------
# SUPABASE CONNECTION
# --------------------------------------------------

def get_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"Supabase connection error: {e}")
        return None


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "access_token" not in st.session_state:
    st.session_state.access_token = None

if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = None

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

supabase = get_supabase()

if supabase is None:
    st.stop()

if st.session_state.access_token and st.session_state.refresh_token:
    try:
        supabase.auth.set_session(
            st.session_state.access_token,
            st.session_state.refresh_token
        )
        st.session_state.logged_in = True
    except Exception:
        st.session_state.access_token = None
        st.session_state.refresh_token = None
        st.session_state.logged_in = False


# --------------------------------------------------
# STAFF LOGIN
# --------------------------------------------------

if not st.session_state.logged_in:

    st.subheader("Staff Login")

    with st.form("login_form"):
        email = st.text_input("Staff Email")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

    if login_button:
        if not email.strip() or not password:
            st.error("Please enter email and password.")
        else:
            try:
                response = supabase.auth.sign_in_with_password({
                    "email": email.strip(),
                    "password": password
                })

                if response.session is None:
                    st.error("Login failed: No session returned.")
                else:
                    st.session_state.access_token = (
                        response.session.access_token
                    )
                    st.session_state.refresh_token = (
                        response.session.refresh_token
                    )
                    st.session_state.logged_in = True
                    st.success("Login successful!")
                    st.rerun()

            except Exception as e:
                st.error(f"Login error: {e}")

    st.stop()


# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

st.success("Staff logged in successfully.")

if st.button("Logout"):
    try:
        supabase.auth.sign_out()
    except Exception:
        pass

    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.logged_in = False
    st.rerun()


# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

DATASET_PATH = Path(__file__).parent / "heart-disease-data.csv"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]


@st.cache_resource
def train_model():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "heart-disease-data.csv must be in the same folder as app.py."
        )

    data = pd.read_csv(DATASET_PATH)
    data.columns = data.columns.str.strip()

    if "target" in data.columns:
        target_column = "target"
    elif "num" in data.columns:
        target_column = "num"
    else:
        raise ValueError("Dataset needs a target or num column.")

    missing = [c for c in FEATURES if c not in data.columns]

    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    for column in FEATURES + [target_column]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.dropna(subset=[target_column])

    X = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    y = (data[target_column] > 0).astype(int)

    if y.nunique() < 2:
        raise ValueError("Dataset must contain both target classes.")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ])

    model.fit(X, y)
    return model


try:
    model = train_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# --------------------------------------------------
# DROPDOWN LABELS AND THEIR DATASET CODES
# --------------------------------------------------

chest_pain_options = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3
}

slope_options = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

thal_options = {
    "Unknown": 0,
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

ecg_options = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}


# --------------------------------------------------
# PATIENT FORM
# --------------------------------------------------

st.header("Patient Details")

tab1, tab2 = st.tabs([
    "New Patient / Prediction",
    "Previous Patient History"
])

with tab1:

    st.subheader("Patient Information")

    with st.form("patient_form"):

        col1, col2 = st.columns(2)

        with col1:
            patient_id = st.text_input("Patient ID")
            patient_name = st.text_input("Patient Name")

            sex_choice = st.selectbox(
                "Sex",
                ["Male", "Female"]
            )

            age = st.number_input(
                "Age", min_value=1, max_value=120, value=45
            )

            chest_pain_name = st.selectbox(
                "Chest Pain Type",
                list(chest_pain_options.keys())
            )

            resting_bp = st.number_input(
                "Resting Blood Pressure (mmHg)",
                min_value=0, max_value=300, value=120
            )

            cholesterol = st.number_input(
                "Cholesterol (mg/dl)",
                min_value=0, max_value=700, value=200
            )

        with col2:

            fasting_sugar = st.selectbox(
                "Fasting Blood Sugar",
                ["120 mg/dl or below", "Above 120 mg/dl"]
            )

            resting_ecg_name = st.selectbox(
                "Resting ECG",
                list(ecg_options.keys())
            )

            max_heart_rate = st.number_input(
                "Maximum Heart Rate",
                min_value=0, max_value=250, value=150
            )

            exercise_angina = st.selectbox(
                "Exercise-Induced Angina",
                ["No", "Yes"]
            )

            oldpeak = st.number_input(
                "Oldpeak (ST Depression)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1
            )

            slope_name = st.selectbox(
                "ST Segment Slope",
                list(slope_options.keys())
            )

            ca = st.selectbox(
                "Number of Major Vessels",
                [0, 1, 2, 3, 4]
            )

            thal_name = st.selectbox(
                "Thal Result",
                list(thal_options.keys())
            )

        predict_button = st.form_submit_button(
            "Predict and Save Patient Visit",
            use_container_width=True
        )

    if predict_button:

        if not patient_id.strip() or not patient_name.strip():
            st.error("Please enter Patient ID and Patient Name.")

        else:
            try:
                sex_value = 1 if sex_choice == "Male" else 0

                fbs_value = (
                    1 if fasting_sugar == "Above 120 mg/dl" else 0
                )

                exang_value = (
                    1 if exercise_angina == "Yes" else 0
                )

                # Convert displayed names into dataset codes.
                cp_value = chest_pain_options[chest_pain_name]
                ecg_value = ecg_options[resting_ecg_name]
                slope_value = slope_options[slope_name]
                thal_value = thal_options[thal_name]

                patient_values = {
                    "age": age,
                    "sex": sex_value,
                    "cp": cp_value,
                    "trestbps": resting_bp,
                    "chol": cholesterol,
                    "fbs": fbs_value,
                    "restecg": ecg_value,
                    "thalach": max_heart_rate,
                    "exang": exang_value,
                    "oldpeak": oldpeak,
                    "slope": slope_value,
                    "ca": ca,
                    "thal": thal_value
                }

                input_data = pd.DataFrame(
                    [patient_values],
                    columns=FEATURES
                )

                prediction = int(model.predict(input_data)[0])

                probabilities = model.predict_proba(input_data)[0]

                positive_probability = float(
                    probabilities[
                        list(model.classes_).index(1)
                    ]
                )

                if prediction == 1:
                    result = "Positive model prediction"
                    st.error(result)
                else:
                    result = "Negative model prediction"
                    st.success(result)

                st.metric(
                    "Model Positive-Class Score",
                    f"{positive_probability * 100:.2f}%"
                )

                st.caption(
                    "This is the model's output score, not a validated "
                    "clinical probability or diagnosis."
                )

                # Save patient information.
                patient_record = {
                    "patient_id": patient_id.strip(),
                    "patient_name": patient_name.strip(),
                    "sex": sex_choice
                }

                supabase.table("patients").upsert(
                    patient_record
                ).execute()

                # Save every visit separately.
                visit_record = {
                    "patient_id": patient_id.strip(),
                    "age": int(age),
                    "chest_pain_type": cp_value,
                    "resting_bp": int(resting_bp),
                    "cholesterol": int(cholesterol),
                    "fasting_blood_sugar": fbs_value,
                    "resting_ecg": ecg_value,
                    "max_heart_rate": int(max_heart_rate),
                    "exercise_angina": exang_value,
                    "oldpeak": float(oldpeak),
                    "slope": slope_value,
                    "ca": int(ca),
                    "thal": thal_value,
                    "prediction": result,
                    "positive_probability": positive_probability
                }

                supabase.table("visits").insert(
                    visit_record
                ).execute()

                st.success(
                    "Patient details and visit saved successfully!"
                )

            except Exception as e:
                st.error(f"Prediction or saving error: {e}")


# --------------------------------------------------
# PREVIOUS PATIENT HISTORY
# --------------------------------------------------

with tab2:

    st.subheader("Search Previous Patient")

    search_id = st.text_input(
        "Enter Patient ID",
        key="history_patient_id"
    )

    if st.button("Search Patient History"):

        if not search_id.strip():
            st.error("Please enter a Patient ID.")

        else:
            try:
                patient_response = (
                    supabase.table("patients")
                    .select("*")
                    .eq("patient_id", search_id.strip())
                    .execute()
                )

                if not patient_response.data:
                    st.warning("No patient found.")

                else:
                    patient = patient_response.data[0]

                    st.markdown("### Patient Summary")
                    st.write("Patient ID:", patient["patient_id"])
                    st.write("Patient Name:", patient["patient_name"])
                    st.write("Sex:", patient["sex"])

                    visits_response = (
                        supabase.table("visits")
                        .select("*")
                        .eq("patient_id", search_id.strip())
                        .order("visit_date", desc=True)
                        .execute()
                    )

                    st.markdown("### Previous Visit Records")

                    if visits_response.data:
                        history = pd.DataFrame(visits_response.data)
                        st.dataframe(
                            history,
                            use_container_width=True
                        )
                    else:
                        st.info("No previous visits found.")

            except Exception as e:
                st.error(f"History search error: {e}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption("HeartCare | Academic Heart Disease Prediction Project")
