
import streamlit as st
import pandas as pd
from supabase import create_client
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="HeartCare", page_icon="❤️", layout="wide")

st.title("❤️ HeartCare")
st.caption("Heart Disease Prediction and Patient Management System")

# SUPABASE CONNECTION

@st.cache_resource
def get_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )

try:
    supabase = get_supabase()
except Exception as e:
    st.error(f"Database connection error: {e}")
    st.stop()

# STAFF LOGIN

if "access_token" not in st.session_state:
    st.session_state.access_token = None
    st.session_state.refresh_token = None

if not st.session_state.access_token:
    st.subheader("Staff Login")

    with st.form("login"):
        email = st.text_input("Staff Email")
        password = st.text_input("Password", type="password")
        login = st.form_submit_button("Login")

    if login:
        try:
            response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })

            st.session_state.access_token = response.session.access_token
            st.session_state.refresh_token = response.session.refresh_token
            st.rerun()

        except Exception:
            st.error("Login failed. Check your email and password.")

    st.stop()

# RESTORE LOGIN SESSION

try:
    supabase.auth.set_session(
        st.session_state.access_token,
        st.session_state.refresh_token
    )
except Exception:
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.error("Session expired. Please log in again.")
    st.rerun()

if st.sidebar.button("Logout"):
    supabase.auth.sign_out()
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.rerun()

# MACHINE LEARNING MODEL

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

@st.cache_resource
def train_model():
    data = pd.read_csv("heart-disease-data.csv")
    data.columns = data.columns.str.strip().str.lower()
    data = data.replace("?", pd.NA)

    target = "target" if "target" in data.columns else "num"

    if target not in data.columns:
        raise ValueError("Dataset needs a target or num column.")

    for column in FEATURES + [target]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=[target])

    X = data[FEATURES]
    y = (data[target] > 0).astype(int)

    if y.nunique() < 2:
        raise ValueError("Dataset must contain both outcome classes.")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X, y)
    return model

try:
    model = train_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# PATIENT DETAILS

st.subheader("Patient Information")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        patient_id = st.text_input("Patient ID")
        patient_name = st.text_input("Patient Name")
        age = st.number_input("Age", 1, 120, 30)

        sex_label = st.selectbox("Sex", ["Male", "Female"])
        sex = 1 if sex_label == "Male" else 0

        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
        chol = st.number_input("Cholesterol", 50, 700, 200)

        fbs_value = st.number_input(
            "Fasting Blood Sugar (mg/dl)", 30, 500, 100
        )
        fbs = 1 if fbs_value > 120 else 0

    with col2:
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        thalach = st.number_input("Maximum Heart Rate", 40, 250, 150)

        exang_label = st.selectbox(
            "Exercise Induced Angina", ["No", "Yes"]
        )
        exang = 1 if exang_label == "Yes" else 0

        oldpeak = st.number_input(
            "ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1
        )

        slope = st.selectbox("ST Slope", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    submitted = st.form_submit_button("Predict Heart Disease")

# PREDICTION AND SAVE

if submitted:
    if not patient_id.strip() or not patient_name.strip():
        st.error("Enter Patient ID and Patient Name.")
    else:
        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }])

        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1]) * 100

        result = (
            "Positive model prediction"
            if prediction == 1
            else "Negative model prediction"
        )

        if prediction == 1:
            st.warning("The model predicts a possible presence of heart disease.")
        else:
            st.success("The model predicts no heart disease.")

        st.metric("Model Positive Probability", f"{probability:.2f}%")

        try:
            supabase.table("patients").upsert({
                "patient_id": patient_id.strip(),
                "patient_name": patient_name.strip(),
                "sex": sex_label
            }).execute()

            supabase.table("visits").insert({
                "patient_id": patient_id.strip(),
                "age": int(age),
                "chest_pain_type": int(cp),
                "resting_bp": int(trestbps),
                "cholesterol": int(chol),
                "fasting_blood_sugar": int(fbs_value),
                "resting_ecg": int(restecg),
                "max_heart_rate": int(thalach),
                "exercise_angina": int(exang),
                "oldpeak": float(oldpeak),
                "slope": int(slope),
                "ca": int(ca),
                "thal": int(thal),
                "prediction": result,
                "positive_probability": float(probability)
            }).execute()

            st.success("Patient visit saved successfully!")

        except Exception as e:
            st.error(f"Could not save patient: {e}")

# SEARCH PATIENT HISTORY

st.divider()
st.subheader("🔍 Patient Previous History")

search_id = st.text_input("Enter Patient ID to search")

if st.button("Search Patient"):
    if not search_id.strip():
        st.warning("Enter a Patient ID.")
    else:
        try:
            patient_response = (
                supabase.table("patients")
                .select("*")
                .eq("patient_id", search_id.strip())
                .execute()
            )

            if not patient_response.data:
                st.warning("Patient not found.")
            else:
                patient = patient_response.data[0]

                st.write("### Patient Details")
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

                st.write("### Previous Visits")

                if visits_response.data:
                    st.dataframe(
                        pd.DataFrame(visits_response.data),
                        use_container_width=True
                    )
                else:
                    st.info("No previous visits found.")

        except Exception as e:
            st.error(f"History search failed: {e}")

st.divider()
st.caption(
    "Academic demonstration only. This prediction is not a medical diagnosis."
)
