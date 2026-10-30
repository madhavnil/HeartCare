
import streamlit as st
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="HeartCare", page_icon="❤️", layout="wide")

st.title("❤️ HeartCare")
st.subheader("Heart Disease Prediction System")
st.info("Educational screening project only. This is not a medical diagnosis.")

# Dataset and model
DATASET = "heart-disease-data.csv"
RECORDS = "patient_records.csv"

features = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

numeric = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical = [x for x in features if x not in numeric]

@st.cache_resource
def train_model():
    data = pd.read_csv(DATASET)
    data.columns = data.columns.str.strip().str.lower()

    if "target" not in data.columns:
        raise ValueError("Dataset must contain a 'target' column.")

    if not all(x in data.columns for x in features):
        raise ValueError("Some required dataset columns are missing.")

    data = data[features + ["target"]].dropna()
    X = data[features]
    y = data["target"].astype(int)

    if y.nunique() != 2:
        raise ValueError("The target column must contain two classes.")

    preprocessor = ColumnTransformer([
        ("numbers", StandardScaler(), numeric),
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    test_model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    test_model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, test_model.predict(X_test))

    model.fit(X, y)
    return model, accuracy

try:
    model, accuracy = train_model()
    st.success("Machine Learning model trained successfully!")
    st.metric("Test-set accuracy", f"{accuracy * 100:.1f}%")
except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    st.stop()

st.divider()
st.header("Patient Information")

with st.form("patient_form"):
    patient_id = st.text_input("Patient ID")
    name = st.text_input("Patient Name")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3],
            format_func=lambda x: [
                "Typical Angina", "Atypical Angina",
                "Non-anginal Pain", "Asymptomatic"
            ][x]
        )
        trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
        chol = st.number_input("Cholesterol", 50, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

    with col2:
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        thalach = st.number_input("Maximum Heart Rate", 50, 250, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)
        slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thal", [0, 1, 2, 3])

    submitted = st.form_submit_button("Predict Heart Disease Risk")

if submitted:
    if not patient_id.strip() or not name.strip():
        st.warning("Please enter Patient ID and Patient Name.")
    else:
        patient = pd.DataFrame([{
            "age": age,
            "sex": 1 if sex == "Male" else 0,
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

        prediction = int(model.predict(patient)[0])
        probability = model.predict_proba(patient)[0]
        risk_probability = probability[list(model.classes_).index(1)] * 100

        if prediction == 1:
            st.error("Prediction: Possible Heart Disease")
        else:
            st.success("Prediction: No Heart Disease indicated by this model")

        st.write(f"Model-estimated positive probability: {risk_probability:.1f}%")
        st.caption("This result is for academic demonstration only. It cannot confirm or rule out disease.")

        record = {
            "Patient ID": patient_id,
            "Patient Name": name,
            "Age": age,
            "Sex": sex,
            "Prediction": "Possible Heart Disease" if prediction == 1 else "No Heart Disease indicated",
            "Model Positive Probability (%)": round(risk_probability, 1)
        }

        old_records = (
            pd.read_csv(RECORDS)
            if os.path.exists(RECORDS)
            else pd.DataFrame()
        )

        old_records = pd.concat(
            [old_records, pd.DataFrame([record])],
            ignore_index=True
        )
        old_records.to_csv(RECORDS, index=False)
        st.success("Patient summary saved.")

st.divider()
st.header("Previous Patient Records")

search_id = st.text_input("Enter Patient ID to find previous summary")

if st.button("Search Previous Record"):
    if os.path.exists(RECORDS):
        records = pd.read_csv(RECORDS)
        result = records[records["Patient ID"].astype(str) == search_id.strip()]
        if not result.empty:
            st.dataframe(result, use_container_width=True)
        else:
            st.warning("No previous record found.")
    else:
        st.info("No patient records have been saved yet.")
