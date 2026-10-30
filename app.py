
import streamlit as st
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="HeartCare",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ HeartCare")
st.subheader("Heart Disease Prediction System")

st.write(
    "A machine learning based patient health screening platform."
)

st.info(
    "Educational project only. Predictions are not medical diagnoses."
)

# ---------------------------------
# FILES
# ---------------------------------

DATASET = "heart-disease-data.csv"
RECORDS = "patient_records.csv"

# ---------------------------------
# DATASET FEATURES
# ---------------------------------

features = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

numeric_features = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

categorical_features = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]

# ---------------------------------
# MACHINE LEARNING MODEL
# ---------------------------------

@st.cache_resource
def train_model():

    data = pd.read_csv(DATASET)

    data.columns = (
        data.columns.str.strip().str.lower()
    )

    # Handle missing values represented by ?
    data = data.replace("?", pd.NA)

    if "target" in data.columns:
        target_column = "target"
    elif "num" in data.columns:
        target_column = "num"
    else:
        raise ValueError(
            "Dataset must contain a target or num column."
        )

    missing = [
        col for col in features
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    # Convert all model columns into numeric values
    for col in features + [target_column]:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    data = data[features + [target_column]].dropna()

    X = data[features]

    # In the original Cleveland dataset,
    # values 1-4 indicate presence of disease.
    y = (data[target_column] > 0).astype(int)

    if y.nunique() != 2:
        raise ValueError(
            "Dataset must contain both target classes."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "Not enough examples in each target class."
        )

    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    accuracy = accuracy_score(
        y_test,
        model.predict(X_test)
    )

    # Retrain on all available data for predictions
    model.fit(X, y)

    return model, accuracy


try:
    model, accuracy = train_model()

    st.success("Machine Learning model loaded successfully!")

    st.metric(
        "Test Dataset Accuracy",
        f"{accuracy * 100:.1f}%"
    )

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# ---------------------------------
# PATIENT INFORMATION FORM
# ---------------------------------

st.divider()
st.header("👤 Patient Information")

with st.form("patient_form"):

    patient_id = st.text_input("Patient ID")

    name = st.text_input("Patient Name")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=45
        )

        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

        chest_pain = st.selectbox(
            "Chest Pain Type",
            [
                "Typical Angina",
                "Atypical Angina",
                "Non-anginal Pain",
                "Asymptomatic"
            ]
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=50,
            max_value=300,
            value=120
        )

        cholesterol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=50,
            max_value=700,
            value=200
        )

        fbs_value = st.number_input(
            "Fasting Blood Sugar (mg/dl)",
            min_value=0,
            max_value=600,
            value=100
        )

        st.caption(
            "The model converts this into 0 or 1 using "
            "the dataset threshold of 120 mg/dl."
        )

    with col2:

        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2]
        )

        max_heart_rate = st.number_input(
            "Maximum Heart Rate (bpm)",
            min_value=50,
            max_value=250,
            value=150
        )

        exercise_angina = st.selectbox(
            "Exercise Induced Angina",
            ["No", "Yes"]
        )

        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "Slope of ST Segment",
            [0, 1, 2]
        )

        ca = st.selectbox(
            "Number of Major Vessels (ca)",
            [0, 1, 2, 3, 4]
        )

        thal = st.selectbox(
            "Thal",
            [0, 1, 2, 3]
        )

    submitted = st.form_submit_button(
        "❤️ Predict Heart Disease Risk",
        use_container_width=True
    )


# ---------------------------------
# PREDICTION
# ---------------------------------

if submitted:

    if not patient_id.strip() or not name.strip():

        st.warning(
            "Please enter Patient ID and Patient Name."
        )

    else:

        fbs = 1 if fbs_value > 120 else 0

        patient = pd.DataFrame([{
            "age": age,
            "sex": 1 if sex == "Male" else 0,
            "cp": [
                0, 1, 2, 3
            ][
                [
                    "Typical Angina",
                    "Atypical Angina",
                    "Non-anginal Pain",
                    "Asymptomatic"
                ].index(chest_pain)
            ],
            "trestbps": resting_bp,
            "chol": cholesterol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": max_heart_rate,
            "exang": 1 if exercise_angina == "Yes" else 0,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }])

        prediction = int(model.predict(patient)[0])

        probabilities = model.predict_proba(patient)[0]

        positive_probability = (
            probabilities[list(model.classes_).index(1)] * 100
        )

        st.divider()
        st.header("📊 Prediction Result")

        if prediction == 1:

            st.error(
                "Possible Heart Disease Indicated"
            )

        else:

            st.success(
                "No Heart Disease Indicated by the Model"
            )

        st.metric(
            "Model Estimated Positive Probability",
            f"{positive_probability:.1f}%"
        )

        st.caption(
            "This probability is a model output, not a "
            "clinically validated individual risk estimate."
        )

        st.subheader("Patient Summary")

        record = {
            "Patient ID": patient_id.strip(),
            "Patient Name": name.strip(),
            "Age": age,
            "Sex": sex,
            "Fasting Blood Sugar": fbs_value,
            "Prediction": (
                "Possible Heart Disease"
                if prediction == 1
                else "No Heart Disease Indicated"
            ),
            "Positive Probability (%)": round(
                positive_probability, 1
            )
        }

        st.dataframe(
            pd.DataFrame([record]),
            use_container_width=True
        )

        # Save patient record
        old_records = (
            pd.read_csv(RECORDS)
            if os.path.exists(RECORDS)
            else pd.DataFrame()
        )

        updated_records = pd.concat(
            [old_records, pd.DataFrame([record])],
            ignore_index=True
        )

        updated_records.to_csv(
            RECORDS,
            index=False
        )

        st.success("Patient summary saved successfully!")


# ---------------------------------
# PREVIOUS PATIENT RECORDS
# ---------------------------------

st.divider()
st.header("📁 Previous Patient Records")

search_id = st.text_input(
    "Enter Patient ID to search previous records"
)

if st.button("Search Patient Record"):

    if os.path.exists(RECORDS):

        records = pd.read_csv(RECORDS)

        result = records[
            records["Patient ID"].astype(str)
            == search_id.strip()
        ]

        if not result.empty:

            st.success("Previous patient record found!")

            st.dataframe(
                result,
                use_container_width=True
            )

        else:

            st.warning("No previous record found.")

    else:

        st.info("No patient records saved yet.")


st.divider()

st.caption(
    "HeartCare | Academic Data Science and Machine Learning Project"
)

st.caption(
    "Not intended for diagnosis or treatment decisions. "
    "Use fictional patient information for demonstrations."
)
