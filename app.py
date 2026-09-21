import streamlit as st
import pandas as pd
from xgboost import XGBClassifier

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = XGBClassifier()
model.load_model("diabetes_risk_model.json")


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diabetes Risk Prediction Dashboard")
st.write(
    "Enter the patient details below to estimate diabetes risk "
    "using the trained Balanced XGBoost model."
)

st.success("✅ Machine Learning Model Loaded Successfully")


# --------------------------------------------------
# PATIENT DETAILS
# --------------------------------------------------

st.divider()
st.header("👤 Patient Details")

col1, col2 = st.columns(2)


# ---------------- LEFT COLUMN ----------------

with col1:

    st.subheader("Health & Lifestyle")

    BMI = st.number_input(
        "BMI",
        min_value=10,
        max_value=80,
        value=25
    )

    Smoker = st.selectbox(
        "Smoker",
        ["No", "Yes"]
    )

    HeartDiseaseorAttack = st.selectbox(
        "Heart Disease or Attack",
        ["No", "Yes"]
    )

    PhysActivity = st.selectbox(
        "Physical Activity",
        ["No", "Yes"]
    )

    Fruits = st.selectbox(
        "Consume Fruits",
        ["No", "Yes"]
    )

    Veggies = st.selectbox(
        "Consume Vegetables",
        ["No", "Yes"]
    )

    HvyAlcoholConsump = st.selectbox(
        "Heavy Alcohol Consumption",
        ["No", "Yes"]
    )

    AnyHealthcare = st.selectbox(
        "Any Healthcare Coverage",
        ["No", "Yes"]
    )

    NoDocbcCost = st.selectbox(
        "Could Not See Doctor Due to Cost",
        ["No", "Yes"]
    )


# ---------------- RIGHT COLUMN ----------------

with col2:

    st.subheader("General & Personal Information")

    GenHlth = st.selectbox(
        "General Health",
        ["Poor", "Fair", "Good", "Very Good", "Excellent"]
    )

    MentHlth = st.number_input(
        "Mental Health - Poor Days (0-30)",
        min_value=0,
        max_value=30,
        value=0
    )

    PhysHlth = st.number_input(
        "Physical Health - Poor Days (0-30)",
        min_value=0,
        max_value=30,
        value=0
    )

    DiffWalk = st.selectbox(
        "Difficulty Walking",
        ["No", "Yes"]
    )

    Sex = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )

    Age = st.selectbox(
        "Age Group",
        [
            "18-24",
            "25-29",
            "30-34",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "55-59",
            "60-64",
            "65-69",
            "70-74",
            "75-79",
            "80+"
        ]
    )

    Education = st.selectbox(
        "Education",
        [
            "Never attended / kindergarten only",
            "Elementary (Grades 1-8)",
            "Some High School (Grades 9-11)",
            "High School Graduate",
            "Some College / Technical School",
            "College Graduate"
        ]
    )

    Income = st.selectbox(
        "Income",
        [
            "Less than $10,000",
            "$10,000 to less than $15,000",
            "$15,000 to less than $20,000",
            "$20,000 to less than $25,000",
            "$25,000 to less than $35,000",
            "$35,000 to less than $50,000",
            "$50,000 to less than $75,000",
            "$75,000 or more"
        ]
    )


# --------------------------------------------------
# INPUT MAPPINGS
# --------------------------------------------------

yes_no = {
    "No": 0,
    "Yes": 1
}

genhlth_map = {
    "Poor": 1,
    "Fair": 2,
    "Good": 3,
    "Very Good": 4,
    "Excellent": 5
}

age_map = {
    "18-24": 1,
    "25-29": 2,
    "30-34": 3,
    "35-39": 4,
    "40-44": 5,
    "45-49": 6,
    "50-54": 7,
    "55-59": 8,
    "60-64": 9,
    "65-69": 10,
    "70-74": 11,
    "75-79": 12,
    "80+": 13
}

education_map = {
    "Never attended / kindergarten only": 1,
    "Elementary (Grades 1-8)": 2,
    "Some High School (Grades 9-11)": 3,
    "High School Graduate": 4,
    "Some College / Technical School": 5,
    "College Graduate": 6
}

income_map = {
    "Less than $10,000": 1,
    "$10,000 to less than $15,000": 2,
    "$15,000 to less than $20,000": 3,
    "$20,000 to less than $25,000": 4,
    "$25,000 to less than $35,000": 5,
    "$35,000 to less than $50,000": 6,
    "$50,000 to less than $75,000": 7,
    "$75,000 or more": 8
}

sex_map = {
    "Female": 0,
    "Male": 1
}


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()

if st.button(
    "🔍 Predict Diabetes Risk",
    type="primary",
    use_container_width=True
):

    # Keep feature names and order exactly as used during training
    input_data = pd.DataFrame([{
        "BMI": BMI,
        "Smoker": yes_no[Smoker],
        "HeartDiseaseorAttack": yes_no[HeartDiseaseorAttack],
        "PhysActivity": yes_no[PhysActivity],
        "Fruits": yes_no[Fruits],
        "Veggies": yes_no[Veggies],
        "HvyAlcoholConsump": yes_no[HvyAlcoholConsump],
        "AnyHealthcare": yes_no[AnyHealthcare],
        "NoDocbcCost": yes_no[NoDocbcCost],
        "GenHlth": genhlth_map[GenHlth],
        "MentHlth": MentHlth,
        "PhysHlth": PhysHlth,
        "DiffWalk": yes_no[DiffWalk],
        "Sex": sex_map[Sex],
        "Age": age_map[Age],
        "Education": education_map[Education],
        "Income": income_map[Income]
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    risk_percent = float(probability * 100)

    # ---------------- RESULTS ----------------

    st.header("📊 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            label="Estimated Diabetes Risk",
            value=f"{risk_percent:.1f}%"
        )

    with result_col2:

        if prediction == 1:
            st.metric(
                label="Model Prediction",
                value="Higher Risk"
            )
        else:
            st.metric(
                label="Model Prediction",
                value="Lower Risk"
            )

    st.write("### Risk Probability")

    st.progress(
        int(round(risk_percent))
    )

    if prediction == 1:

        st.warning(
            "⚠️ The model predicts a higher likelihood "
            "of diabetes risk."
        )

    else:

        st.success(
            "✅ The model predicts a lower likelihood "
            "of diabetes risk."
        )

    st.info(
        "ℹ️ This prediction is generated by a machine-learning "
        "model for educational and research purposes only. "
        "It should not be considered a medical diagnosis."
    )
    # --------------------------------------------------
    # MODEL RISK FACTOR ANALYSIS
    # --------------------------------------------------

    st.divider()
    st.header("📈 Model Risk Factor Analysis")

    feature_names = [
        "BMI",
        "Smoker",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "GenHlth",
        "MentHlth",
        "PhysHlth",
        "DiffWalk",
        "Sex",
        "Age",
        "Education",
        "Income"
    ]

    importance_df = pd.DataFrame({
        "Risk Factor": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(10)

    st.write(
        "The chart below shows the features that were most influential "
        "in the model's predictions across the dataset."
    )

    st.bar_chart(
        importance_df.set_index("Risk Factor")
    )

    st.caption(
        "Feature importance indicates influence on the machine-learning "
        "model's predictions and does not imply that a factor causes diabetes."
    )