import streamlit as st
import pandas as pd
from xgboost import XGBClassifier


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)


# -------------------------------------------------
# LOAD TRAINED MODEL
# -------------------------------------------------

model = XGBClassifier()
model.load_model("diabetes_risk_model.json")


# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

if "started" not in st.session_state:
    st.session_state.started = False


# -------------------------------------------------
# FUNCTION TO CLEAR FORM VALUES
# -------------------------------------------------

def clear_form_data():
    keys_to_clear = [
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

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


# -------------------------------------------------
# WELCOME PAGE
# -------------------------------------------------

if not st.session_state.started:

    st.title("🩺 Diabetes Risk Prediction")

    st.subheader(
        "Machine Learning-Based Diabetes Risk Prediction "
        "and Risk Factor Analysis"
    )

    st.write(
        """
        Welcome to the **Diabetes Risk Prediction System**.

        This application uses machine learning to estimate diabetes risk
        based on health, lifestyle, and demographic information.

        The system uses a **Balanced XGBoost Model** to identify diabetes
        risk and also provides an analysis of important risk factors.
        """
    )

    st.write("### 🔍 What does this system provide?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "📋 **Health Assessment**\n\n"
            "Enter health, lifestyle, and personal information."
        )

    with col2:
        st.info(
            "🤖 **ML Risk Prediction**\n\n"
            "Estimate diabetes risk using the trained Balanced XGBoost model."
        )

    with col3:
        st.info(
            "📊 **Risk Factor Analysis**\n\n"
            "View the factors that are influential in the model's predictions."
        )

    st.write("")

    if st.button(
        "🚀 Get Started",
        type="primary",
        use_container_width=True
    ):

        # Clear previous patient values
        clear_form_data()

        # Open dashboard
        st.session_state.started = True

        st.rerun()

    st.caption(
        "This application is developed for educational and research purposes."
    )

    st.stop()


# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------

if st.button("← Back to Home"):

    # Clear previous patient values
    clear_form_data()

    # Return to welcome page
    st.session_state.started = False

    st.rerun()


st.title("🩺 Diabetes Risk Prediction Dashboard")

st.write(
    "Enter the patient details below to estimate diabetes risk "
    "using the trained Balanced XGBoost model."
)

st.success("✅ Machine Learning Model Loaded Successfully")

st.divider()


# -------------------------------------------------
# PATIENT DETAILS
# -------------------------------------------------

st.header("👤 Patient Details")

col1, col2 = st.columns(2)


# -------------------------------------------------
# LEFT COLUMN - HEALTH & LIFESTYLE
# -------------------------------------------------

with col1:

    st.subheader("🏃 Health & Lifestyle")

    BMI = st.number_input(
        "BMI",
        min_value=0,
        max_value=80,
        value=0,
        key="BMI"
    )

    Smoker = st.selectbox(
        "Smoker",
        ["No", "Yes"],
        key="Smoker"
    )

    HeartDiseaseorAttack = st.selectbox(
        "Heart Disease or Attack",
        ["No", "Yes"],
        key="HeartDiseaseorAttack"
    )

    PhysActivity = st.selectbox(
        "Physical Activity",
        ["Yes", "No"],
        key="PhysActivity"
    )

    Fruits = st.selectbox(
        "Consume Fruits",
        ["Yes", "No"],
        key="Fruits"
    )

    Veggies = st.selectbox(
        "Consume Vegetables",
        ["Yes", "No"],
        key="Veggies"
    )

    HvyAlcoholConsump = st.selectbox(
        "Heavy Alcohol Consumption",
        ["No", "Yes"],
        key="HvyAlcoholConsump"
    )

    AnyHealthcare = st.selectbox(
        "Any Healthcare Coverage",
        ["Yes", "No"],
        key="AnyHealthcare"
    )

    NoDocbcCost = st.selectbox(
        "Could Not See Doctor Due to Cost",
        ["No", "Yes"],
        key="NoDocbcCost"
    )


# -------------------------------------------------
# RIGHT COLUMN - GENERAL & PERSONAL INFORMATION
# -------------------------------------------------

with col2:

    st.subheader("👤 General & Personal Information")

    GenHlth = st.selectbox(
        "General Health",
        [
            "Excellent",
            "Very Good",
            "Good",
            "Fair",
            "Poor"
        ],
        key="GenHlth"
    )

    MentHlth = st.number_input(
        "Mental Health - Poor Days",
        min_value=0,
        max_value=30,
        value=0,
        key="MentHlth"
    )

    PhysHlth = st.number_input(
        "Physical Health - Poor Days",
        min_value=0,
        max_value=30,
        value=0,
        key="PhysHlth"
    )

    DiffWalk = st.selectbox(
        "Difficulty Walking",
        ["No", "Yes"],
        key="DiffWalk"
    )

    Sex = st.selectbox(
        "Sex",
        ["Female", "Male"],
        key="Sex"
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
        ],
        key="Age"
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
        ],
        key="Education"
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
        ],
        key="Income"
    )


# -------------------------------------------------
# VALUE MAPPINGS
# -------------------------------------------------

yes_no = {
    "No": 0,
    "Yes": 1
}

sex_map = {
    "Female": 0,
    "Male": 1
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


# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

st.divider()

if st.button(
    "🔍 Predict Diabetes Risk",
    type="primary",
    use_container_width=True
):

    # ---------------------------------------------
    # INPUT VALIDATION
    # ---------------------------------------------

    if BMI == 0:

        st.warning(
            "⚠️ Please enter BMI before making a prediction."
        )

        st.stop()


    # ---------------------------------------------
    # PREPARE INPUT DATA
    # ---------------------------------------------

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


    # ---------------------------------------------
    # MODEL PREDICTION
    # ---------------------------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    risk_percent = float(probability * 100)


    # ---------------------------------------------
    # DISPLAY RESULT
    # ---------------------------------------------

    st.header("📊 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Estimated Diabetes Risk",
            f"{risk_percent:.1f}%"
        )

    with result_col2:

        if prediction == 1:

            st.metric(
                "Model Prediction",
                "Higher Risk"
            )

        else:

            st.metric(
                "Model Prediction",
                "Lower Risk"
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


    # ---------------------------------------------
    # RISK FACTOR ANALYSIS
    # ---------------------------------------------

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


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.caption(
    "🩺 Diabetes Risk Prediction System | "
    "Machine Learning-Based Final Year Project"
)
