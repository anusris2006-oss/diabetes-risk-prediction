import streamlit as st
import pandas as pd
from xgboost import XGBClassifier


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="wide"
)


# =========================================================
# LOAD MODELS
# =========================================================

lifestyle_model = XGBClassifier()
lifestyle_model.load_model("diabetes_lifestyle_model.json")

symptom_model = XGBClassifier()
symptom_model.load_model("diabetes_symptom_model.json")


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"


# =========================================================
# NAVIGATION
# =========================================================

def start_app():
    st.session_state.clear()
    st.session_state.page = "dashboard"


def go_home():
    st.session_state.clear()
    st.session_state.page = "home"


# =========================================================
# HOME PAGE
# =========================================================

if st.session_state.page == "home":

    st.title("Diabetes Risk Prediction System")

    st.subheader(
        "Non-Invasive Lifestyle and Symptom-Based "
        "Diabetes Screening"
    )

    st.write(
        """
        This system estimates diabetes risk without requiring
        blood glucose measurements.

        It combines two machine-learning models:

        1. A lifestyle-based risk assessment model trained using
        a large health and lifestyle dataset.

        2. A symptom-based screening model trained using early
        diabetes symptom information.

        The system also provides general preventive health guidance.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "**Lifestyle Risk Assessment**\n\n"
            "Evaluates lifestyle, health and demographic factors."
        )

    with col2:
        st.info(
            "**Symptom Screening**\n\n"
            "Evaluates common early symptoms associated with diabetes."
        )

    with col3:
        st.info(
            "**Health Guidance**\n\n"
            "Provides general preventive lifestyle suggestions."
        )

    st.write("")

    st.button(
        "Get Started",
        type="primary",
        use_container_width=True,
        on_click=start_app
    )

    st.caption(
        "Developed for educational and research purposes."
    )

    st.stop()


# =========================================================
# DASHBOARD
# =========================================================

st.button(
    "Back to Home",
    on_click=go_home
)

st.title("Diabetes Risk Prediction Dashboard")

st.write(
    "Complete both sections below to generate a combined "
    "non-invasive diabetes screening result."
)

st.info(
    "All required fields must be completed before prediction."
)

st.divider()


# =========================================================
# SECTION 1 - LIFESTYLE ASSESSMENT
# =========================================================

st.header("1. Lifestyle Risk Assessment")

col1, col2 = st.columns(2)


# =========================================================
# LEFT COLUMN
# =========================================================

with col1:

    st.subheader("Health and Lifestyle Information")

    Height = st.number_input(
        "Height (cm) *",
        min_value=0.0,
        max_value=250.0,
        value=0.0,
        step=0.1,
        key="Height"
    )

    Weight = st.number_input(
        "Weight (kg) *",
        min_value=0.0,
        max_value=300.0,
        value=0.0,
        step=0.1,
        key="Weight"
    )

    if Height > 0 and Weight > 0:
        height_m = Height / 100
        BMI = Weight / (height_m ** 2)

        st.info(
            f"Calculated BMI: {BMI:.1f}"
        )

    else:
        BMI = 0.0


    Smoker = st.selectbox(
        "Smoker *",
        ["Select", "No", "Yes"],
        key="Smoker"
    )

    HeartDiseaseorAttack = st.selectbox(
        "Heart Disease or Previous Heart Attack *",
        ["Select", "No", "Yes"],
        key="HeartDiseaseorAttack"
    )

    PhysActivity = st.selectbox(
        "Regular Physical Activity *",
        ["Select", "No", "Yes"],
        key="PhysActivity"
    )

    Fruits = st.selectbox(
        "Regular Fruit Consumption *",
        ["Select", "No", "Yes"],
        key="Fruits"
    )

    Veggies = st.selectbox(
        "Regular Vegetable Consumption *",
        ["Select", "No", "Yes"],
        key="Veggies"
    )


# =========================================================
# RIGHT COLUMN
# =========================================================

with col2:

    st.subheader("General and Personal Information")

    GenHlth = st.selectbox(
        "General Health *",
        [
            "Select",
            "Excellent",
            "Very Good",
            "Good",
            "Fair",
            "Poor"
        ],
        key="GenHlth"
    )

    physical_health_options = ["Select"] + list(range(0, 31))

    PhysHlth = st.selectbox(
        "Poor Physical Health Days in the Last 30 Days *",
        physical_health_options,
        key="PhysHlth"
    )

    DiffWalk = st.selectbox(
        "Difficulty Walking *",
        ["Select", "No", "Yes"],
        key="DiffWalk"
    )

    Sex = st.selectbox(
        "Sex *",
        ["Select", "Female", "Male"],
        key="Sex"
    )

    Age = st.selectbox(
        "Age Group *",
        [
            "Select",
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


st.caption("* Required field")

st.divider()


# =========================================================
# SECTION 2 - SYMPTOM SCREENING
# =========================================================

st.header("2. Early Symptom Screening")

st.write(
    "Answer the following questions based on symptoms you may "
    "have experienced."
)

sym_col1, sym_col2 = st.columns(2)


# =========================================================
# SYMPTOM LEFT COLUMN
# =========================================================

with sym_col1:

    Polyuria = st.selectbox(
        "Frequent Urination *",
        ["Select", "No", "Yes"],
        key="Polyuria"
    )

    Polydipsia = st.selectbox(
        "Excessive Thirst *",
        ["Select", "No", "Yes"],
        key="Polydipsia"
    )

    SuddenWeightLoss = st.selectbox(
        "Sudden Unexplained Weight Loss *",
        ["Select", "No", "Yes"],
        key="SuddenWeightLoss"
    )

    Weakness = st.selectbox(
        "Frequent Weakness or Fatigue *",
        ["Select", "No", "Yes"],
        key="Weakness"
    )

    Polyphagia = st.selectbox(
        "Increased Hunger *",
        ["Select", "No", "Yes"],
        key="Polyphagia"
    )

    GenitalThrush = st.selectbox(
        "Frequent Genital Infection *",
        ["Select", "No", "Yes"],
        key="GenitalThrush"
    )

    VisualBlurring = st.selectbox(
        "Blurred Vision *",
        ["Select", "No", "Yes"],
        key="VisualBlurring"
    )


# =========================================================
# SYMPTOM RIGHT COLUMN
# =========================================================

with sym_col2:

    Itching = st.selectbox(
        "Frequent Itching *",
        ["Select", "No", "Yes"],
        key="Itching"
    )

    Irritability = st.selectbox(
        "Increased Irritability *",
        ["Select", "No", "Yes"],
        key="Irritability"
    )

    DelayedHealing = st.selectbox(
        "Slow Wound Healing *",
        ["Select", "No", "Yes"],
        key="DelayedHealing"
    )

    PartialParesis = st.selectbox(
        "Muscle Weakness or Partial Loss of Movement *",
        ["Select", "No", "Yes"],
        key="PartialParesis"
    )

    MuscleStiffness = st.selectbox(
        "Muscle Stiffness *",
        ["Select", "No", "Yes"],
        key="MuscleStiffness"
    )

    Alopecia = st.selectbox(
        "Hair Loss *",
        ["Select", "No", "Yes"],
        key="Alopecia"
    )

    Obesity = st.selectbox(
        "Obesity *",
        ["Select", "No", "Yes"],
        key="Obesity"
    )


st.caption("* Required field")


# =========================================================
# VALUE MAPPINGS
# =========================================================

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


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

if st.button(
    "Generate Diabetes Screening Result",
    type="primary",
    use_container_width=True
):

    # =====================================================
    # VALIDATION
    # =====================================================

    missing_fields = []

    if Height <= 0:
        missing_fields.append("Height")

    if Weight <= 0:
        missing_fields.append("Weight")

    lifestyle_selects = {
        "Smoker": Smoker,
        "Heart Disease or Previous Heart Attack": HeartDiseaseorAttack,
        "Physical Activity": PhysActivity,
        "Fruit Consumption": Fruits,
        "Vegetable Consumption": Veggies,
        "General Health": GenHlth,
        "Physical Health Days": PhysHlth,
        "Difficulty Walking": DiffWalk,
        "Sex": Sex,
        "Age Group": Age
    }

    for name, value in lifestyle_selects.items():
        if value == "Select":
            missing_fields.append(name)


    symptom_selects = {
        "Frequent Urination": Polyuria,
        "Excessive Thirst": Polydipsia,
        "Sudden Weight Loss": SuddenWeightLoss,
        "Weakness or Fatigue": Weakness,
        "Increased Hunger": Polyphagia,
        "Genital Infection": GenitalThrush,
        "Blurred Vision": VisualBlurring,
        "Itching": Itching,
        "Irritability": Irritability,
        "Slow Wound Healing": DelayedHealing,
        "Muscle Weakness": PartialParesis,
        "Muscle Stiffness": MuscleStiffness,
        "Hair Loss": Alopecia,
        "Obesity": Obesity
    }

    for name, value in symptom_selects.items():
        if value == "Select":
            missing_fields.append(name)


    if missing_fields:

        st.error(
            "Please complete all required fields before generating "
            "the screening result."
        )

        st.write("**Missing fields:**")

        for field in missing_fields:
            st.write(f"- {field}")

        st.stop()


    # =====================================================
    # LIFESTYLE MODEL INPUT
    # =====================================================

    lifestyle_input = pd.DataFrame([{
        "BMI": BMI,
        "Smoker": yes_no[Smoker],
        "HeartDiseaseorAttack": yes_no[HeartDiseaseorAttack],
        "PhysActivity": yes_no[PhysActivity],
        "Fruits": yes_no[Fruits],
        "Veggies": yes_no[Veggies],
        "GenHlth": genhlth_map[GenHlth],
        "PhysHlth": PhysHlth,
        "DiffWalk": yes_no[DiffWalk],
        "Sex": sex_map[Sex],
        "Age": age_map[Age]
    }])


    lifestyle_prediction = lifestyle_model.predict(
        lifestyle_input
    )[0]

    lifestyle_probability = lifestyle_model.predict_proba(
        lifestyle_input
    )[0][1]

    lifestyle_percent = float(
        lifestyle_probability * 100
    )


    # =====================================================
    # SYMPTOM MODEL INPUT
    # =====================================================

    symptom_input = pd.DataFrame([{
        "polyuria": yes_no[Polyuria],
        "polydipsia": yes_no[Polydipsia],
        "sudden_weight_loss": yes_no[SuddenWeightLoss],
        "weakness": yes_no[Weakness],
        "polyphagia": yes_no[Polyphagia],
        "genital_thrush": yes_no[GenitalThrush],
        "visual_blurring": yes_no[VisualBlurring],
        "itching": yes_no[Itching],
        "irritability": yes_no[Irritability],
        "delayed_healing": yes_no[DelayedHealing],
        "partial_paresis": yes_no[PartialParesis],
        "muscle_stiffness": yes_no[MuscleStiffness],
        "alopecia": yes_no[Alopecia],
        "obesity": yes_no[Obesity]
    }])


    symptom_prediction = symptom_model.predict(
        symptom_input
    )[0]

    symptom_probability = symptom_model.predict_proba(
        symptom_input
    )[0][1]

    symptom_percent = float(
        symptom_probability * 100
    )


    # =====================================================
    # RESULTS
    # =====================================================

    st.divider()

    st.header("Screening Results")

    result1, result2, result3 = st.columns(3)

    with result1:
        st.metric(
            "Calculated BMI",
            f"{BMI:.1f}"
        )

    with result2:
        st.metric(
            "Lifestyle Risk Estimate",
            f"{lifestyle_percent:.1f}%"
        )

    with result3:
        st.metric(
            "Symptom Risk Estimate",
            f"{symptom_percent:.1f}%"
        )


    # =====================================================
    # LIFESTYLE RESULT
    # =====================================================

    st.subheader("Lifestyle-Based Assessment")

    st.progress(
        max(
            0,
            min(
                int(round(lifestyle_percent)),
                100
            )
        )
    )

    if lifestyle_prediction == 1:

        st.warning(
            "The lifestyle model indicates a higher diabetes "
            "risk pattern."
        )

    else:

        st.success(
            "The lifestyle model indicates a lower diabetes "
            "risk pattern."
        )


    # =====================================================
    # SYMPTOM RESULT
    # =====================================================

    st.subheader("Symptom-Based Assessment")

    st.progress(
        max(
            0,
            min(
                int(round(symptom_percent)),
                100
            )
        )
    )

    if symptom_prediction == 1:

        st.warning(
            "The symptom model identified a higher-risk "
            "symptom pattern."
        )

    else:

        st.success(
            "The symptom model identified a lower-risk "
            "symptom pattern."
        )


    # =====================================================
    # COMBINED SCREENING SUMMARY
    # =====================================================

    st.divider()

    st.header("Combined Screening Summary")

    if lifestyle_prediction == 1 and symptom_prediction == 1:

        combined_status = "Higher Screening Concern"

        st.error(
            "Both the lifestyle assessment and symptom screening "
            "indicate a higher-risk pattern."
        )

    elif lifestyle_prediction == 1 or symptom_prediction == 1:

        combined_status = "Moderate Screening Concern"

        st.warning(
            "One of the two screening models indicates a "
            "higher-risk pattern."
        )

    else:

        combined_status = "Lower Screening Concern"

        st.success(
            "Both screening models indicate a lower-risk pattern."
        )


    st.metric(
        "Overall Screening Classification",
        combined_status
    )


    st.caption(
        "The two probabilities are generated by separate models "
        "trained on different datasets. They are therefore presented "
        "separately rather than averaged into a single medical risk score."
    )


    # =====================================================
    # LIFESTYLE RISK FACTOR ANALYSIS
    # =====================================================

    st.divider()

    st.header("Lifestyle Model Risk Factor Analysis")

    lifestyle_features = [
        "BMI",
        "Smoker",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "GenHlth",
        "PhysHlth",
        "DiffWalk",
        "Sex",
        "Age"
    ]

    lifestyle_importance = pd.DataFrame({
        "Risk Factor": lifestyle_features,
        "Importance": lifestyle_model.feature_importances_
    })

    lifestyle_importance = lifestyle_importance.sort_values(
        by="Importance",
        ascending=False
    )

    st.bar_chart(
        lifestyle_importance.set_index("Risk Factor")
    )

    st.caption(
        "Feature importance indicates influence within the "
        "machine-learning model and does not imply causation."
    )


    # =====================================================
    # SYMPTOM MODEL ANALYSIS
    # =====================================================

    st.header("Symptom Model Analysis")

    symptom_display_names = {
        "polyuria": "Frequent Urination",
        "polydipsia": "Excessive Thirst",
        "sudden_weight_loss": "Sudden Weight Loss",
        "weakness": "Weakness or Fatigue",
        "polyphagia": "Increased Hunger",
        "genital_thrush": "Genital Infection",
        "visual_blurring": "Blurred Vision",
        "itching": "Itching",
        "irritability": "Irritability",
        "delayed_healing": "Slow Wound Healing",
        "partial_paresis": "Muscle Weakness",
        "muscle_stiffness": "Muscle Stiffness",
        "alopecia": "Hair Loss",
        "obesity": "Obesity"
    }

    symptom_features = list(
        symptom_display_names.keys()
    )

    symptom_importance = pd.DataFrame({
        "Symptom": [
            symptom_display_names[x]
            for x in symptom_features
        ],
        "Importance": symptom_model.feature_importances_
    })

    symptom_importance = symptom_importance.sort_values(
        by="Importance",
        ascending=False
    )

    st.bar_chart(
        symptom_importance.set_index("Symptom")
    )

    st.caption(
        "This chart represents overall feature importance in the "
        "symptom model, not a patient-specific medical explanation."
    )


    # =====================================================
    # PERSONALIZED HEALTH GUIDANCE
    # =====================================================

    st.divider()

    st.header("Personalized Preventive Guidance")

    attention = []
    positives = []


    # BMI
    if BMI >= 25:
        attention.append(
            "Focus on sustainable healthy-weight habits."
        )
    elif BMI >= 18.5:
        positives.append(
            "BMI is within the commonly used adult reference range."
        )


    # Activity
    if PhysActivity == "No":
        attention.append(
            "Gradually increase regular physical activity "
            "according to your ability."
        )
    else:
        positives.append(
            "Regular physical activity reported."
        )


    # Diet
    if Fruits == "No" or Veggies == "No":
        attention.append(
            "Improve dietary variety with appropriate fruits "
            "and vegetables."
        )
    else:
        positives.append(
            "Regular fruit and vegetable consumption reported."
        )


    # Smoking
    if Smoker == "Yes":
        attention.append(
            "Consider appropriate support for smoking cessation."
        )
    else:
        positives.append(
            "No smoking reported."
        )


    # General health
    if GenHlth in ["Poor", "Fair"]:
        attention.append(
            "Consider regular health monitoring and professional guidance."
        )


    # Symptoms
    if Polyuria == "Yes":
        attention.append(
            "Frequent urination was reported."
        )

    if Polydipsia == "Yes":
        attention.append(
            "Excessive thirst was reported."
        )

    if SuddenWeightLoss == "Yes":
        attention.append(
            "Sudden unexplained weight loss was reported."
        )

    if VisualBlurring == "Yes":
        attention.append(
            "Blurred vision was reported."
        )

    if DelayedHealing == "Yes":
        attention.append(
            "Slow wound healing was reported."
        )


    # =====================================================
    # PRIORITY ACTIONS
    # =====================================================

    st.subheader("Areas That May Need Attention")

    if attention:

        for item in attention:
            st.write(f"- {item}")

    else:

        st.success(
            "No major lifestyle or symptom concerns were "
            "identified from the selected responses."
        )


    st.subheader("Positive Habits to Maintain")

    if positives:

        for item in positives:
            st.write(f"- {item}")

    else:

        st.write(
            "Focus on the areas identified above."
        )


    # =====================================================
    # NEXT STEP
    # =====================================================

    st.subheader("Recommended Next Step")

    if lifestyle_prediction == 1 or symptom_prediction == 1:

        st.warning(
            "One or both screening models identified a higher-risk "
            "pattern. Consider discussing the result with a qualified "
            "healthcare professional, who can determine whether "
            "appropriate clinical assessment or testing is needed."
        )

    else:

        st.success(
            "Both models indicate a lower-risk pattern. Continue "
            "maintaining healthy lifestyle habits and routine "
            "healthcare as appropriate."
        )


    st.info(
        "This system is intended for educational and research "
        "purposes. It provides non-invasive risk screening and "
        "does not diagnose diabetes or replace clinical testing."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Non-Invasive Diabetes Risk Prediction System | "
    "Machine Learning-Based Final Year Project"
)
