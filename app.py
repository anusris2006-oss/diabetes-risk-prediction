
import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from fpdf import FPDF
from datetime import datetime


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #153e75;
    text-align: center;
}

.sub-title {
    text-align: center;
    color: #64748b;
    font-size: 16px;
    margin-bottom: 25px;
}

.section-title {
    color: #153e75;
    font-size: 23px;
    font-weight: 700;
}

div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}

div.stDownloadButton > button {
    border-radius: 10px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MACHINE LEARNING MODELS
# =========================================================

@st.cache_resource
def load_models():

    lifestyle = XGBClassifier()
    lifestyle.load_model("diabetes_lifestyle_model.json")

    symptom = XGBClassifier()
    symptom.load_model("diabetes_symptom_model.json")

    return lifestyle, symptom


try:
    lifestyle_model, symptom_model = load_models()

except Exception as e:
    st.error(f"Unable to load prediction models: {e}")
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None


def start_app():
    st.session_state.page = "dashboard"
    st.session_state.prediction_results = None


def go_home():
    st.session_state.page = "home"
    st.session_state.prediction_results = None


# =========================================================
# PDF GENERATION FUNCTION
# =========================================================

def generate_pdf(patient_data, results, attention, positives):

    pdf = FPDF()

    pdf.set_auto_page_break(
        auto=True,
        margin=15
    )

    pdf.add_page()

    # Header
    pdf.set_fill_color(21, 62, 117)
    pdf.rect(0, 0, 210, 35, "F")

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 17)

    pdf.set_y(10)

    pdf.cell(
        0, 12,
        "DIABETES RISK PREDICTION REPORT",
        align="C",
        ln=True
    )

    pdf.set_font("Arial", size=9)

    pdf.cell(
        0, 6,
        "Non-Invasive Lifestyle and Symptom-Based Screening",
        align="C",
        ln=True
    )

    pdf.set_text_color(0, 0, 0)
    pdf.set_y(43)

    # Date
    pdf.set_font("Arial", size=10)

    pdf.cell(
        0, 8,
        "Report Generated: " +
        datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        ln=True
    )

    # Patient Information
    pdf.ln(4)

    pdf.set_fill_color(219, 234, 254)
    pdf.set_font("Arial", "B", 13)

    pdf.cell(
        0, 10,
        "1. PATIENT INFORMATION",
        ln=True,
        fill=True
    )

    pdf.ln(3)
    pdf.set_font("Arial", size=10)

    for key, value in patient_data.items():

        pdf.set_font("Arial", "B", 10)
        pdf.cell(65, 8, str(key) + ":")

        pdf.set_font("Arial", size=10)

        pdf.multi_cell(
            0, 8, str(value)
        )

    # Prediction Results
    pdf.ln(5)

    pdf.set_fill_color(219, 234, 254)
    pdf.set_font("Arial", "B", 13)

    pdf.cell(
        0, 10,
        "2. SCREENING RESULTS",
        ln=True,
        fill=True
    )

    pdf.ln(3)

    for key, value in results.items():

        pdf.set_font("Arial", "B", 10)
        pdf.cell(75, 9, str(key) + ":")

        pdf.set_font("Arial", size=10)

        pdf.multi_cell(
            0, 9, str(value)
        )

    # Attention
    pdf.ln(5)

    pdf.set_fill_color(219, 234, 254)
    pdf.set_font("Arial", "B", 13)

    pdf.cell(
        0, 10,
        "3. AREAS THAT MAY NEED ATTENTION",
        ln=True,
        fill=True
    )

    pdf.ln(3)
    pdf.set_font("Arial", size=10)

    if attention:

        for item in attention:
            pdf.multi_cell(0, 8, "- " + item)

    else:

        pdf.multi_cell(
            0, 8,
            "No major concerns identified from the selected responses."
        )

    # Positive Habits
    pdf.ln(5)

    pdf.set_fill_color(219, 234, 254)
    pdf.set_font("Arial", "B", 13)

    pdf.cell(
        0, 10,
        "4. POSITIVE HABITS TO MAINTAIN",
        ln=True,
        fill=True
    )

    pdf.ln(3)
    pdf.set_font("Arial", size=10)

    if positives:

        for item in positives:
            pdf.multi_cell(0, 8, "- " + item)

    else:

        pdf.multi_cell(
            0, 8,
            "Continue maintaining healthy lifestyle habits."
        )

    # Medical Disclaimer
    pdf.ln(7)

    pdf.set_fill_color(241, 245, 249)
    pdf.set_font("Arial", "B", 11)

    pdf.cell(
        0, 9,
        "MEDICAL DISCLAIMER",
        ln=True,
        fill=True
    )

    pdf.set_font("Arial", size=9)

    pdf.multi_cell(
        0, 7,
        "This report is generated by a machine-learning-based "
        "screening system for educational and research purposes. "
        "It is not a medical diagnosis and does not replace "
        "laboratory testing or professional medical advice. "
        "Consult a qualified healthcare professional for "
        "appropriate clinical assessment."
    )

    # Footer
    pdf.ln(8)

    pdf.set_font("Arial", "I", 8)

    pdf.cell(
        0, 8,
        "Non-Invasive Diabetes Risk Prediction System",
        align="C"
    )

    return bytes(pdf.output())


# =========================================================
# HOME PAGE
# =========================================================

if st.session_state.page == "home":

    st.markdown(
        '<div class="main-title">🩺 Diabetes Risk Prediction System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Non-Invasive Lifestyle and Symptom-Based Diabetes Screening</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Welcome to the Screening System")

    st.write(
        """
        This application uses machine learning to estimate diabetes
        risk patterns using lifestyle information and commonly
        reported early symptoms.

        Two independent XGBoost models are used to generate
        separate screening estimates.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "**Lifestyle Assessment**\n\n"
            "Evaluates health, demographic and lifestyle factors."
        )

    with col2:
        st.info(
            "**Symptom Screening**\n\n"
            "Evaluates common symptoms associated with diabetes."
        )

    with col3:
        st.info(
            "**Preventive Guidance**\n\n"
            "Provides general healthy lifestyle suggestions."
        )

    st.divider()

    st.warning(
        "This application is intended for educational and research "
        "purposes and does not provide a medical diagnosis."
    )

    st.button(
        "🚀 Get Started",
        type="primary",
        use_container_width=True,
        on_click=start_app
    )

    st.divider()

    st.caption(
        "Machine Learning-Based Final Year Project"
    )

    st.stop()


# =========================================================
# DASHBOARD
# =========================================================

col1, col2 = st.columns([1, 5])

with col1:
    st.button(
        "⬅ Home",
        on_click=go_home
    )

st.title("Diabetes Risk Prediction Dashboard")

st.write(
    "Complete both assessment sections to generate your "
    "screening result."
)

st.info(
    "All required fields must be completed before prediction."
)

st.divider()


# =========================================================
# SECTION 1: LIFESTYLE ASSESSMENT
# =========================================================

st.header("1. Lifestyle Risk Assessment")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Health and Lifestyle")

    Height = st.number_input(
        "Height (cm) *",
        min_value=0.0,
        max_value=250.0,
        value=0.0,
        step=0.1
    )

    Weight = st.number_input(
        "Weight (kg) *",
        min_value=0.0,
        max_value=300.0,
        value=0.0,
        step=0.1
    )

    if Height > 0 and Weight > 0:

        BMI = Weight / ((Height / 100) ** 2)

        st.metric(
            "Calculated BMI",
            f"{BMI:.1f}"
        )

    else:
        BMI = 0.0

    Smoker = st.selectbox(
        "Smoker *",
        ["Select", "No", "Yes"]
    )

    HeartDiseaseorAttack = st.selectbox(
        "Heart Disease or Previous Heart Attack *",
        ["Select", "No", "Yes"]
    )

    PhysActivity = st.selectbox(
        "Regular Physical Activity *",
        ["Select", "No", "Yes"]
    )

    Fruits = st.selectbox(
        "Regular Fruit Consumption *",
        ["Select", "No", "Yes"]
    )

    Veggies = st.selectbox(
        "Regular Vegetable Consumption *",
        ["Select", "No", "Yes"]
    )


with col2:

    st.subheader("Personal Information")

    GenHlth = st.selectbox(
        "General Health *",
        [
            "Select",
            "Excellent",
            "Very Good",
            "Good",
            "Fair",
            "Poor"
        ]
    )

    PhysHlth = st.selectbox(
        "Poor Physical Health Days in Last 30 Days *",
        ["Select"] + list(range(0, 31))
    )

    DiffWalk = st.selectbox(
        "Difficulty Walking *",
        ["Select", "No", "Yes"]
    )

    Sex = st.selectbox(
        "Sex *",
        ["Select", "Female", "Male"]
    )

    Age = st.selectbox(
        "Age Group *",
        [
            "Select",
            "18-24", "25-29", "30-34",
            "35-39", "40-44", "45-49",
            "50-54", "55-59", "60-64",
            "65-69", "70-74", "75-79", "80+"
        ]
    )


st.caption("* Required fields")

st.divider()


# =========================================================
# SECTION 2: SYMPTOM SCREENING
# =========================================================

st.header("2. Early Symptom Screening")

st.write(
    "Select the appropriate answer for each symptom."
)

sym_col1, sym_col2 = st.columns(2)

with sym_col1:

    Polyuria = st.selectbox(
        "Frequent Urination *",
        ["Select", "No", "Yes"]
    )

    Polydipsia = st.selectbox(
        "Excessive Thirst *",
        ["Select", "No", "Yes"]
    )

    SuddenWeightLoss = st.selectbox(
        "Sudden Unexplained Weight Loss *",
        ["Select", "No", "Yes"]
    )

    Weakness = st.selectbox(
        "Frequent Weakness or Fatigue *",
        ["Select", "No", "Yes"]
    )

    Polyphagia = st.selectbox(
        "Increased Hunger *",
        ["Select", "No", "Yes"]
    )

    GenitalThrush = st.selectbox(
        "Frequent Genital Infection *",
        ["Select", "No", "Yes"]
    )

    VisualBlurring = st.selectbox(
        "Blurred Vision *",
        ["Select", "No", "Yes"]
    )


with sym_col2:

    Itching = st.selectbox(
        "Frequent Itching *",
        ["Select", "No", "Yes"]
    )

    Irritability = st.selectbox(
        "Increased Irritability *",
        ["Select", "No", "Yes"]
    )

    DelayedHealing = st.selectbox(
        "Slow Wound Healing *",
        ["Select", "No", "Yes"]
    )

    PartialParesis = st.selectbox(
        "Muscle Weakness or Partial Loss of Movement *",
        ["Select", "No", "Yes"]
    )

    MuscleStiffness = st.selectbox(
        "Muscle Stiffness *",
        ["Select", "No", "Yes"]
    )

    Alopecia = st.selectbox(
        "Hair Loss *",
        ["Select", "No", "Yes"]
    )

    Obesity = st.selectbox(
        "Obesity *",
        ["Select", "No", "Yes"]
    )


st.caption("* Required fields")


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
# GENERATE PREDICTION
# =========================================================

st.divider()

if st.button(
    "🔍 Generate Diabetes Screening Result",
    type="primary",
    use_container_width=True
):

    missing_fields = []

    if Height <= 0:
        missing_fields.append("Height")

    if Weight <= 0:
        missing_fields.append("Weight")

    lifestyle_selects = {
        "Smoker": Smoker,
        "Heart Disease": HeartDiseaseorAttack,
        "Physical Activity": PhysActivity,
        "Fruit Consumption": Fruits,
        "Vegetable Consumption": Veggies,
        "General Health": GenHlth,
        "Physical Health Days": PhysHlth,
        "Difficulty Walking": DiffWalk,
        "Sex": Sex,
        "Age": Age
    }

    symptom_selects = {
        "Frequent Urination": Polyuria,
        "Excessive Thirst": Polydipsia,
        "Sudden Weight Loss": SuddenWeightLoss,
        "Weakness": Weakness,
        "Increased Hunger": Polyphagia,
        "Genital Infection": GenitalThrush,
        "Blurred Vision": VisualBlurring,
        "Itching": Itching,
        "Irritability": Irritability,
        "Slow Healing": DelayedHealing,
        "Muscle Weakness": PartialParesis,
        "Muscle Stiffness": MuscleStiffness,
        "Hair Loss": Alopecia,
        "Obesity": Obesity
    }

    for name, value in lifestyle_selects.items():

        if value == "Select":
            missing_fields.append(name)

    for name, value in symptom_selects.items():

        if value == "Select":
            missing_fields.append(name)

    if missing_fields:

        st.error(
            "Please complete all required fields."
        )

        for field in missing_fields:
            st.write(f"- {field}")

        st.stop()


    # =====================================================
    # LIFESTYLE MODEL
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
    # SYMPTOM MODEL
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
    # COMBINED CLASSIFICATION
    # =====================================================

    if lifestyle_prediction == 1 and symptom_prediction == 1:

        combined_status = "Higher Screening Concern"

    elif lifestyle_prediction == 1 or symptom_prediction == 1:

        combined_status = "Moderate Screening Concern"

    else:

        combined_status = "Lower Screening Concern"


    # =====================================================
    # PERSONALIZED GUIDANCE
    # =====================================================

    attention = []
    positives = []

    if BMI >= 25:

        attention.append(
            "Focus on sustainable healthy-weight habits."
        )

    elif BMI >= 18.5:

        positives.append(
            "BMI is within the commonly used adult reference range."
        )

    if PhysActivity == "No":

        attention.append(
            "Gradually increase physical activity according to your ability."
        )

    else:

        positives.append(
            "Regular physical activity reported."
        )

    if Fruits == "No" or Veggies == "No":

        attention.append(
            "Improve dietary variety with appropriate fruits and vegetables."
        )

    else:

        positives.append(
            "Regular fruit and vegetable consumption reported."
        )

    if Smoker == "Yes":

        attention.append(
            "Consider appropriate support for smoking cessation."
        )

    else:

        positives.append(
            "No smoking reported."
        )

    if GenHlth in ["Poor", "Fair"]:

        attention.append(
            "Consider regular health monitoring and professional guidance."
        )

    symptom_guidance = {
        Polyuria: "Frequent urination was reported.",
        Polydipsia: "Excessive thirst was reported.",
        SuddenWeightLoss: "Sudden unexplained weight loss was reported.",
        VisualBlurring: "Blurred vision was reported.",
        DelayedHealing: "Slow wound healing was reported."
    }

    for answer, message in symptom_guidance.items():

        if answer == "Yes":
            attention.append(message)


    # =====================================================
    # SAVE RESULTS IN SESSION STATE
    # =====================================================

    patient_data = {
        "Height": f"{Height} cm",
        "Weight": f"{Weight} kg",
        "BMI": f"{BMI:.1f}",
        "Age Group": Age,
        "Sex": Sex,
        "Smoker": Smoker,
        "Heart Disease": HeartDiseaseorAttack,
        "Physical Activity": PhysActivity,
        "Fruit Consumption": Fruits,
        "Vegetable Consumption": Veggies,
        "General Health": GenHlth,
        "Physical Health Days": str(PhysHlth),
        "Difficulty Walking": DiffWalk,
        "Frequent Urination": Polyuria,
        "Excessive Thirst": Polydipsia,
        "Sudden Weight Loss": SuddenWeightLoss,
        "Weakness": Weakness,
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

    results = {
        "Lifestyle Risk Estimate": f"{lifestyle_percent:.1f}%",
        "Lifestyle Classification":
            "Higher Risk Pattern" if lifestyle_prediction == 1
            else "Lower Risk Pattern",
        "Symptom Risk Estimate": f"{symptom_percent:.1f}%",
        "Symptom Classification":
            "Higher Risk Pattern" if symptom_prediction == 1
            else "Lower Risk Pattern",
        "Overall Screening Classification": combined_status
    }

    st.session_state.prediction_results = {
        "patient_data": patient_data,
        "results": results,
        "attention": attention,
        "positives": positives,
        "lifestyle_prediction": int(lifestyle_prediction),
        "symptom_prediction": int(symptom_prediction),
        "lifestyle_percent": lifestyle_percent,
        "symptom_percent": symptom_percent,
        "BMI": BMI
    }


# =========================================================
# DISPLAY RESULTS
# =========================================================

data = st.session_state.prediction_results

if data is not None:

    st.divider()

    st.header("📊 Screening Results")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Calculated BMI",
            f"{data['BMI']:.1f}"
        )

    with col2:

        st.metric(
            "Lifestyle Risk Estimate",
            f"{data['lifestyle_percent']:.1f}%"
        )

    with col3:

        st.metric(
            "Symptom Risk Estimate",
            f"{data['symptom_percent']:.1f}%"
        )

    # Lifestyle result
    st.subheader("Lifestyle-Based Assessment")

    st.progress(
        int(max(0, min(100, round(data["lifestyle_percent"]))))
    )

    if data["lifestyle_prediction"] == 1:

        st.warning(
            "The lifestyle model indicates a higher-risk pattern."
        )

    else:

        st.success(
            "The lifestyle model indicates a lower-risk pattern."
        )

    # Symptom result
    st.subheader("Symptom-Based Assessment")

    st.progress(
        int(max(0, min(100, round(data["symptom_percent"]))))
    )

    if data["symptom_prediction"] == 1:

        st.warning(
            "The symptom model identified a higher-risk symptom pattern."
        )

    else:

        st.success(
            "The symptom model identified a lower-risk symptom pattern."
        )

    # Combined summary
    st.divider()

    st.header("Combined Screening Summary")

    if data["lifestyle_prediction"] == 1 and data["symptom_prediction"] == 1:

        st.error(
            "Both models indicate a higher-risk screening pattern."
        )

    elif data["lifestyle_prediction"] == 1 or data["symptom_prediction"] == 1:

        st.warning(
            "One model indicates a higher-risk screening pattern."
        )

    else:

        st.success(
            "Both models indicate a lower-risk screening pattern."
        )

    st.metric(
        "Overall Screening Classification",
        data["results"]["Overall Screening Classification"]
    )

    st.caption(
        "The two model probabilities are generated independently "
        "from different datasets and are not averaged into a "
        "single medical risk score."
    )


    # =====================================================
    # LIFESTYLE FEATURE IMPORTANCE
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
    }).sort_values(
        by="Importance",
        ascending=False
    )

    st.bar_chart(
        lifestyle_importance.set_index("Risk Factor")
    )

    st.caption(
        "Feature importance represents overall model influence, "
        "not a patient-specific explanation or proof of causation."
    )


    # =====================================================
    # SYMPTOM FEATURE IMPORTANCE
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

    symptom_importance = pd.DataFrame({
        "Symptom": list(symptom_display_names.values()),
        "Importance": symptom_model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    st.bar_chart(
        symptom_importance.set_index("Symptom")
    )

    st.caption(
        "This chart represents overall feature importance, "
        "not a patient-specific medical explanation."
    )


    # =====================================================
    # PERSONALIZED GUIDANCE
    # =====================================================

    st.divider()

    st.header("💡 Personalized Preventive Guidance")

    st.subheader("Areas That May Need Attention")

    if data["attention"]:

        for item in data["attention"]:
            st.write(f"- {item}")

    else:

        st.success(
            "No major concerns identified from the selected responses."
        )

    st.subheader("Positive Habits to Maintain")

    if data["positives"]:

        for item in data["positives"]:
            st.write(f"- {item}")

    else:

        st.write(
            "Continue focusing on healthy lifestyle habits."
        )


    # Next step
    st.subheader("Recommended Next Step")

    if (
        data["lifestyle_prediction"] == 1
        or data["symptom_prediction"] == 1
    ):

        st.warning(
            "Consider discussing the screening results with a "
            "qualified healthcare professional. Clinical assessment "
            "and appropriate testing may be needed."
        )

    else:

        st.success(
            "Continue healthy lifestyle habits and routine healthcare "
            "as appropriate. A lower-risk screening result does not "
            "rule out diabetes."
        )

    st.info(
        "This system is intended for educational and research purposes. "
        "It does not diagnose diabetes or replace clinical testing."
    )


    # =====================================================
    # PDF DOWNLOAD
    # =====================================================

    st.divider()

    st.header("📄 Download Patient Screening Report")

    st.write(
        "Download a PDF containing the patient's responses, "
        "individual model predictions, screening classification "
        "and preventive guidance."
    )

    pdf_data = generate_pdf(
        data["patient_data"],
        data["results"],
        data["attention"],
        data["positives"]
    )

    st.download_button(
        label="📥 Download Diabetes Prediction Report (PDF)",
        data=pdf_data,
        file_name="Diabetes_Risk_Prediction_Report.pdf",
        mime="application/pdf",
        use_container_width=True,
        type="primary",
        on_click="ignore"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🩺 Non-Invasive Diabetes Risk Prediction System | "
    "Machine Learning-Based Final Year Project"
)
