
import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from fpdf import FPDF
from datetime import datetime
from supabase import create_client


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="wide"
)


# =========================================================
# SUPABASE CONNECTION AND AUTHENTICATION
# =========================================================

def create_supabase_client():
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["anon_key"]
    )


if "access_token" not in st.session_state:
    st.session_state.access_token = None

if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = None

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "page" not in st.session_state:
    st.session_state.page = "home"


try:
    supabase = create_supabase_client()
except Exception:
    st.error(
        "Unable to connect to Supabase. "
        "Please check your Streamlit Cloud Secrets."
    )
    st.stop()


def save_auth_session(auth_session):
    """Store the Supabase authentication tokens for this session."""
    st.session_state.access_token = auth_session.access_token
    st.session_state.refresh_token = auth_session.refresh_token
    st.session_state.user_id = str(auth_session.user.id)


def clear_auth_session():
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.user_id = None


# Restore the authenticated session after Streamlit reruns.
if (
    st.session_state.access_token
    and st.session_state.refresh_token
):
    try:
        auth_response = supabase.auth.set_session(
            st.session_state.access_token,
            st.session_state.refresh_token
        )

        if auth_response.session:
            save_auth_session(auth_response.session)
        else:
            clear_auth_session()

    except Exception:
        clear_auth_session()


# =========================================================
# LOGIN AND SIGNUP PAGE
# =========================================================

if not st.session_state.user_id:

    st.title("Diabetes Risk Prediction System")

    st.subheader("User Authentication")

    st.write(
        "Create an account or log in to access your "
        "personal diabetes screening dashboard."
    )

    login_tab, signup_tab = st.tabs(
        ["Login", "Create Account"]
    )

    with login_tab:

        st.subheader("Login")

        with st.form("login_form"):

            login_email = st.text_input(
                "Email Address"
            )

            login_password = st.text_input(
                "Password",
                type="password"
            )

            login_button = st.form_submit_button(
                "Login",
                type="primary",
                use_container_width=True
            )

        if login_button:

            if not login_email or not login_password:
                st.error("Please enter your email and password.")

            else:
                try:
                    response = supabase.auth.sign_in_with_password({
                        "email": login_email,
                        "password": login_password
                    })

                    if response.session:
                        save_auth_session(response.session)
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Login failed. Please try again.")

                
                except Exception as e:
    st.error(f"Signup error: {e}")

    with signup_tab:

        st.subheader("Create Account")

        with st.form("signup_form"):

            signup_email = st.text_input(
                "Email Address",
                key="signup_email"
            )

            signup_password = st.text_input(
                "Create Password",
                type="password",
                key="signup_password"
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password"
            )

            signup_button = st.form_submit_button(
                "Create Account",
                type="primary",
                use_container_width=True
            )

        if signup_button:

            if not signup_email or not signup_password:
                st.error("Please complete all fields.")

            elif signup_password != confirm_password:
                st.error("Passwords do not match.")

            elif len(signup_password) < 6:
                st.error(
                    "Password must contain at least 6 characters."
                )

            else:
                try:
                    response = supabase.auth.sign_up({
                        "email": signup_email,
                        "password": signup_password
                    })

                    if response.session:
                        save_auth_session(response.session)
                        st.success("Account created successfully!")
                        st.rerun()

                    else:
                        st.success(
                            "Account created! Please verify your "
                            "email using the confirmation link, "
                            "then return here and log in."
                        )

                except Exception:
                    st.error(
                        "Unable to create your account. "
                        "The email may already be registered, "
                        "or the password may not meet the requirements."
                    )

    st.stop()


# =========================================================
# AUTHENTICATED SESSION
# =========================================================

st.sidebar.title("Diabetes Risk System")

st.sidebar.caption(
    f"Logged in: {st.session_state.user_id[:8]}..."
)

selected_page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Prediction Dashboard",
        "Prediction History"
    ]
)

if selected_page == "Home":
    st.session_state.page = "home"

elif selected_page == "Prediction Dashboard":
    st.session_state.page = "dashboard"

else:
    st.session_state.page = "history"


if st.sidebar.button(
    "Logout",
    use_container_width=True
):

    try:
        supabase.auth.sign_out()
    except Exception:
        pass

    clear_auth_session()
    st.session_state.page = "home"
    st.rerun()


# =========================================================
# LOAD MACHINE LEARNING MODELS
# =========================================================

@st.cache_resource
def load_models():

    lifestyle_model = XGBClassifier()
    lifestyle_model.load_model(
        "diabetes_lifestyle_model.json"
    )

    symptom_model = XGBClassifier()
    symptom_model.load_model(
        "diabetes_symptom_model.json"
    )

    return lifestyle_model, symptom_model


try:
    lifestyle_model, symptom_model = load_models()

except Exception as e:
    st.error(
        f"Unable to load machine learning models: {e}"
    )
    st.stop()


# =========================================================
# PDF GENERATION FUNCTION
# =========================================================

def generate_pdf(
    patient_data,
    results,
    attention,
    positives
):

    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_fill_color(21, 62, 117)
    pdf.rect(0, 0, 210, 35, "F")

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(10, 9)

    pdf.cell(
        190,
        10,
        "DIABETES RISK PREDICTION REPORT",
        align="C"
    )

    pdf.set_font("Arial", size=9)
    pdf.set_xy(10, 21)

    pdf.cell(
        190,
        7,
        "Lifestyle and Symptom-Based Screening",
        align="C"
    )

    pdf.set_text_color(0, 0, 0)
    pdf.set_y(42)

    pdf.set_font("Arial", size=9)

    pdf.cell(
        pdf.epw,
        7,
        "Report Generated: "
        + datetime.now().strftime("%d-%m-%Y %H:%M"),
        new_x="LMARGIN",
        new_y="NEXT"
    )

    def add_section(title, items):

        pdf.ln(4)

        pdf.set_fill_color(219, 234, 254)
        pdf.set_font("Arial", "B", 12)

        pdf.cell(
            pdf.epw,
            10,
            title,
            border=0,
            fill=True,
            new_x="LMARGIN",
            new_y="NEXT"
        )

        pdf.ln(3)
        pdf.set_font("Arial", size=10)

        for key, value in items:

            text = f"{key}: {value}"

            pdf.set_x(pdf.l_margin)

            pdf.multi_cell(
                pdf.epw,
                7,
                text,
                border=0,
                align="L"
            )

    add_section(
        "1. PATIENT INFORMATION",
        list(patient_data.items())
    )

    add_section(
        "2. SCREENING RESULTS",
        list(results.items())
    )

    attention_items = [
        ("Attention", item)
        for item in attention
    ]

    if not attention_items:
        attention_items = [
            ("Status", "No major concerns identified.")
        ]

    add_section(
        "3. AREAS THAT MAY NEED ATTENTION",
        attention_items
    )

    positive_items = [
        ("Positive Habit", item)
        for item in positives
    ]

    if not positive_items:
        positive_items = [
            ("Status", "Continue maintaining healthy habits.")
        ]

    add_section(
        "4. POSITIVE HABITS TO MAINTAIN",
        positive_items
    )

    pdf.ln(5)

    pdf.set_fill_color(241, 245, 249)
    pdf.set_font("Arial", "B", 11)

    pdf.cell(
        pdf.epw,
        9,
        "MEDICAL DISCLAIMER",
        border=0,
        fill=True,
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.set_font("Arial", size=9)
    pdf.set_x(pdf.l_margin)

    pdf.multi_cell(
        pdf.epw,
        6,
        "This report is generated by a machine-learning-based "
        "screening system for educational and research purposes. "
        "It is not a medical diagnosis and does not replace "
        "laboratory testing or professional medical advice. "
        "Consult a qualified healthcare professional for "
        "appropriate clinical assessment."
    )

    pdf.ln(8)
    pdf.set_font("Arial", "I", 8)

    pdf.cell(
        pdf.epw,
        8,
        "Non-Invasive Diabetes Risk Prediction System",
        align="C"
    )

    return bytes(pdf.output())


# =========================================================
# SAVE PREDICTION TO SUPABASE
# =========================================================

def save_prediction(record):

    response = (
        supabase
        .table("screening_results")
        .insert(record)
        .execute()
    )

    return response


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

        1. A lifestyle-based risk assessment model.
        2. A symptom-based screening model.

        The system also provides preventive health guidance,
        PDF reports and personal prediction history.
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
            "**Prediction History**\n\n"
            "Stores and compares your previous screening results."
        )

    st.divider()

    st.success(
        "Your personal account is connected successfully."
    )

    st.write(
        "Choose **Prediction Dashboard** from the left "
        "navigation menu to begin screening."
    )

    st.caption(
        "Developed for educational and research purposes."
    )

    st.stop()


# =========================================================
# PREDICTION HISTORY PAGE
# =========================================================

if st.session_state.page == "history":

    st.title("My Prediction History")

    st.write(
        "View and compare your previously saved screening results."
    )

    try:

        response = (
            supabase
            .table("screening_results")
            .select(
                "id, created_at, bmi, "
                "lifestyle_probability, symptom_probability, "
                "combined_status"
            )
            .eq("user_id", st.session_state.user_id)
            .order("created_at", desc=False)
            .execute()
        )

        history_data = response.data

        if not history_data:

            st.info(
                "No previous predictions found. "
                "Complete your first screening to see your history."
            )

        else:

            history_df = pd.DataFrame(history_data)

            history_df["created_at"] = pd.to_datetime(
                history_df["created_at"],
                errors="coerce"
            )

            st.subheader("Previous Screening Records")

            display_df = history_df[
                [
                    "created_at",
                    "bmi",
                    "lifestyle_probability",
                    "symptom_probability",
                    "combined_status"
                ]
            ].copy()

            display_df.columns = [
                "Date",
                "BMI",
                "Lifestyle Estimate (%)",
                "Symptom Estimate (%)",
                "Overall Classification"
            ]

            st.dataframe(
                display_df.sort_values(
                    "Date",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            st.subheader("Prediction Comparison")

            chart_df = history_df.set_index("created_at")

            chart_df = chart_df[
                [
                    "bmi",
                    "lifestyle_probability",
                    "symptom_probability"
                ]
            ].rename(
                columns={
                    "bmi": "BMI",
                    "lifestyle_probability": "Lifestyle Estimate",
                    "symptom_probability": "Symptom Estimate"
                }
            )

            st.write("**BMI Progression**")

            st.line_chart(
                chart_df[["BMI"]],
                use_container_width=True
            )

            st.write(
                "**Lifestyle and Symptom Model Estimates**"
            )

            st.line_chart(
                chart_df[
                    [
                        "Lifestyle Estimate",
                        "Symptom Estimate"
                    ]
                ],
                use_container_width=True
            )

            latest = history_df.iloc[-1]

            st.divider()

            st.subheader("Latest Screening Record")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "BMI",
                f"{latest['bmi']:.1f}"
            )

            c2.metric(
                "Lifestyle Estimate",
                f"{latest['lifestyle_probability']:.1f}%"
            )

            c3.metric(
                "Symptom Estimate",
                f"{latest['symptom_probability']:.1f}%"
            )

            st.info(
                f"Latest classification: "
                f"{latest['combined_status']}"
            )

            st.caption(
                "Model estimates are screening outputs, not "
                "clinical diagnoses. Changes between records do "
                "not establish changes in actual disease status."
            )

    except Exception as e:

        st.error(
            f"Unable to retrieve prediction history: {e}"
        )

    st.stop()


# =========================================================
# PREDICTION DASHBOARD
# =========================================================

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

    physical_health_options = [
        "Select"
    ] + list(range(0, 31))

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
    "Answer the following questions based on symptoms "
    "you may have experienced."
)

sym_col1, sym_col2 = st.columns(2)


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

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

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
            "Please complete all required fields before "
            "generating the screening result."
        )

        st.write("**Missing fields:**")

        for field in missing_fields:
            st.write(f"- {field}")

        st.stop()

    # -----------------------------------------------------
    # LIFESTYLE MODEL INPUT
    # -----------------------------------------------------

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

    lifestyle_prediction = int(
        lifestyle_model.predict(lifestyle_input)[0]
    )

    lifestyle_probability = float(
        lifestyle_model.predict_proba(lifestyle_input)[0][1]
    )

    lifestyle_percent = lifestyle_probability * 100

    # -----------------------------------------------------
    # SYMPTOM MODEL INPUT
    # -----------------------------------------------------

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

    symptom_prediction = int(
        symptom_model.predict(symptom_input)[0]
    )

    symptom_probability = float(
        symptom_model.predict_proba(symptom_input)[0][1]
    )

    symptom_percent = symptom_probability * 100

    # -----------------------------------------------------
    # COMBINED CLASSIFICATION
    # -----------------------------------------------------

    if lifestyle_prediction == 1 and symptom_prediction == 1:

        combined_status = "Higher Screening Concern"

    elif lifestyle_prediction == 1 or symptom_prediction == 1:

        combined_status = "Moderate Screening Concern"

    else:

        combined_status = "Lower Screening Concern"

    # -----------------------------------------------------
    # PATIENT DATA
    # -----------------------------------------------------

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
        "Lifestyle Classification": (
            "Higher Risk Pattern"
            if lifestyle_prediction == 1
            else "Lower Risk Pattern"
        ),
        "Symptom Risk Estimate": f"{symptom_percent:.1f}%",
        "Symptom Classification": (
            "Higher Risk Pattern"
            if symptom_prediction == 1
            else "Lower Risk Pattern"
        ),
        "Overall Screening Classification": combined_status
    }

    # -----------------------------------------------------
    # PERSONALIZED HEALTH GUIDANCE
    # -----------------------------------------------------

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
            "Gradually increase regular physical activity "
            "according to your ability."
        )

    else:

        positives.append(
            "Regular physical activity reported."
        )

    if Fruits == "No" or Veggies == "No":

        attention.append(
            "Improve dietary variety with appropriate fruits "
            "and vegetables."
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

    if Polyuria == "Yes":
        attention.append("Frequent urination was reported.")

    if Polydipsia == "Yes":
        attention.append("Excessive thirst was reported.")

    if SuddenWeightLoss == "Yes":
        attention.append("Sudden unexplained weight loss was reported.")

    if VisualBlurring == "Yes":
        attention.append("Blurred vision was reported.")

    if DelayedHealing == "Yes":
        attention.append("Slow wound healing was reported.")

    # -----------------------------------------------------
    # SAVE RESULT TO SUPABASE
    # -----------------------------------------------------

    record = {
        "user_id": st.session_state.user_id,
        "bmi": float(BMI),
        "lifestyle_probability": float(lifestyle_percent),
        "lifestyle_prediction": int(lifestyle_prediction),
        "symptom_probability": float(symptom_percent),
        "symptom_prediction": int(symptom_prediction),
        "combined_status": combined_status,
        "patient_data": patient_data
    }

    try:

        save_prediction(record)

        st.success(
            "Your screening result has been saved "
            "successfully to your personal history!"
        )

    except Exception as e:

        st.error(
            "Your prediction was generated, but it could not "
            f"be saved to Supabase: {e}"
        )

    # -----------------------------------------------------
    # DISPLAY SCREENING RESULTS
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # LIFESTYLE RESULT
    # -----------------------------------------------------

    st.subheader("Lifestyle-Based Assessment")

    st.progress(
        max(0, min(int(round(lifestyle_percent)), 100))
    )

    if lifestyle_prediction == 1:

        st.warning(
            "The lifestyle model indicates a higher-risk pattern."
        )

    else:

        st.success(
            "The lifestyle model indicates a lower-risk pattern."
        )

    # -----------------------------------------------------
    # SYMPTOM RESULT
    # -----------------------------------------------------

    st.subheader("Symptom-Based Assessment")

    st.progress(
        max(0, min(int(round(symptom_percent)), 100))
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

    # -----------------------------------------------------
    # COMBINED SUMMARY
    # -----------------------------------------------------

    st.divider()

    st.header("Combined Screening Summary")

    if combined_status == "Higher Screening Concern":

        st.error(
            "Both screening models indicate a higher-risk pattern."
        )

    elif combined_status == "Moderate Screening Concern":

        st.warning(
            "One of the two screening models indicates a "
            "higher-risk pattern."
        )

    else:

        st.success(
            "Both screening models indicate a lower-risk pattern."
        )

    st.metric(
        "Overall Screening Classification",
        combined_status
    )

    st.caption(
        "The two probabilities are generated by separate models "
        "trained on different datasets. They are presented "
        "separately rather than averaged into a single medical score."
    )

    # -----------------------------------------------------
    # LIFESTYLE FEATURE IMPORTANCE
    # -----------------------------------------------------

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
        "Feature importance indicates influence within the model "
        "and does not imply causation."
    )

    # -----------------------------------------------------
    # SYMPTOM FEATURE IMPORTANCE
    # -----------------------------------------------------

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

    symptom_features = list(symptom_display_names.keys())

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
        "This chart represents overall feature importance in "
        "the symptom model, not a patient-specific explanation."
    )

    # -----------------------------------------------------
    # PERSONALIZED GUIDANCE
    # -----------------------------------------------------

    st.divider()

    st.header("Personalized Preventive Guidance")

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

    st.subheader("Recommended Next Step")

    if (
        lifestyle_prediction == 1
        or symptom_prediction == 1
    ):

        st.warning(
            "One or both screening models identified a higher-risk "
            "pattern. Consider discussing the result with a qualified "
            "healthcare professional."
        )

    else:

        st.success(
            "Both models indicate a lower-risk pattern. Continue "
            "maintaining healthy lifestyle habits and routine "
            "healthcare as appropriate."
        )

    st.info(
        "This system is intended for educational and research "
        "purposes. It does not diagnose diabetes or replace "
        "clinical testing."
    )

    # -----------------------------------------------------
    # PDF DOWNLOAD
    # -----------------------------------------------------

    pdf_data = generate_pdf(
        patient_data,
        results,
        attention,
        positives
    )

    st.divider()

    st.subheader("Download Your Screening Report")

    st.download_button(
        label="Download Diabetes Prediction Report (PDF)",
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
    "Non-Invasive Diabetes Risk Prediction System | "
    "Machine Learning-Based Final Year Project"
)
