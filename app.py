import streamlit as st
import pandas as pd
from xgboost import XGBClassifier


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="wide"
)


# =========================================================
# PROFESSIONAL LIGHT THEME
# =========================================================

# =========================================================
# PROFESSIONAL CLEAN THEME
# =========================================================

st.markdown("""
<style>

/* ========================================================
   PAGE
======================================================== */

.stApp {
    background-color: #F4F7FA;
    color: #1F2937;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ========================================================
   HEADINGS
======================================================== */

h1 {
    color: #163A5F !important;
    font-family: Arial, Helvetica, sans-serif !important;
    font-weight: 700 !important;
}

h2 {
    color: #1D4F73 !important;
    font-family: Arial, Helvetica, sans-serif !important;
    font-weight: 700 !important;
}

h3 {
    color: #2A5F82 !important;
    font-family: Arial, Helvetica, sans-serif !important;
    font-weight: 600 !important;
}


/* ========================================================
   NORMAL TEXT
======================================================== */

.stMarkdown,
.stMarkdown p,
.stMarkdown li {
    color: #263746 !important;
    font-family: Arial, Helvetica, sans-serif !important;
}


/* ========================================================
   FORM LABELS
======================================================== */

[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
.stSelectbox label,
.stSelectbox label *,
.stNumberInput label,
.stNumberInput label * {

    color: #1F2937 !important;
    opacity: 1 !important;

    -webkit-text-fill-color: #1F2937 !important;

    font-family: Arial, Helvetica, sans-serif !important;
    font-weight: 600 !important;
}


/* ========================================================
   SELECT BOX
   Transparent + one complete border
======================================================== */

div[data-baseweb="select"] > div {

    background: transparent !important;

    border: 2px solid #5E6B78 !important;

    border-radius: 8px !important;

    box-shadow: none !important;

    color: #1F2937 !important;

    min-height: 43px !important;
}


/* Text inside select */
div[data-baseweb="select"] span {

    color: #1F2937 !important;

    -webkit-text-fill-color: #1F2937 !important;

    font-weight: 500 !important;
}


/* Select arrow */
div[data-baseweb="select"] svg {

    fill: #374151 !important;
}


/* Select focus */
div[data-baseweb="select"] > div:focus-within {

    border: 2px solid #163A5F !important;

    box-shadow: none !important;
}


/* ========================================================
   NUMBER INPUT
   Transparent + complete uniform border
======================================================== */

.stNumberInput div[data-baseweb="input"] {

    background: transparent !important;

    border: 2px solid #5E6B78 !important;

    border-radius: 8px !important;

    box-shadow: none !important;

    overflow: hidden !important;
}


/* Number input text */
.stNumberInput input {

    background: transparent !important;

    color: #1F2937 !important;

    -webkit-text-fill-color: #1F2937 !important;

    border: none !important;

    box-shadow: none !important;

    font-weight: 500 !important;
}


/* Remove inner border around +/- section */
.stNumberInput div[data-baseweb="input"] > div {

    border: none !important;

    background: transparent !important;

    box-shadow: none !important;
}


/* Plus and minus buttons */
.stNumberInput button {

    background: transparent !important;

    color: #1F2937 !important;

    border: none !important;

    box-shadow: none !important;
}


/* Remove borders from +/- icons */
.stNumberInput button svg {

    fill: #1F2937 !important;
}


/* Number input focus */
.stNumberInput div[data-baseweb="input"]:focus-within {

    border: 2px solid #163A5F !important;

    box-shadow: none !important;
}


/* ========================================================
   BUTTONS
======================================================== */

.stButton > button {

    background-color: #163A5F !important;

    color: #FFFFFF !important;

    border: 2px solid #163A5F !important;

    border-radius: 8px !important;

    font-weight: 600 !important;

    padding: 0.6rem 1.2rem !important;

    box-shadow: none !important;
}


.stButton > button p {

    color: #FFFFFF !important;
}


.stButton > button:hover {

    background-color: #21577D !important;

    border-color: #21577D !important;

    color: #FFFFFF !important;
}


/* ========================================================
   METRIC CARDS
======================================================== */

div[data-testid="stMetric"] {

    background-color: #FFFFFF !important;

    border: 1px solid #D6DEE7 !important;

    border-radius: 10px !important;

    padding: 20px !important;
}


div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] * {

    color: #536471 !important;
}


div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] * {

    color: #163A5F !important;

    font-weight: 700 !important;
}


/* ========================================================
   EXPANDERS
======================================================== */

div[data-testid="stExpander"] {

    background-color: #FFFFFF !important;

    border: 1px solid #D6DEE7 !important;

    border-radius: 8px !important;
}


/* ========================================================
   DIVIDERS
======================================================== */

hr {

    border: none !important;

    border-top: 1px solid #D6DEE7 !important;
}


/* ========================================================
   STREAMLIT HEADER
======================================================== */

header[data-testid="stHeader"] {

    background-color: transparent !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

model = XGBClassifier()
model.load_model("diabetes_risk_model.json")


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"


# =========================================================
# NAVIGATION FUNCTIONS
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
        "Machine Learning-Based Risk Prediction, "
        "Risk Factor Analysis and Health Guidance"
    )

    st.write(
        """
        This application uses machine learning to estimate diabetes risk
        based on health, lifestyle, and demographic information.

        The system uses a Balanced XGBoost model to estimate diabetes
        risk, analyse important risk factors, and provide general
        preventive lifestyle guidance.
        """
    )

    st.write("### System Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "**Health Assessment**\n\n"
            "Enter health, lifestyle, and demographic information."
        )

    with col2:
        st.info(
            "**Machine Learning Prediction**\n\n"
            "Estimate diabetes risk using the trained "
            "Balanced XGBoost model."
        )

    with col3:
        st.info(
            "**Personalized Health Guidance**\n\n"
            "Receive general preventive lifestyle guidance "
            "based on the information entered."
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
    "Complete all the fields below to estimate diabetes risk."
)

st.info(
    "All fields are required before a prediction can be generated."
)

st.divider()


# =========================================================
# PATIENT DETAILS
# =========================================================

st.header("Patient Details")

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

    # Automatic BMI
    if Height > 0 and Weight > 0:

        height_in_meters = Height / 100

        BMI = Weight / (height_in_meters ** 2)

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
        "Physical Activity *",
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

    HvyAlcoholConsump = st.selectbox(
        "Heavy Alcohol Consumption *",
        ["Select", "No", "Yes"],
        key="HvyAlcoholConsump"
    )

    AnyHealthcare = st.selectbox(
        "Healthcare Coverage *",
        ["Select", "No", "Yes"],
        key="AnyHealthcare"
    )

    NoDocbcCost = st.selectbox(
        "Unable to See a Doctor Due to Cost *",
        ["Select", "No", "Yes"],
        key="NoDocbcCost"
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

    mental_health_options = ["Select"] + list(range(0, 31))

    MentHlth = st.selectbox(
        "Poor Mental Health Days in the Last 30 Days *",
        mental_health_options,
        key="MentHlth"
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

    Education = st.selectbox(
        "Education Level *",
        [
            "Select",
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
        "Income Range *",
        [
            "Select",
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


st.caption("* Required field")


# =========================================================
# MODEL VALUE MAPPINGS
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


# =========================================================
# PREDICTION
# =========================================================

st.divider()

if st.button(
    "Predict Diabetes Risk",
    type="primary",
    use_container_width=True
):

    # -----------------------------------------------------
    # REQUIRED FIELD VALIDATION
    # -----------------------------------------------------

    missing_fields = []

    if Height <= 0:
        missing_fields.append("Height")

    if Weight <= 0:
        missing_fields.append("Weight")

    if Smoker == "Select":
        missing_fields.append("Smoker")

    if HeartDiseaseorAttack == "Select":
        missing_fields.append(
            "Heart Disease or Previous Heart Attack"
        )

    if PhysActivity == "Select":
        missing_fields.append("Physical Activity")

    if Fruits == "Select":
        missing_fields.append("Fruit Consumption")

    if Veggies == "Select":
        missing_fields.append("Vegetable Consumption")

    if HvyAlcoholConsump == "Select":
        missing_fields.append("Heavy Alcohol Consumption")

    if AnyHealthcare == "Select":
        missing_fields.append("Healthcare Coverage")

    if NoDocbcCost == "Select":
        missing_fields.append(
            "Unable to See a Doctor Due to Cost"
        )

    if GenHlth == "Select":
        missing_fields.append("General Health")

    if MentHlth == "Select":
        missing_fields.append("Mental Health Days")

    if PhysHlth == "Select":
        missing_fields.append("Physical Health Days")

    if DiffWalk == "Select":
        missing_fields.append("Difficulty Walking")

    if Sex == "Select":
        missing_fields.append("Sex")

    if Age == "Select":
        missing_fields.append("Age Group")

    if Education == "Select":
        missing_fields.append("Education Level")

    if Income == "Select":
        missing_fields.append("Income Range")


    # -----------------------------------------------------
    # STOP IF SOMETHING IS MISSING
    # -----------------------------------------------------

    if missing_fields:

        st.error(
            "Please complete all required fields before "
            "generating a prediction."
        )

        st.write("**Missing fields:**")

        for field in missing_fields:
            st.write(f"- {field}")

        st.stop()


    # -----------------------------------------------------
    # PREPARE MODEL INPUT
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    risk_percent = float(probability * 100)


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.divider()

    st.header("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Calculated BMI",
            f"{BMI:.1f}"
        )

    with result_col2:
        st.metric(
            "Estimated Diabetes Risk",
            f"{risk_percent:.1f}%"
        )

    with result_col3:

        if prediction == 1:
            st.metric(
                "Risk Classification",
                "Higher Risk"
            )
        else:
            st.metric(
                "Risk Classification",
                "Lower Risk"
            )


    # -----------------------------------------------------
    # RISK PROBABILITY
    # -----------------------------------------------------

    st.subheader("Risk Probability")

    progress_value = int(round(risk_percent))

    progress_value = max(
        0,
        min(progress_value, 100)
    )

    st.progress(progress_value)

    if prediction == 1:

        st.warning(
            "The model indicates a higher likelihood "
            "of diabetes risk."
        )

    else:

        st.success(
            "The model indicates a lower likelihood "
            "of diabetes risk."
        )

    st.caption(
        "The prediction is generated by a machine-learning "
        "model for educational and research purposes and "
        "should not be considered a medical diagnosis."
    )


    # =====================================================
    # MODEL RISK FACTOR ANALYSIS
    # =====================================================

    st.divider()

    st.header("Model Risk Factor Analysis")

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
        "The chart below shows the features that were most "
        "influential in the trained model across the dataset."
    )

    st.bar_chart(
        importance_df.set_index("Risk Factor")
    )

    st.caption(
        "Feature importance represents influence within the "
        "machine-learning model. It does not establish that "
        "a factor causes diabetes."
    )


    # =====================================================
    # PERSONALIZED HEALTH GUIDANCE
    # =====================================================

    st.divider()

    st.header("Personalized Health Guidance")

    st.write(
        "Based on the information entered, the following "
        "section highlights lifestyle areas that may deserve "
        "attention and positive habits that can be maintained."
    )

    st.caption(
        "This section provides general preventive information, "
        "not an individualized medical treatment plan."
    )


    # =====================================================
    # IDENTIFY ATTENTION AREAS
    # =====================================================

    attention_areas = []
    positive_habits = []


    # BMI
    if BMI >= 25:

        attention_areas.append(
            (
                "Weight Management",
                f"Your calculated BMI is {BMI:.1f}. "
                "Consider focusing on sustainable healthy-weight "
                "habits through balanced food choices and regular "
                "physical activity."
            )
        )

    elif BMI < 18.5:

        attention_areas.append(
            (
                "Weight and Nutrition",
                f"Your calculated BMI is {BMI:.1f}. "
                "Consider discussing healthy weight and nutrition "
                "with an appropriate healthcare professional."
            )
        )

    else:

        positive_habits.append(
            f"Calculated BMI: {BMI:.1f}"
        )


    # Physical activity
    if PhysActivity == "No":

        attention_areas.append(
            (
                "Physical Activity",
                "You reported no regular physical activity. "
                "Consider gradually building regular physical "
                "activity into your routine according to your "
                "current ability and health status."
            )
        )

    else:

        positive_habits.append(
            "Regular physical activity reported"
        )


    # Fruit
    if Fruits == "No":

        attention_areas.append(
            (
                "Fruit Intake",
                "You reported that you do not regularly consume "
                "fruit. Consider improving dietary variety by "
                "including appropriate whole fruits."
            )
        )

    else:

        positive_habits.append(
            "Regular fruit consumption reported"
        )


    # Vegetables
    if Veggies == "No":

        attention_areas.append(
            (
                "Vegetable Intake",
                "You reported that you do not regularly consume "
                "vegetables. Consider increasing the variety of "
                "vegetables in your overall eating pattern."
            )
        )

    else:

        positive_habits.append(
            "Regular vegetable consumption reported"
        )


    # Smoking
    if Smoker == "Yes":

        attention_areas.append(
            (
                "Smoking",
                "You reported a history of smoking. Avoiding "
                "tobacco and seeking appropriate smoking-cessation "
                "support can contribute to improved overall health."
            )
        )

    else:

        positive_habits.append(
            "No smoking history reported"
        )


    # Alcohol
    if HvyAlcoholConsump == "Yes":

        attention_areas.append(
            (
                "Heavy Alcohol Consumption",
                "You reported heavy alcohol consumption. "
                "Consider discussing alcohol use with a qualified "
                "healthcare professional."
            )
        )

    else:

        positive_habits.append(
            "No heavy alcohol consumption reported"
        )


    # General health
    if GenHlth in ["Poor", "Fair"]:

        attention_areas.append(
            (
                "General Health",
                f"You rated your general health as {GenHlth}. "
                "Regular health monitoring and discussion with "
                "a qualified healthcare professional may be useful."
            )
        )


    # Physical health
    if PhysHlth >= 10:

        attention_areas.append(
            (
                "Physical Well-being",
                f"You reported {PhysHlth} poor physical-health "
                "days during the last 30 days. Persistent physical "
                "health concerns may be worth discussing with a "
                "qualified healthcare professional."
            )
        )


    # Difficulty walking
    if DiffWalk == "Yes":

        attention_areas.append(
            (
                "Mobility",
                "You reported difficulty walking. Physical activity "
                "choices should take your mobility and current "
                "health status into account."
            )
        )


    # =====================================================
    # AREAS NEEDING ATTENTION
    # =====================================================

    st.subheader("Areas That May Need Attention")

    if len(attention_areas) == 0:

        st.success(
            "No major modifiable lifestyle concerns were "
            "identified from the selected factors."
        )

    else:

        for title, message in attention_areas:

            with st.expander(
                title,
                expanded=True
            ):
                st.write(message)


    # =====================================================
    # HEALTHY EATING GUIDANCE
    # =====================================================

    st.subheader("Healthy Eating Guidance")

    if Fruits == "No" and Veggies == "No":

        st.warning(
            "Your responses indicate that both fruit and "
            "vegetable intake may need attention."
        )

    elif Fruits == "No":

        st.warning(
            "Your responses indicate that fruit intake "
            "may need attention."
        )

    elif Veggies == "No":

        st.warning(
            "Your responses indicate that vegetable intake "
            "may need attention."
        )

    else:

        st.success(
            "You reported consuming both fruits and vegetables. "
            "Continue maintaining a balanced eating pattern."
        )

    st.markdown(
        """
**General healthy-eating principles**

- Include a variety of vegetables and suitable whole fruits.
- Choose whole grains and other higher-fibre foods where appropriate.
- Include suitable protein sources such as pulses, beans, eggs or fish.
- Limit frequent intake of highly processed foods.
- Limit foods and drinks high in added sugars.
- Pay attention to portion sizes and overall dietary balance.
- Choose water instead of sugar-sweetened drinks when possible.
        """
    )


    # =====================================================
    # PHYSICAL ACTIVITY GUIDANCE
    # =====================================================

    st.subheader(
        "Physical Activity and Lifestyle Guidance"
    )

    if DiffWalk == "Yes":

        st.warning(
            "You reported difficulty walking. Consider physical "
            "activities appropriate for your mobility and health "
            "status, and seek professional guidance before making "
            "major changes to your activity routine."
        )

    elif PhysActivity == "No":

        st.warning(
            "You reported no regular physical activity. "
            "Consider gradually building physical activity "
            "into your routine according to your current "
            "health and ability."
        )

    else:

        st.success(
            "You reported being physically active. Maintaining "
            "regular physical activity is a positive lifestyle habit."
        )


    # =====================================================
    # PRIORITY ACTION PLAN
    # =====================================================

    st.subheader("Priority Action Plan")

    priorities = []

    if BMI >= 25:

        priorities.append(
            "Focus on sustainable healthy-weight habits."
        )

    elif BMI < 18.5:

        priorities.append(
            "Consider appropriate guidance regarding "
            "healthy weight and nutrition."
        )

    if PhysActivity == "No":

        priorities.append(
            "Gradually build regular physical activity "
            "according to your ability."
        )

    if Fruits == "No" or Veggies == "No":

        priorities.append(
            "Improve the variety and overall quality "
            "of your eating pattern."
        )

    if Smoker == "Yes":

        priorities.append(
            "Consider appropriate support for smoking cessation."
        )

    if HvyAlcoholConsump == "Yes":

        priorities.append(
            "Address heavy alcohol consumption with "
            "appropriate support."
        )

    if GenHlth in ["Poor", "Fair"]:

        priorities.append(
            "Consider regular health monitoring and "
            "professional guidance."
        )

    if DiffWalk == "Yes":

        priorities.append(
            "Consider mobility-appropriate physical activity."
        )


    if len(priorities) == 0:

        st.success(
            "Continue maintaining the positive lifestyle "
            "habits you have reported."
        )

    else:

        for number, item in enumerate(
            priorities[:4],
            start=1
        ):

            st.markdown(
                f"**Priority {number}:** {item}"
            )


    # =====================================================
    # POSITIVE HABITS
    # =====================================================

    st.subheader("Positive Habits to Maintain")

    if len(positive_habits) > 0:

        for habit in positive_habits:
            st.write(f"- {habit}")

    else:

        st.write(
            "Use the areas identified above as starting "
            "points for gradual lifestyle improvement."
        )


    # =====================================================
    # RECOMMENDED NEXT STEP
    # =====================================================

    st.subheader("Recommended Next Step")

    if prediction == 1:

        st.warning(
            "The model estimated a higher likelihood of diabetes "
            "risk. Consider discussing your overall risk profile "
            "with a qualified healthcare professional, who can "
            "determine whether clinical assessment or testing "
            "is appropriate."
        )

    else:

        st.success(
            "The model estimated a lower likelihood of diabetes "
            "risk. Continue maintaining healthy lifestyle habits "
            "and routine healthcare as appropriate."
        )


    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.info(
        "This system is intended for educational and research "
        "purposes. The prediction and lifestyle guidance do not "
        "constitute a medical diagnosis or individualized "
        "treatment advice."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Diabetes Risk Prediction System | "
    "Machine Learning-Based Final Year Project"
)
