import streamlit as st
import pandas as pd
from xgboost import XGBClassifier


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# LOAD TRAINED MODEL
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
    """
    Clear old values and open a fresh dashboard.
    """
    st.session_state.clear()
    st.session_state.page = "dashboard"


def go_home():
    """
    Clear patient details and return to the welcome page.
    """
    st.session_state.clear()
    st.session_state.page = "home"


# =========================================================
# HOME / WELCOME PAGE
# =========================================================

if st.session_state.page == "home":

    st.title("🩺 Diabetes Risk Prediction")

    st.subheader(
        "Machine Learning-Based Diabetes Risk Prediction, "
        "Risk Factor Analysis and Health Guidance"
    )

    st.write(
        """
        Welcome to the **Diabetes Risk Prediction System**.

        This application uses machine learning to estimate diabetes risk
        based on health, lifestyle, and demographic information.

        The system uses a **Balanced XGBoost Model** to estimate diabetes
        risk, analyse important risk factors, and provide general
        preventive lifestyle guidance.
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
            "Estimate diabetes risk using the trained "
            "Balanced XGBoost model."
        )

    with col3:
        st.info(
            "🌿 **Health Guidance**\n\n"
            "Receive general preventive lifestyle guidance "
            "based on the information entered."
        )

    st.write("")

    st.button(
        "🚀 Get Started",
        type="primary",
        use_container_width=True,
        on_click=start_app
    )

    st.caption(
        "This application is developed for educational "
        "and research purposes."
    )

    st.stop()


# =========================================================
# DASHBOARD
# =========================================================

st.button(
    "← Back to Home",
    on_click=go_home
)

st.title("🩺 Diabetes Risk Prediction Dashboard")

st.write(
    "Enter the details below to estimate diabetes risk "
    "using the trained Balanced XGBoost model."
)

st.success("✅ Machine Learning Model Loaded Successfully")

st.divider()


# =========================================================
# PATIENT DETAILS
# =========================================================

st.header("👤 Patient Details")

col1, col2 = st.columns(2)


# =========================================================
# LEFT COLUMN - HEALTH & LIFESTYLE
# =========================================================

with col1:

    st.subheader("🏃 Health & Lifestyle")

    # -----------------------------------------------------
    # HEIGHT
    # -----------------------------------------------------

    Height = st.number_input(
        "Height (cm)",
        min_value=0.0,
        max_value=250.0,
        value=0.0,
        step=0.1,
        key="Height"
    )

    # -----------------------------------------------------
    # WEIGHT
    # -----------------------------------------------------

    Weight = st.number_input(
        "Weight (kg)",
        min_value=0.0,
        max_value=300.0,
        value=0.0,
        step=0.1,
        key="Weight"
    )

    # -----------------------------------------------------
    # AUTOMATIC BMI CALCULATION
    # -----------------------------------------------------

    if Height > 0 and Weight > 0:

        height_in_meters = Height / 100

        BMI = Weight / (height_in_meters ** 2)

        st.info(
            f"⚖️ Calculated BMI: **{BMI:.1f}**"
        )

    else:

        BMI = 0.0


    # -----------------------------------------------------
    # OTHER HEALTH & LIFESTYLE INPUTS
    # -----------------------------------------------------

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


# =========================================================
# RIGHT COLUMN - GENERAL & PERSONAL INFORMATION
# =========================================================

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
        "Mental Health - Poor Days (0-30)",
        min_value=0,
        max_value=30,
        value=0,
        key="MentHlth"
    )

    PhysHlth = st.number_input(
        "Physical Health - Poor Days (0-30)",
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
# PREDICTION SECTION
# =========================================================

st.divider()

if st.button(
    "🔍 Predict Diabetes Risk",
    type="primary",
    use_container_width=True
):

    # =====================================================
    # INPUT VALIDATION
    # =====================================================

    if Height == 0 and Weight == 0:

        st.warning(
            "⚠️ Please enter your height and weight "
            "before making a prediction."
        )

        st.stop()

    elif Height == 0:

        st.warning(
            "⚠️ Please enter your height before making a prediction."
        )

        st.stop()

    elif Weight == 0:

        st.warning(
            "⚠️ Please enter your weight before making a prediction."
        )

        st.stop()


    # =====================================================
    # PREPARE MODEL INPUT
    # =====================================================

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


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    risk_percent = float(probability * 100)


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.divider()

    st.header("📊 Prediction Result")

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
                "Model Prediction",
                "Higher Risk"
            )

        else:

            st.metric(
                "Model Prediction",
                "Lower Risk"
            )


    # -----------------------------------------------------
    # RISK PROBABILITY BAR
    # -----------------------------------------------------

    st.write("### Risk Probability")

    progress_value = int(round(risk_percent))

    progress_value = max(
        0,
        min(progress_value, 100)
    )

    st.progress(progress_value)


    # -----------------------------------------------------
    # MODEL RESULT MESSAGE
    # -----------------------------------------------------

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


    # =====================================================
    # MODEL RISK FACTOR ANALYSIS
    # =====================================================

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
        "The chart below shows the features that were most "
        "influential in the model's predictions across the dataset."
    )

    st.bar_chart(
        importance_df.set_index("Risk Factor")
    )

    st.caption(
        "Feature importance indicates influence on the "
        "machine-learning model's predictions and does not "
        "imply that a factor causes diabetes."
    )


    # =====================================================
    # PERSONALIZED HEALTH GUIDANCE
    # =====================================================

    st.divider()

    st.header("🌿 Personalized Health Guidance")

    st.write(
        "Based on the information you entered, this section "
        "highlights lifestyle areas that may deserve attention "
        "and positive habits that can be maintained."
    )

    st.caption(
        "The guidance below is general preventive information "
        "and is not an individualized medical or treatment plan."
    )


    # =====================================================
    # ATTENTION AREAS & POSITIVE HABITS
    # =====================================================

    attention_areas = []
    positive_habits = []


    # -----------------------------------------------------
    # BMI / WEIGHT MANAGEMENT
    # -----------------------------------------------------

    if BMI >= 25:

        attention_areas.append(
            (
                "⚖️ Weight Management",
                f"Your calculated BMI is {BMI:.1f}. "
                "Consider focusing on sustainable healthy-weight "
                "habits through balanced food choices and regular "
                "physical activity."
            )
        )

    elif BMI >= 18.5:

        positive_habits.append(
            f"⚖️ Your calculated BMI is {BMI:.1f}."
        )

    else:

        attention_areas.append(
            (
                "⚖️ Weight & Nutrition",
                f"Your calculated BMI is {BMI:.1f}. "
                "Consider discussing healthy weight and nutrition "
                "with an appropriate healthcare professional."
            )
        )


    # -----------------------------------------------------
    # PHYSICAL ACTIVITY
    # -----------------------------------------------------

    if PhysActivity == "No":

        attention_areas.append(
            (
                "🏃 Physical Activity",
                "You reported that you are not currently physically "
                "active. Consider gradually building regular physical "
                "activity into your routine according to your ability "
                "and health status."
            )
        )

    else:

        positive_habits.append(
            "🏃 You reported being physically active."
        )


    # -----------------------------------------------------
    # FRUIT INTAKE
    # -----------------------------------------------------

    if Fruits == "No":

        attention_areas.append(
            (
                "🍎 Fruit Intake",
                "You reported not regularly consuming fruit. "
                "Consider including a greater variety of nutritious "
                "foods, including suitable whole fruits."
            )
        )

    else:

        positive_habits.append(
            "🍎 You reported regular fruit consumption."
        )


    # -----------------------------------------------------
    # VEGETABLE INTAKE
    # -----------------------------------------------------

    if Veggies == "No":

        attention_areas.append(
            (
                "🥦 Vegetable Intake",
                "You reported not regularly consuming vegetables. "
                "Consider increasing the variety of vegetables "
                "in your overall eating pattern."
            )
        )

    else:

        positive_habits.append(
            "🥦 You reported regular vegetable consumption."
        )


    # -----------------------------------------------------
    # SMOKING
    # -----------------------------------------------------

    if Smoker == "Yes":

        attention_areas.append(
            (
                "🚭 Smoking",
                "You reported a history of smoking. Avoiding tobacco "
                "and seeking appropriate smoking-cessation support "
                "can be an important part of improving overall health."
            )
        )

    else:

        positive_habits.append(
            "🚭 You reported no smoking history."
        )


    # -----------------------------------------------------
    # HEAVY ALCOHOL CONSUMPTION
    # -----------------------------------------------------

    if HvyAlcoholConsump == "Yes":

        attention_areas.append(
            (
                "⚠️ Heavy Alcohol Consumption",
                "You reported heavy alcohol consumption. Consider "
                "discussing alcohol use with a qualified healthcare "
                "professional and following appropriate health guidance."
            )
        )

    else:

        positive_habits.append(
            "✅ No heavy alcohol consumption reported."
        )


    # -----------------------------------------------------
    # GENERAL HEALTH
    # -----------------------------------------------------

    if GenHlth in ["Poor", "Fair"]:

        attention_areas.append(
            (
                "❤️ General Health",
                f"You rated your general health as {GenHlth}. "
                "Regular health monitoring and discussion with a "
                "qualified healthcare professional may be useful."
            )
        )

    elif GenHlth in ["Excellent", "Very Good"]:

        positive_habits.append(
            f"❤️ You rated your general health as {GenHlth}."
        )


    # -----------------------------------------------------
    # PHYSICAL HEALTH DAYS
    # -----------------------------------------------------

    if PhysHlth >= 10:

        attention_areas.append(
            (
                "🩺 Physical Well-being",
                f"You reported {PhysHlth} poor physical-health days "
                "during the last 30 days. Consider discussing persistent "
                "physical-health concerns with a qualified healthcare "
                "professional."
            )
        )


    # -----------------------------------------------------
    # DIFFICULTY WALKING
    # -----------------------------------------------------

    if DiffWalk == "Yes":

        attention_areas.append(
            (
                "🚶 Mobility",
                "You reported difficulty walking. Physical activity "
                "recommendations should take your mobility and current "
                "health status into account."
            )
        )


    # =====================================================
    # AREAS THAT MAY NEED ATTENTION
    # =====================================================

    st.subheader("⚠️ Areas That May Need Attention")

    if len(attention_areas) == 0:

        st.success(
            "No major modifiable lifestyle concerns were identified "
            "from the selected factors."
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

    st.subheader("🥗 Healthy Eating Guidance")

    if Fruits == "No" and Veggies == "No":

        st.warning(
            "Your responses indicate that both fruit and vegetable "
            "intake may need attention."
        )

    elif Fruits == "No":

        st.warning(
            "Your responses indicate that fruit intake may need attention."
        )

    elif Veggies == "No":

        st.warning(
            "Your responses indicate that vegetable intake may need attention."
        )

    else:

        st.success(
            "You reported consuming both fruits and vegetables. "
            "Continue maintaining a balanced eating pattern."
        )


    st.markdown(
        """
**General healthy-eating principles:**

- Include a variety of vegetables and suitable whole fruits.
- Choose whole grains and other higher-fibre foods where appropriate.
- Include appropriate protein sources such as pulses, beans, eggs,
  fish, or other suitable options.
- Limit frequent intake of highly processed foods and foods or drinks
  high in added sugars.
- Pay attention to portion sizes and overall dietary balance.
- Choose water instead of sugar-sweetened drinks when possible.
        """
    )


    # =====================================================
    # PHYSICAL ACTIVITY GUIDANCE
    # =====================================================

    st.subheader(
        "🏃 Physical Activity & Lifestyle Guidance"
    )

    if DiffWalk == "Yes":

        st.warning(
            "You reported difficulty walking. Consider physical "
            "activities appropriate for your mobility and health "
            "status, and seek professional advice before making "
            "major changes to your activity routine."
        )

    elif PhysActivity == "No":

        st.warning(
            "You reported no regular physical activity. Consider "
            "gradually building regular physical activity into your "
            "routine according to your current health and ability."
        )

    else:

        st.success(
            "You reported being physically active. Maintaining "
            "regular physical activity is a positive lifestyle habit."
        )


    # =====================================================
    # PRIORITY ACTION PLAN
    # =====================================================

    st.subheader("🎯 Your Priority Action Plan")

    priorities = []


    if BMI >= 25:

        priorities.append(
            "⚖️ Focus on sustainable healthy-weight habits."
        )

    elif BMI < 18.5:

        priorities.append(
            "⚖️ Consider appropriate guidance regarding healthy "
            "weight and nutrition."
        )


    if PhysActivity == "No":

        priorities.append(
            "🏃 Gradually build regular physical activity "
            "according to your ability."
        )


    if Fruits == "No" or Veggies == "No":

        priorities.append(
            "🥗 Improve the variety and overall quality "
            "of your eating pattern."
        )


    if Smoker == "Yes":

        priorities.append(
            "🚭 Consider appropriate support for smoking cessation."
        )


    if HvyAlcoholConsump == "Yes":

        priorities.append(
            "⚠️ Address heavy alcohol consumption with "
            "appropriate professional support."
        )


    if GenHlth in ["Poor", "Fair"]:

        priorities.append(
            "❤️ Consider regular health monitoring and "
            "professional guidance."
        )


    if DiffWalk == "Yes":

        priorities.append(
            "🚶 Consider mobility-appropriate activity "
            "and professional guidance."
        )


    if len(priorities) == 0:

        st.success(
            "Continue maintaining the positive lifestyle habits "
            "you have reported."
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

    st.subheader("💚 Positive Habits to Maintain")

    if len(positive_habits) > 0:

        for habit in positive_habits:

            st.write(
                f"• {habit}"
            )

    else:

        st.write(
            "Use the areas identified above as starting points "
            "for gradual lifestyle improvement."
        )


    # =====================================================
    # RECOMMENDED NEXT STEP
    # =====================================================

    st.subheader("👩‍⚕️ Recommended Next Step")

    if prediction == 1:

        st.warning(
            "The model estimated a higher likelihood of diabetes risk. "
            "Consider discussing your overall risk profile with a "
            "qualified healthcare professional, who can determine "
            "whether appropriate clinical assessment or testing is needed."
        )

    else:

        st.success(
            "The model estimated a lower likelihood of diabetes risk. "
            "Continue maintaining healthy lifestyle habits and routine "
            "healthcare as appropriate."
        )


    # =====================================================
    # FINAL DISCLAIMER
    # =====================================================

    st.info(
        "ℹ️ This guidance is intended for educational and preventive "
        "information only. It does not diagnose diabetes and does not "
        "replace individualized advice from a qualified healthcare professional."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🩺 Diabetes Risk Prediction System | "
    "Machine Learning-Based Final Year Project"
)
