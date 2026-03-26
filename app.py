"""
app.py  ─  Heart Disease Risk Predictor
Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import joblib
import json
import os

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# LOAD MODEL & SCALER
# ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
COLS_PATH   = os.path.join(BASE_DIR, "models", "feature_cols.json")

@st.cache_resource
def load_artifacts():
    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    with open(COLS_PATH) as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols

try:
    model, scaler, feature_cols = load_artifacts()
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please run `python src/train_model.py` first.")
    st.stop()

NUMERICAL_FEATURES = ["BMI", "MentHlth", "PhysHlth"]

# ──────────────────────────────────────────────────────────────
# HELPER: preprocess user input
# ──────────────────────────────────────────────────────────────
def preprocess_input(user_dict):
    row = np.array([float(user_dict[f]) for f in feature_cols]).reshape(1, -1)
    num_idx = [feature_cols.index(f) for f in NUMERICAL_FEATURES if f in feature_cols]
    row[:, num_idx] = scaler.transform(row[:, num_idx])
    return row

# ──────────────────────────────────────────────────────────────
# HELPER: Risk level from probability
# ──────────────────────────────────────────────────────────────
def get_risk_level(prob):
    if prob < 0.25:
        return "🟢 LOW RISK",    "green",  "Low"
    elif prob < 0.55:
        return "🟡 MEDIUM RISK", "orange", "Medium"
    else:
        return "🔴 HIGH RISK",   "red",    "High"

# ──────────────────────────────────────────────────────────────
# SIDEBAR — Health Information Panel
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📚 Health Reference Guide")
    st.markdown("Use this panel to understand your health values before filling the form.")

    with st.expander("📊 BMI Categories", expanded=True):
        st.markdown("""
| Category | BMI Range |
|---|---|
| Underweight | < 18.5 |
| ✅ Normal | 18.5 – 24.9 |
| ⚠️ Overweight | 25.0 – 29.9 |
| 🔴 Obese | ≥ 30.0 |

**Formula:** BMI = weight(kg) ÷ height(m)²

*Example:* 70 kg, 1.75 m → 70 ÷ (1.75²) = **22.9 → Normal**
        """)

    with st.expander("🩸 Blood Pressure Guide"):
        st.markdown("""
| Category | Systolic | Diastolic |
|---|---|---|
| ✅ Normal | < 120 | < 80 |
| ⚠️ Elevated | 120–129 | < 80 |
| 🔴 High (Stage 1) | 130–139 | 80–89 |
| 🔴 High (Stage 2) | ≥ 140 | ≥ 90 |

*If your doctor has told you that you have high blood pressure, select **Yes** for High BP.*
        """)

    with st.expander("🧪 Cholesterol Guide"):
        st.markdown("""
| Category | Total Cholesterol |
|---|---|
| ✅ Desirable | < 200 mg/dL |
| ⚠️ Borderline High | 200–239 mg/dL |
| 🔴 High | ≥ 240 mg/dL |

*If your last blood test showed high cholesterol OR a doctor told you, select **Yes** for High Cholesterol.*
        """)

    with st.expander("🍺 Heavy Alcohol Definition"):
        st.markdown("""
**Heavy Drinking (CDC definition):**
- Men: More than **14 drinks per week**
- Women: More than **7 drinks per week**
        """)

    st.markdown("---")
    st.caption("⚠️ This tool is for educational purposes only. Always consult a healthcare professional.")

# ──────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("""
This tool estimates your risk of heart disease based on your lifestyle and health indicators.
Fill in the form below and click **Predict My Risk**.

> 📌 *All fields are required. Use the sidebar for help understanding any value.*
""")

st.divider()

# ──────────────────────────────────────────────────────────────
# INPUT FORM — Organised into logical sections
# ──────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    # ── SECTION 1: Basic Information ──
    st.subheader("👤 Basic Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        sex = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male",
            help="Your biological sex."
        )

    with col2:
        age = st.selectbox(
            "Age Group",
            options=list(range(1, 14)),
            format_func=lambda x: {
                1:"18–24", 2:"25–29", 3:"30–34", 4:"35–39", 5:"40–44",
                6:"45–49", 7:"50–54", 8:"55–59", 9:"60–64", 10:"65–69",
                11:"70–74", 12:"75–79", 13:"80+"
            }[x],
            index=4,
            help="Select your age group."
        )

    with col3:
        bmi = st.number_input(
            "BMI (Body Mass Index)",
            min_value=10.0,
            max_value=100.0,
            value=25.0,
            step=0.5,
            help="See the sidebar for how to calculate your BMI. Normal range: 18.5–24.9."
        )

    # BMI indicator
    if bmi < 18.5:
        st.info("📊 Your BMI indicates: **Underweight**")
    elif bmi < 25:
        st.success("📊 Your BMI indicates: **Normal Weight** ✅")
    elif bmi < 30:
        st.warning("📊 Your BMI indicates: **Overweight** ⚠️")
    else:
        st.error("📊 Your BMI indicates: **Obese** 🔴")

    st.divider()

    # ── SECTION 2: Medical Conditions ──
    st.subheader("🏥 Medical History")
    col1, col2, col3 = st.columns(3)

    with col1:
        high_bp = st.radio(
            "Do you have High Blood Pressure?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Has a doctor ever told you that you have high blood pressure?"
        )

    with col2:
        high_chol = st.radio(
            "Do you have High Cholesterol?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Has a doctor ever told you that your blood cholesterol is high?"
        )

    with col3:
        chol_check = st.radio(
            "Cholesterol Check in last 5 years?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Have you had your cholesterol checked within the last 5 years?"
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        stroke = st.radio(
            "Have you ever had a Stroke?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

    with col5:
        diabetes = st.selectbox(
            "Diabetes Status",
            options=[0, 1, 2],
            format_func=lambda x: {0: "No Diabetes", 1: "Pre-Diabetic", 2: "Diabetic"}[x],
            help="0 = No diabetes, 1 = Pre-diabetes or borderline diabetes, 2 = Diabetes."
        )

    with col6:
        diff_walk = st.radio(
            "Difficulty Walking or Climbing Stairs?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Do you have serious difficulty walking or climbing stairs?"
        )

    st.divider()

    # ── SECTION 3: Lifestyle ──
    st.subheader("🏃 Lifestyle & Habits")
    col1, col2, col3 = st.columns(3)

    with col1:
        smoker = st.radio(
            "Have you smoked at least 100 cigarettes in your life?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="This includes cigars, pipes, and electronic cigarettes."
        )

    with col2:
        phys_activity = st.radio(
            "Physical Activity in last 30 days?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Any physical activity or exercise (not counting your job)?"
        )

    with col3:
        hvy_alcohol = st.radio(
            "Heavy Alcohol Consumption?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Men: >14 drinks/week. Women: >7 drinks/week. See sidebar."
        )

    col4, col5 = st.columns(2)

    with col4:
        fruits = st.radio(
            "Do you eat Fruits at least once per day?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

    with col5:
        veggies = st.radio(
            "Do you eat Vegetables at least once per day?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

    st.divider()

    # ── SECTION 4: Health Status ──
    st.subheader("💭 Overall Health Status")
    col1, col2, col3 = st.columns(3)

    with col1:
        gen_hlth = st.select_slider(
            "General Health (1 = Excellent, 5 = Poor)",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: {1:"1 - Excellent", 2:"2 - Very Good", 3:"3 - Good",
                                     4:"4 - Fair", 5:"5 - Poor"}[x],
            help="How would you say that in general your health is?"
        )

    with col2:
        ment_hlth = st.slider(
            "Poor Mental Health Days (last 30 days)",
            min_value=0, max_value=30, value=0,
            help="How many days during the past 30 days was your mental health not good?"
        )

    with col3:
        phys_hlth = st.slider(
            "Poor Physical Health Days (last 30 days)",
            min_value=0, max_value=30, value=0,
            help="How many days during the past 30 days was your physical health not good?"
        )

    st.divider()

    # ── SECTION 5: Healthcare Access ──
    st.subheader("🏨 Healthcare Access")
    col1, col2 = st.columns(2)

    with col1:
        any_healthcare = st.radio(
            "Do you have any Healthcare Coverage?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Health insurance, prepaid plans, government plans, etc."
        )

    with col2:
        no_doc_cost = st.radio(
            "Skipped Doctor Visit due to Cost?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?"
        )

    st.divider()

    # ── SECTION 6: Education & Income ──
    st.subheader("📋 Demographics")
    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox(
            "Highest Education Level",
            options=list(range(1, 7)),
            format_func=lambda x: {
                1:"Never attended school", 2:"Grades 1–8",
                3:"Grades 9–11", 4:"Grade 12 / GED",
                5:"Some College", 6:"College Graduate"
            }[x],
            index=3,
        )

    with col2:
        income = st.selectbox(
            "Household Annual Income",
            options=list(range(1, 9)),
            format_func=lambda x: {
                1:"< $10,000", 2:"$10,000–$14,999", 3:"$15,000–$19,999",
                4:"$20,000–$24,999", 5:"$25,000–$34,999",
                6:"$35,000–$49,999", 7:"$50,000–$74,999", 8:"≥ $75,000"
            }[x],
            index=4,
        )

    st.divider()

    # ── SUBMIT ──
    submitted = st.form_submit_button(
        "🔍 Predict My Risk",
        use_container_width=True,
        type="primary"
    )

# ──────────────────────────────────────────────────────────────
# PREDICTION & OUTPUT
# ──────────────────────────────────────────────────────────────
if submitted:
    user_input = {
        "HighBP": high_bp,
        "HighChol": high_chol,
        "CholCheck": chol_check,
        "BMI": bmi,
        "Smoker": smoker,
        "Stroke": stroke,
        "Diabetes": diabetes,
        "PhysActivity": phys_activity,
        "Fruits": fruits,
        "Veggies": veggies,
        "HvyAlcoholConsump": hvy_alcohol,
        "AnyHealthcare": any_healthcare,
        "NoDocbcCost": no_doc_cost,
        "GenHlth": gen_hlth,
        "MentHlth": ment_hlth,
        "PhysHlth": phys_hlth,
        "DiffWalk": diff_walk,
        "Sex": sex,
        "Age": age,
        "Education": education,
        "Income": income,
    }

    # ── Validate all keys match feature_cols ──
    missing = [f for f in feature_cols if f not in user_input]
    if missing:
        st.error(f"Missing inputs: {missing}. Please contact the developer.")
        st.stop()

    try:
        X = preprocess_input(user_input)
        prob = float(model.predict_proba(X)[0][1])
        risk_label, risk_color, risk_short = get_risk_level(prob)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    st.divider()
    st.header("📋 Your Results")

    # ── Risk Meter ──
    col1, col2 = st.columns([1, 2])

    with col1:
        pct = int(prob * 100)
        st.markdown(f"""
<div style='text-align:center; padding: 20px; border-radius: 12px;
     background: {"#ffe0e0" if risk_color=="red" else "#fff8e0" if risk_color=="orange" else "#e0ffe0"};
     border: 2px solid {risk_color};'>
  <h1 style='color:{risk_color}; margin:0;'>{risk_label}</h1>
  <h2 style='color:{risk_color}; margin:5px 0;'>{pct}%</h2>
  <p style='color:gray; margin:0;'>Estimated Heart Disease Risk</p>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.progress(min(prob, 1.0))

        if risk_short == "Low":
            st.success("""
### 🟢 Low Risk — Great news!
Your health indicators suggest a **low risk** of heart disease.

**What this means:** Your lifestyle and medical history show relatively few risk factors.
Keep up your healthy habits — they're making a real difference.
            """)
        elif risk_short == "Medium":
            st.warning("""
### 🟡 Medium Risk — Pay Attention
Your health indicators suggest a **moderate risk** of heart disease.

**What this means:** Some of your risk factors are elevated. This doesn't mean
you will develop heart disease, but it's worth paying attention to your health.
Consider discussing these results with your doctor.
            """)
        else:
            st.error("""
### 🔴 High Risk — Please Consult a Doctor
Your health indicators suggest a **high risk** of heart disease.

**What this means:** Multiple risk factors are present. This is a screening tool,
not a diagnosis — but we strongly recommend consulting a healthcare professional soon.
            """)

    st.divider()

    # ── Risk Factor Summary ──
    st.subheader("🔍 Your Key Risk Factors")
    st.markdown("Below are the risk factors identified from your inputs:")

    risk_factors   = []
    positive_factors = []

    if high_bp == 1:     risk_factors.append("🔴 **High Blood Pressure** — A major risk factor for heart disease")
    if high_chol == 1:   risk_factors.append("🔴 **High Cholesterol** — Increases plaque buildup in arteries")
    if smoker == 1:      risk_factors.append("🔴 **Smoking history** — Significantly raises heart disease risk")
    if stroke == 1:      risk_factors.append("🔴 **Previous Stroke** — Indicates existing vascular disease")
    if diabetes == 2:    risk_factors.append("🔴 **Diabetes** — Doubles the risk of heart disease")
    if diabetes == 1:    risk_factors.append("🟡 **Pre-Diabetes** — Increases risk if unmanaged")
    if bmi >= 30:        risk_factors.append(f"🔴 **Obesity** (BMI={bmi:.1f}) — High BMI strains the heart")
    elif bmi >= 25:      risk_factors.append(f"🟡 **Overweight** (BMI={bmi:.1f}) — Slight increased risk")
    if hvy_alcohol == 1: risk_factors.append("🟡 **Heavy Alcohol Use** — Can raise blood pressure and damage heart")
    if diff_walk == 1:   risk_factors.append("🟡 **Difficulty Walking** — May indicate cardiovascular deconditioning")
    if gen_hlth >= 4:    risk_factors.append("🟡 **Poor General Health** — Associated with higher disease risk")
    if phys_hlth >= 15:  risk_factors.append(f"🟡 **Frequent Physical Health Issues** ({phys_hlth} days/month)")
    if phys_activity == 0: risk_factors.append("🟡 **No Physical Activity** — Sedentary lifestyle increases risk")

    if high_bp == 0:       positive_factors.append("✅ No high blood pressure")
    if high_chol == 0:     positive_factors.append("✅ No high cholesterol")
    if smoker == 0:        positive_factors.append("✅ Non-smoker")
    if phys_activity == 1: positive_factors.append("✅ Physically active")
    if fruits == 1:        positive_factors.append("✅ Eats fruits daily")
    if veggies == 1:       positive_factors.append("✅ Eats vegetables daily")
    if hvy_alcohol == 0:   positive_factors.append("✅ No heavy alcohol use")
    if 18.5 <= bmi < 25:   positive_factors.append(f"✅ Healthy BMI ({bmi:.1f})")
    if diabetes == 0:      positive_factors.append("✅ No diabetes")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**⚠️ Risk Factors Present:**")
        if risk_factors:
            for r in risk_factors:
                st.markdown(f"- {r}")
        else:
            st.markdown("- None identified 🎉")

    with col2:
        st.markdown("**💚 Protective Factors:**")
        for p in positive_factors:
            st.markdown(f"- {p}")

    st.divider()

    # ── General Suggestions (NOT medical advice) ──
    st.subheader("💡 General Wellness Suggestions")
    st.info("⚠️ These are general health tips, **NOT** medical advice. Always consult your doctor.")

    suggestions = []
    if phys_activity == 0:
        suggestions.append("🏃 **Get moving:** Aim for 30 minutes of moderate exercise most days (walking, cycling).")
    if fruits == 0 or veggies == 0:
        suggestions.append("🥗 **Improve diet:** Include more fruits and vegetables — aim for 5 servings a day.")
    if smoker == 1:
        suggestions.append("🚭 **Consider quitting smoking:** It's the single most impactful change for heart health.")
    if high_bp == 1:
        suggestions.append("🧂 **Monitor blood pressure:** Reduce sodium intake and discuss medication options with your doctor.")
    if high_chol == 1:
        suggestions.append("🥑 **Watch cholesterol:** Reduce saturated fats; omega-3 rich foods (fish, flaxseed) can help.")
    if bmi >= 25:
        suggestions.append("⚖️ **Weight management:** Even a 5–10% reduction in body weight can significantly lower heart risk.")
    if hvy_alcohol == 1:
        suggestions.append("🍺 **Reduce alcohol:** Cutting down to moderate levels can lower blood pressure.")
    if ment_hlth >= 14:
        suggestions.append("🧠 **Mental health matters:** Chronic stress and poor mental health increase heart risk. Consider speaking with a counselor.")

    if not suggestions:
        suggestions.append("🌟 You're doing great! Keep maintaining your healthy lifestyle.")

    for s in suggestions:
        st.markdown(f"- {s}")

    st.divider()
    st.caption("""
**Disclaimer:** This prediction is based on a machine learning model trained on the BRFSS 2015 survey data.
It is intended for **educational and research purposes only** and should not replace professional medical advice,
diagnosis, or treatment. The model has a recall of ~77%, meaning it may miss some cases.
Always consult a qualified healthcare professional for medical concerns.
    """)
