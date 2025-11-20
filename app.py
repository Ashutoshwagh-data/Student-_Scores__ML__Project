import streamlit as st
import pandas as pd
import pickle

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="centered"
)

# --- CUSTOM CSS (Premium UI) ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #c3eaff 0%, #e7ffe7 100%);
        font-family: 'Segoe UI';
    }
    .main-title {
        text-align: center;
        font-size: 42px;
        color: #0a3d62;
        font-weight: 800;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #2f3640;
        margin-bottom: 30px;
    }
    .card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0px 4px 25px rgba(0,0,0,0.1);
        border-left: 6px solid #0984e3;
    }
    .predict-btn button {
        background-color: #0984e3 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 18px;
    }
    </style>
""", unsafe_allow_html=True)

# --- TITLES ---
st.markdown("<div class='main-title'>🎓 Student Score Prediction Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Enter student details to predict the final exam score</div>", unsafe_allow_html=True)

# Load model
MODEL_PATH = "Student_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except:
    st.error("❌ Model file 'Student_model.pkl' not found.")
    st.stop()

# --- FORM CARD ---
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("📘 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    hours = st.number_input("📚 Hours Studied", 0.0, 24.0, 5.0)
    extra = st.selectbox("🎭 Extracurricular Activities", ["Yes", "No"])

with col2:
    previous_score = st.number_input("🧮 Previous Score", 0, 100, 70)
    sleep = st.number_input("😴 Sleep Hours", 0.0, 24.0, 7.0)

sample = st.number_input("📝 Sample Question Papers Practiced", 0, 50, 5)

st.markdown("</div>", unsafe_allow_html=True)

# Convert categorical to numeric
extra_numeric = 1 if extra == "Yes" else 0

input_df = pd.DataFrame([{
    "Hours_Studied": hours,
    "Previous_Scores": previous_score,
    "Extracurricular_Activities": extra_numeric,
    "Sleep_Hours": sleep,
    "Sample_Question_Papers_Practiced": sample
}])

# --- PREDICT BUTTON ---
st.markdown("<div class='predict-btn'>", unsafe_allow_html=True)
if st.button("🔮 Predict Score"):
    score = model.predict(input_df)[0]

    st.success(f"🎯 **Predicted Score: {score}**")
st.markdown("</div>", unsafe_allow_html=True)
