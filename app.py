import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Student Score Predictor", page_icon="🎯", layout="centered")

# ---- Custom Style ----
st.markdown("""
<style>
.card {
    background: linear-gradient(135deg, #ffffff 0%, #f2f7ff 100%);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.title {
    font-size: 28px;
    font-weight: 700;
    color: #2c3e50;
}
.subtitle {
    font-size: 20px;
    font-weight: 600;
    color: #34495e;
}
.label {
    font-size: 16px;
    font-weight: 500;
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

# ---- Load model & Data ----
df = pd.read_csv("student_scores.csv")
model = pickle.load(open("student_model.pkl", "rb"))

# ---- Title ----
st.markdown('<div class="title">🎯 Student Score Predictor</div>', unsafe_allow_html=True)

# ---- Dataset ----
st.markdown('<div class="subtitle">📘 Dataset Preview</div>', unsafe_allow_html=True)
st.dataframe(df)

# ---- Input Card ----
st.markdown('<div class="subtitle">📥 Enter Details for Prediction</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
hours = st.number_input("📘 Hours Studied", min_value=0.0, max_value=24.0, step=0.5, format="%.2f")
attendance = st.number_input("📊 Attendance (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.2f")
assignments = st.number_input("📝 Assignments Submitted", min_value=0, max_value=20, step=1)
st.markdown('</div>', unsafe_allow_html=True)

# ---- Input Preview ----
st.markdown('<div class="subtitle">🔍 Input Preview</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="card">
<b>Hours Studied:</b> {hours} <br>
<b>Attendance:</b> {attendance}% <br>
<b>Assignments Submitted:</b> {assignments}
</div>
""", unsafe_allow_html=True)

# ---- Prediction ----
if st.button("🚀 Predict Score"):
    input_data = pd.DataFrame([[hours, attendance, assignments]],
                              columns=['Hours Studied', 'Attendance', 'Assignments Submitted'])
    pred = model.predict(input_data)[0]

    st.success(f"🎉 Predicted Score: **{pred:.2f}**")
