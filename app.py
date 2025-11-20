import streamlit as st
import pandas as pd
import pickle

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Student Score Predictor", layout="centered")

# -------------------------
# Attractive CSS
# -------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#cfd9ff,#e8f0ff);
}
.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.12);
}
h1 {
    text-align:center;
    color:#2c3e50;
    font-weight:800;
}
.label {
    font-size:18px;
    font-weight:600;
    color:#34495e;
}
.result-box {
    background:#4c84ff;
    padding:15px;
    border-radius:15px;
    color:white;
    font-size:22px;
    text-align:center;
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Title
# -------------------------
st.markdown("<h1>🎓 Student Score Prediction</h1>", unsafe_allow_html=True)
st.write("")

# -------------------------
# Load Model
# -------------------------
try:
    model = pickle.load(open("Student_model.pkl", "rb"))
except:
    st.error("❌ 'Student_model.pkl' file missing! Upload it beside app.py")
    st.stop()

# -------------------------
# Load Dataset (Preview Only)
# -------------------------
df = pd.read_csv("/mnt/data/student_scores (1).csv")

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📘 Dataset Preview")
st.dataframe(df, use_container_width=True)
st.markdown("</div><br>", unsafe_allow_html=True)

# -------------------------
# Input Section
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("✏️ Enter Student Details")

hours = st.number_input("Hours Studied", 0, 12, 5)
attendance = st.number_input("Attendance (%)", 0, 100, 80)
assignments = st.number_input("Assignments Submitted", 0, 20, 10)

st.markdown("</div><br>", unsafe_allow_html=True)

# -------------------------
# Predict Button
# -------------------------
if st.button("🎯 Predict Score"):
    input_df = pd.DataFrame([[hours, attendance, assignments]],
                            columns=["Hours_Studied", "Attendance", "Assignments_Submitted"])
    
    score = model.predict(input_df)[0]

    st.markdown(f"<div class='result-box'>Predicted Score: {score:.2f}</div>",
                unsafe_allow_html=True)

