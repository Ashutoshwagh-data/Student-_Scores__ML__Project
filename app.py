import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("📘 Student Score Prediction App")

# ---- Embedded CSV Data As a DataFrame ----
data = {
    "Hours": [1.5, 3.2, 2.5, 5.1, 7.8, 6.9, 8.5, 3.0, 4.5, 9.0],
    "Attendance": [60, 65, 70, 75, 80, 78, 85, 68, 73, 90],
    "Scores": [20, 40, 35, 55, 78, 70, 90, 42, 50, 95]
}

df = pd.DataFrame(data)

# ---- Load Model ----
MODEL_PATH = "Student_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except:
    st.error("❌ Model file 'Student_model.pkl' not found. Upload it to the same folder as app.py")
    st.stop()

# ---- User Input ----
st.subheader("Enter Student Details")

hours = st.number_input("Study Hours", min_value=0.0, max_value=12.0, step=0.5)
att = st.number_input("Attendance (%)", min_value=0, max_value=100, step=1)

# ---- Predict Button ----
if st.button("Predict Score"):
    input_df = pd.DataFrame([[hours, att]], columns=["Hours", "Attendance"])

    predicted_score = model.predict(input_df)[0]

    st.success(f"🎯 Predicted Student Score: **{predicted_score:.2f}**")
