import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Streamlit page config
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="centered"
)

# ---- CLEAN & SAFE CSS ----
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f0f7ff 0%, #f7fff2 100%);
    font-family: 'Segoe UI', sans-serif;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 8px 30px rgba(13,38,76,0.08);
    border-left: 6px solid #2b6cb0;
}
</style>
""", unsafe_allow_html=True)

MODEL_FILE = "Student_model.pkl"
CSV_PATH = "/mnt/data/student_scores (1).csv"

# Train model if not found
def train_model():
    df = pd.read_csv(CSV_PATH)

    X = df[['Hours_Studied', 'Attendance', 'Assignments_Submitted']]
    y = df['Score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=43
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    return model

# Load or train
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
else:
    model = train_model()

# ---- UI HEADER ----
st.markdown("<h1 style='text-align:center;'>🎓 Student Score Predictor</h1>", unsafe_allow_html=True)

# ---- INPUT FORM ----
st.markdown("<div class='card'>", u
