import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Student Score Prediction System")
st.write("Predict student performance based on study hours, attendance, and assignments submitted.")

# -----------------------------
# Load Dataset
# -----------------------------
DATA_PATH = "student_scores (1).csv"
df = pd.read_csv(DATA_PATH)

with st.expander("📄 Show Dataset Preview"):
    st.dataframe(df)

# -----------------------------
# Train Model
# -----------------------------
X = df[['Hours_Studied', 'Attendance', 'Assignments_Submitted']]
y = df['Score']

model = LinearRegression()
model.fit(X, y)

# Save model
with open("student_score_model.pkl", "wb") as f:
    pickle.dump(model, f)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Enter Student Details")

hours = st.sidebar.number_input("Hours Studied", min_value=0, max_value=24, value=5)
attendance = st.sidebar.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
assignments = st.sidebar.number_input("Assignments Submitted", min_value=0, max_value=20, value=7)

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict Score"):
    
    input_df = pd.DataFrame([[hours, att]()]()_
