import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

st.title("📘 Student Score Prediction App")

# ---------------------------------------
# Load CSV
# ---------------------------------------
DATA_PATH = "student_scores (1).csv"

df = pd.read_csv(DATA_PATH)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------------------------
# Train Model
# ---------------------------------------
X = df[['Hours_Studied', 'Attendance', 'Assignments_Submitted']]
y = df['Score']

model = LinearRegression()
model.fit(X, y)

# Save model
with open("student_score_model.pkl", "wb") as f:
    pickle.dump(model, f)

# ---------------------------------------
# User Input
# ---------------------------------------
st.header("Enter Student Details")

hours = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
assignments = st.number_input("Assignments Submitted", min_value=0, max_value=20, value=7)

# ---------------------------------------
# Predict
# ---------------------------------------
if st.button("Predict Score"):
    input_df = pd.DataFrame([[hours, attendance, assignments]],
                            columns=["Hours_Studied", "Attendance", "Assignments_Submitted"])

    predicted_score = model.predict(input_df)[0]

    st.success(f"🎯 Predicted Student Score: **{predicted_score:.2f}_**
