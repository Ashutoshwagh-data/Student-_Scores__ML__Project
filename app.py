import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "Student_model (2).pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

st.title("📘 Student / Employee Feedback Prediction App")

# -----------------------------
# User Input Fields
# -----------------------------
st.header("Enter Details")

age = st.number_input("Age", min_value=18, max_value=80, value=21)
experience = st.number_input("Experience (years)", min_value=0, max_value=40, value=1)
salary = st.number_input("Salary", min_value=1000, max_value=500000, value=30000)

department = st.selectbox("Department", ["HR", "IT", "Sales", "Finance"])

# Convert department to numeric (same as used during training)
dept_mapping = {"HR": 0, "IT": 1, "Sales": 2, "Finance": 3}
department_num = dept_mapping[department]

# -----------------------------
# Make Prediction
# -----------------------------
if st.button("Predict Feedback"):
    try:
        input_data = pd.DataFrame([[age, experience, salary, department_num]],
                                  columns=["Age", "Experience", "Salary", "Department"])

        prediction = model.predict(input_data)[0]

        st.success(f"🎯 Predicted Feedback: **{prediction}**")
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
