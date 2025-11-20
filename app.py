import streamlit as st
import pickle
import numpy as np

MODEL_PATH = "Student_model.pkl"   # use new renamed file

st.title("Employee Feedback Prediction")

# Load model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"❌ Model file '{MODEL_PATH}' not found. Upload it to the same folder as app.py")
    st.stop()

# Inputs
age = st.number_input("Age", 18, 70, 21)
experience = st.number_input("Experience (years)", 0, 50, 5)
salary = st.number_input("Salary", 1000, 1000000, 50000)
department = st.selectbox("Department", ["HR", "Sales", "IT", "Finance", "Admin"])

# Encoding
dep_map = {"HR": 0, "Sales": 1, "IT": 2, "Finance": 3, "Admin": 4}
department_num = dep_map[department]

# Predict
if st.button("Predict Feedback"):
    inp = np.array([[age, experience, salary, department_num]])
    pred = model.predict(inp)
    st.success(f"Predicted Feedback: {pred[0]}")
