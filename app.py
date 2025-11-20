# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
with open('Student_model (1).pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Employee Feedback Prediction")

# Input fields
age = st.number_input("Age", min_value=18, max_value=70, value=21)
experience = st.number_input("Experience (years)", min_value=0, max_value=50, value=5)
salary = st.number_input("Salary", min_value=1000, max_value=1000000, value=50000)
department = st.selectbox("Department", ["HR", "Sales", "IT", "Finance", "Admin"])

# Convert department to numeric (same mapping used during training)
dep_map = {"HR":0, "Sales":1, "IT":2, "Finance":3, "Admin":4}
department_num = dep_map[department]

if st.button("Predict Feedback"):
    input_data = np.array([[age, experience, salary, department_num]])
    pred = model.predict(input_data)
    st.success(f"Predicted Feedback: {pred[0]}")



