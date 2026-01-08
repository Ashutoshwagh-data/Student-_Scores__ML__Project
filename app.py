import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open('Student_model (1).pkl', 'rb'))

st.set_page_config(page_title="Student Score Prediction", layout="centered")

st.title("🎓 Student Score Prediction App")
st.write("Enter student score details")

# ---- INPUT FIELDS (Student Scores) ----
math_score = st.number_input("Math Score", min_value=0, max_value=100)
reading_score = st.number_input("Reading Score", min_value=0, max_value=100)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100)

if st.button("Predict Performance"):
    input_data = np.array([[math_score, reading_score, writing_score]])
    prediction = model.predict(input_data)

    st.success(f"📊 Predicted Result: {prediction[0]}")

