import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# ---------------------------------------
# Streamlit Page Setup
# ---------------------------------------
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Student Score Prediction System")
st.write("Predict student score based on study hours, attendance and assignments.")

# ---------------------------------------
# Load Dataset
# ---------------------------------------
DATA_PATH = "student_scores (1).csv"

df = pd.read_csv(DATA_PATH)

with st.expander("📄 Show Dataset"):
    st.dataframe(df)

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
# Sidebar Input
# ---------------------------------------
st.sidebar.head
