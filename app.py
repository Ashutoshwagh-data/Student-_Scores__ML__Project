import streamlit as st
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# -------------------------
# EMBEDDED CSV
# -------------------------
csv_data = """
Hours,Scores
1.5,20
2.0,22
2.5,30
3.0,35
3.5,40
4.0,45
4.5,50
5.0,55
5.5,60
6.0,65
6.5,70
7.0,75
7.5,80
8.0,85
8.5,90
9.0,95
9.5,98
"""

df = pd.read_csv(StringIO(csv_data))

# -------------------------
# BEAUTIFUL UI SETTINGS
# -------------------------
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="centered"
)

st.markdown(
    """
    <h1 style="text-align:center; color:#4CAF50;">🎓 Student Score Prediction App</h1>
    <p style="text-align:center; font-size:18px;">
    Predict student marks based on study hours using advanced ML models.
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Model Training Section
# -------------------------
X = df[['Hours']]
y = df['Scores']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create models
linear_model = LinearRegression()

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(poly.fit_transform(X_train), y_train)

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select ML Model:",
    ("Linear Regression", "Polynomial Regression (Best Fit)", "Random Forest (High Accuracy)")
)

hours = st.sidebar.slider("Study Hours", 0.0, 12.0, 5.0, 0.5)

# -------------------------
# Prediction Logic
# -------------------------
if model_choice == "Linear Regression":
    prediction = linear_model.fit(X_train, y_train).predict([[hours]])[0]

elif model_choice == "Polynomial Regression (Best Fit)":
    prediction = poly_model.predict(poly.transform([[hours]]))[0]

else:
    prediction = rf_model.predict([[hours]])[0]

# -------------------------
# Output Card
# -------------------------
st.markdown(
    f"""
    <div style="
        background-color:#e8f5e9;
        padding:20px;
        border-radius:15px;
        box-shadow:0px 4px 10px rgba(0,0,0,0.1);
        margin-top:20px;
        text-align:center;
    ">
        <h2 style="color:#2e7d32;">📘 Predicted Score</h2>
        <p style="font-size:28px; font-weight:bold; color:#1b5e20;">{prediction:.2f}</p>
        <p style="font-size:16px;">Using <b>{model_choice}</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

# Show Dataset
st.subheader("📊 Dataset")
st.dataframe(df)
