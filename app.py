import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------- Embedded CSV ------------
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

# ----------- Page UI -------------
st.set_page_config(page_title="Student Score Predictor", page_icon="🎓")

st.markdown(
    """
    <h1 style='text-align:center; color:#4CAF50;'>🎓 Student Score Prediction</h1>
    <p style='text-align:center; font-size:18px;'>Simple & beautiful ML app to predict scores using hours of study.</p>
    """,
    unsafe_allow_html=True
)

# ----------- Sidebar -------------
st.sidebar.header("⚙️ Model Settings")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Linear Regression", "Polynomial Regression", "Random Forest"]
)

degree = st.sidebar.slider("Polynomial Degree", 2, 6, 3)
hours_input = st.sidebar.number_input("Study Hours", min_value=0.0, max_value=12.0, value=5.0, step=0.5)

# ----------- Training -------------
X = df[["Hours"]]
y = df["Scores"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select model
if model_choice == "Linear Regression":
    model = LinearRegression()

elif model_choice == "Polynomial Regression":
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("lr", LinearRegression())
    ])

else:
    model = RandomForestRegressor(n_estimators=200, random_state=42)

model.fit(X_train, y_train)
pred_test = model.predict(X_test)

# Metrics
rmse = mean_squared_error(y_test, pred_test, squared=False)
r2 = r2_score(y_test, pred_test)

# ----------- Prediction -------------
pred_value = model.predict(np.array([[hours_input]]))[0]

st.markdown(
    f"""
    <div style="background:#e8f5e9; padding:15px; border-radius:10px; margin-top:10px;">
        <h3>📘 Predicted Score for <b>{hours_input}</b> hours:</h3>
        <h2 style="color:#2e7d32;">{pred_value:.2f}</h2>
        <p>Using <b>{model_choice}</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------- Metrics Display -------------
col1, col2 = st.columns(2)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("R² Score", f"{r2:.2f}")

# ----------- Graphs -------------
st.subheader("📊 Hours vs Scores")
fig1, ax1 = plt.subplots()
ax1.scatter(df["Hours"], df["Scores"])
ax1.set_xlabel("Hours")
ax1.set_ylabel("Scores")
st.pyplot(fig1)

st.subheader("📈 Actual vs Predicted")
fig2, ax2 = plt.subplots()
ax2.scatter(y_test, pred_test)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
ax2.set_xlabel("Actual")
ax2.set_ylabel("Predicted")
st.pyplot(fig2)
