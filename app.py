import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px

st.set_page_config(page_title="Student Score Prediction", page_icon="📘")

# ---------- HEADER ----------
st.markdown("""
<h1 style="text-align:center; color:#1C4E80;">📘 Student Score Prediction</h1>
<h4 style="text-align:center; color:#4F6272;">Enter Hours & Score • Train Model • Predict</h4>
""", unsafe_allow_html=True)

# ---------- NUMBER OF INPUT ROWS ----------
st.write("### ✏️ Enter number of rows:")
rows = st.number_input("How many data rows you want to enter?", min_value=1, max_value=50, value=5)

hours_list = []
score_list = []

st.write("### 🧮 Enter Data")

# ---------- INPUT BOXES ----------
for i in range(rows):
    col1, col2 = st.columns(2)
    with col1:
        h = st.number_input(f"Hours (Row {i+1})", min_value=0.0, max_value=24.0, key=f"h{i}")
    with col2:
        s = st.number_input(f"Score (Row {i+1})", min_value=0.0, max_value=100.0, key=f"s{i}")

    hours_list.append(h)
    score_list.append(s)

# ---------- TRAIN MODEL BUTTON ----------
if st.button("Train Model"):
    df = pd.DataFrame({"Hours": hours_list, "Score": score_list})
    st.success("✔ Data collected successfully!")

    st.write("### 📄 Dataset Preview")
    st.dataframe(df)

    # Graph
    fig = px.scatter(df, x="Hours", y="Score", title="Study Hours vs Score")
    st.plotly_chart(fig)

    # Train Model
    X = df[["Hours"]]
    y = df["Score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)

    mse = mean_squared_error(y_test, pred_test)
    rmse = mse ** 0.5

    st.info(f"📊 **Model RMSE:** {rmse:.2f}")

    # Prediction
    st.write("### 🎯 Predict Student Score")
    hours_input = st.slider("Study Hours", 0, 12, 5)
    predicted_score = model.predict([[hours_input]])[0]

    st.success(f"📘 Predicted Score for {hours_input} hours: **{predicted_score:.2f}**")

# Footer
st.markdown("---")
st.markdown("##### Developed by Ashutosh • ML Project • 2025")
