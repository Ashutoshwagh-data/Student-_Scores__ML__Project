import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Student Score Prediction",
    page_icon="📘",
    layout="centered"
)

# -------------------- HEADER UI --------------------
st.markdown("""
<style>
    .title {
        font-size: 38px;
        font-weight: 800;
        color: #1C4E80;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub {
        font-size: 18px;
        text-align: center;
        color: #4F6272;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>📘 Student Score Prediction App</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Simple • Clean • Attractive UI</div>", unsafe_allow_html=True)

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader("📤 Upload your CSV file (hours vs score)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")
    st.write("### 📄 Dataset Preview")
    st.dataframe(df)

    # -------------------- SCATTER PLOT (NO TRENDLINE) --------------------
    fig = px.scatter(df, x=df.columns[0], y=df.columns[1],
                     title="Study Hours vs Score")
    st.plotly_chart(fig)

    # -------------------- MODEL TRAINING --------------------
    X = df[[df.columns[0]]]
    y = df[df.columns[1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)

    # RMSE FIX
    mse = mean_squared_error(y_test, pred_test)
    rmse = mse ** 0.5

    st.info(f"📊 **Model RMSE:** {rmse:.2f}")

    # -------------------- PREDICTION --------------------
    st.write("### 🎯 Predict Student Score")
    hours = st.slider("Study Hours", 0, 12, 5)

    predicted_score = model.predict([[hours]])[0]

    st.success(f"📘 Predicted Score for {hours} hours: **{predicted_score:.2f}**")

else:
    st.warning("Please upload a CSV file to proceed.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("##### Developed by Ashutosh • ML Project • 2025")
