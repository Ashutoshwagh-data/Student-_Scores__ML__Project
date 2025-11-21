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

st.markdown("""
<style>
    .main {
        background: #F8FAFF;
    }
    .title {
        font-size: 40px;
        font-weight: 800;
        color: #2A4D69;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub {
        font-size: 18px;
        text-align: center;
        color: #4F6272;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("<div class='title'>📘 Student Score Prediction App</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Simple • Clean • Attractive UI • Better ML Model • Graph Insights</div>", unsafe_allow_html=True)

# -------------------- FILE UPLOAD --------------------
st.write("### 📤 Upload your CSV file (hours vs score)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")
    st.write("### 📄 Dataset Preview")
    st.dataframe(df)

    # -------------------- GRAPH --------------------
    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], trendline="ols",
                     title="Study Hours vs Score Relationship")
    st.plotly_chart(fig)

    # -------------------- MODEL TRAINING --------------------
    X = df[[df.columns[0]]]
    y = df[df.columns[1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)

    # FIXED RMSE (No squared=False)
    mse = mean_squared_error(y_test, pred_test)
    rmse = mse ** 0.5

    st.info(f"📊 **Model RMSE:** {rmse:.2f}")

    # -------------------- USER INPUT PREDICTION --------------------
    st.write("### 🎯 Predict Student Score")
    hours = st.slider("Study Hours", 0, 12, 5)

    predicted_score = model.predict([[hours]])[0]

    st.success(f"📘 Predicted Score for {hours} hours: **{predicted_score:.2f}**")

else:
    st.warning("Please upload a CSV file to proceed.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("##### Developed by Ashutosh • Machine Learning Project • 2025")
