import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from fpdf import FPDF
import time

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Student Score Predictor (All features)",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------
# Helper functions
# -------------------------
def load_embedded_csv():
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
    return pd.read_csv(StringIO(csv_data))

def train_models(X_train, y_train, rf_n_estimators, poly_degree, progress_callback=None):
    """Train linear, polynomial (pipeline), and random forest models."""
    # Linear
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Polynomial pipeline: scaling -> poly features -> linear
    poly_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ("lr", LinearRegression())
    ])
    poly_pipe.fit(X_train, y_train)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=rf_n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    if progress_callback:
        progress_callback(100)

    return {"Linear": lr, "Polynomial": poly_pipe, "RandomForest": rf}

def evaluate_model(model, X_test, y_test, transform_func=None):
    """Return predictions and metrics."""
    if transform_func:
        preds = model.predict(transform_func(X_test))
    else:
        preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    return preds, rmse, r2

def plot_hours_vs_score(df):
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.scatter(df['Hours'], df['Scores'])
    ax.set_xlabel("Hours")
    ax.set_ylabel("Scores")
    ax.set_title("Hours vs Scores")
    plt.tight_layout()
    return fig

def plot_actual_vs_pred(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.scatter(y_true, y_pred)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel("Actual Scores")
    ax.set_ylabel("Predicted Scores")
    ax.set_title("Actual vs Predicted")
    plt.tight_layout()
    return fig

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.scatter(y_pred, residuals)
    ax.axhline(0, linestyle='--', color='red')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals (Actual - Pred)")
    ax.set_title("Residuals vs Predicted")
    plt.tight_layout()
    return fig

def create_pdf_report(prediction, hours, model_name, metrics, figs):
    """
    Create a PDF report containing:
    - input hours & prediction
    - model used
    - metrics table
    - embedded figures
    Returns bytes.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Student Score Prediction Report", ln=True, align="C")
    pdf.ln(6)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Input study hours: {hours}", ln=True)
    pdf.cell(0, 8, f"Model used: {model_name}", ln=True)
    pdf.cell(0, 8, f"Predicted score: {prediction:.2f}", ln=True)
    pdf.ln(6)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Metrics", ln=True)
    pdf.set_font("Arial", size=11)
    for k, v in metrics.items():
        pdf.cell(0, 7, f"{k}: {v:.4f}", ln=True)
    pdf.ln(6)

    # Add each figure as an image
    for i, fig in enumerate(figs):
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        pdf.image(buf, w=170)
        pdf.ln(6)
        buf.close()
    out = pdf.output(dest='S').encode('latin-1')
    return out

# -------------------------
# Sidebar (controls)
# -------------------------
st.sidebar.header("Settings & Controls")

# Theme toggle (light/dark via simple CSS)
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)
if theme == "Dark":
    st.markdown(
        """
        <style>
        .reportview-container, .sidebar .sidebar-content {
            background-color: #0f1722;
            color: #cbd5e1;
        }
        .stButton>button {color: #111827;}
        </style>
        """,
        unsafe_allow_html=True
    )

use_uploaded = st.sidebar.checkbox("Upload my CSV (override sample)", value=False)
uploaded_file = None
if use_uploaded:
    uploaded_file = st.sidebar.file_uploader("Upload CSV (must have columns: Hours, Scores)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("Model & Training")
model_choice = st.sidebar.selectbox("Choose Model", ["Linear Regression", "Polynomial Regression", "Random Forest"])
rf_estimators = st.sidebar.slider("RF: n_estimators", 50, 500, 200, step=50)
poly_deg = st.sidebar.slider("Polynomial degree (if chosen)", 2, 6, 3)
test_size = st.sidebar.slider("Test size (%)", 10, 40, 20)

st.sidebar.markdown("---")
st.sidebar.subheader("Prediction Input")
predict_hours = st.sidebar.number_input("Study hours for prediction", min_value=0.0, max_value=24.0, value=5.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("Extras")
download_model_flag = st.sidebar.checkbox("Allow download of trained model (.pkl)", value=True)
generate_pdf_flag = st.sidebar.checkbox("Allow PDF report generation", value=True)

# -------------------------
# Main UI
# -------------------------
st.title("🎓 Student Score Predictor — All features")
st.write("Comprehensive demo: upload data, train advanced models, view metrics, plots, download model & PDF report.")

# Load data (uploaded override or embedded)
if use_uploaded and uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Uploaded CSV loaded successfully.")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
else:
    df = load_embedded_csv()
    st.info("Using embedded sample dataset. (Toggle 'Upload my CSV' to use your file.)")

# Basic data checks
if not {"Hours", "Scores"}.issubset(df.columns):
    st.error("Dataset must contain 'Hours' and 'Scores' columns. Please upload correct CSV.")
    st.stop()

# Show dataset preview
with st.expander("Dataset (Preview)"):
    st.dataframe(df)

# Visual: Hours vs Score
st.subheader("📈 Hours vs Scores")
fig_hs = plot_hours_vs_score(df)
st.pyplot(fig_hs)

# Prepare data
X = df[["Hours"]].values
y = df["Scores"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42)

# Training button
if st.button("Train Models"):
    status_text = st.empty()
    progress = st.progress(0)
    status_text.info("Training started...")

    # Simulate incremental progress for UX
    for p in range(0, 20, 5):
        progress.progress(p)
        time.sleep(0.05)

    models = train_models(X_train, y_train, rf_n_estimators=rf_estimators, poly_degree=poly_deg,
                          progress_callback=lambda v: progress.progress(v))
    status_text.success("Training completed ✅")
    st.success("Models trained: Linear, Polynomial, RandomForest")

    # Evaluate chosen model
    chosen = model_choice
    if chosen == "Linear Regression":
        model = models["Linear"]
        preds, rmse, r2 = evaluate_model(model, X_test, y_test)
    elif chosen == "Polynomial Regression":
        model = models["Polynomial"]
        preds, rmse, r2 = evaluate_model(model, X_test, y_test, transform_func=lambda x: x)
        # pipeline handles transform internally
    else:
        model = models["RandomForest"]
        preds, rmse, r2 = evaluate_model(model, X_test, y_test)

    # Display metrics
    st.subheader("📊 Performance Metrics")
    st.metric("RMSE", f"{rmse:.4f}")
    st.metric("R² Score", f"{r2:.4f}")

    # Plots: Actual vs Predicted
    fig_avp = plot_actual_vs_pred(y_test, preds)
    st.pyplot(fig_avp)

    fig_res = plot_residuals(y_test, preds)
    st.pyplot(fig_res)

    # Predict for user input
    if model_choice == "Polynomial Regression":
        # pipeline handles transform internally
        pred_value = model.predict(np.array([[predict_hours]]))[0]
    else:
        pred_value = model.predict(np.array([[predict_hours]]))[0]

    st.markdown(
        f"""
        <div style="background-color:#e6fffa;padding:15px;border-radius:10px;margin-top:10px;">
            <h3 style="margin:0">🔮 Predicted Score for <b>{predict_hours}</b> study hours:</h3>
            <h2 style="margin:0;color:#065f46">{pred_value:.2f}</h2>
            <p style="margin:0;font-size:13px;color:#064e3b">Model used: <b>{model_choice}</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Allow download of trained model
    if download_model_flag:
        try:
            buf = BytesIO()
            pickle.dump(model, buf)
            buf.seek(0)
            st.download_button("⬇️ Download trained model (.pkl)", data=buf, file_name="trained_model.pkl")
        except Exception as e:
            st.warning(f"Could not create model download: {e}")

    # Allow PDF report download
    if generate_pdf_flag:
        # Create small set of figs for the report
        figs_for_pdf = [fig_hs, fig_avp, fig_res]
        metrics = {"RMSE": rmse, "R2": r2}
        pdf_bytes = create_pdf_report(prediction=pred_value, hours=predict_hours, model_name=model_choice,
                                      metrics=metrics, figs=figs_for_pdf)
        st.download_button("📄 Download PDF report", data=pdf_bytes, file_name="prediction_report.pdf", mime="application/pdf")

    # Show test-set table with predictions (optional)
    show_table = st.checkbox("Show Test set: Actual vs Predicted", value=False)
    if show_table:
        df_test = pd.DataFrame({"Hours": X_test.flatten(), "Actual": y_test, "Predicted": preds})
        st.dataframe(df_test.sort_values(by="Hours").reset_index(drop=True))

else:
    st.info("Press 'Train Models' to start training and see results.")

# Footer / Help
st.markdown("---")
st.write("Need more features? e.g., hyperparameter tuning UI, model comparison dashboard, or deployment guide — बोल आणि मी add करतो!")
