import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Student Score Predictor",
    layout="wide",
    page_icon="🎓"
)

# -----------------------------------------------------------
# CSS THEME
# -----------------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
}
h1, h2, h3, h4 {
    color: white;
}
.card {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 6px 30px rgba(0,0,0,0.25);
    color: white;
}
.big-card {
    background: rgba(255,255,255,0.20);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.3);
}
.result {
    font-size: 36px;
    font-weight: 800;
    text-align: center;
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# TITLE
# -----------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🎓 Student Performance Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#dfe6e9;'>AI-based Academic Score Predictor</h4>", unsafe_allow_html=True)
st.write("")

# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
try:
    model = pickle.load(open("Student_model.pkl", "rb"))
except:
    st.error("❌ Model file 'Student_model.pkl' not found.")
    st.stop()

# -----------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------
st.sidebar.header("📁 Upload Data")
csv_file = st.sidebar.file_uploader("Upload Student Dataset (CSV)", type=["csv"])

if csv_file:
    df = pd.read_csv(csv_file)
    st.sidebar.success("Dataset Uploaded ✔")
else:
    df = None

# -----------------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Enter Student Attributes")
    
    hours = st.number_input("Hours Studied", 0, 12, 5)
    attendance = st.number_input("Attendance (%)", 0, 100, 80)
    assignments = st.number_input("Assignments Submitted", 0, 20, 10)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🤖 AI Prediction")
    
    if st.button("Predict Score"):
        input_df = pd.DataFrame([[hours, attendance, assignments]],
                                columns=["Hours_Studied", "Attendance", "Assignments_Submitted"])
        result = model.predict(input_df)[0]
        
        st.markdown(f"<p class='result'>Predicted Score: {result:.2f}</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# DATA ANALYTICS SECTION
# -----------------------------------------------------------
if df is not None:
    st.markdown("<br><div class='big-card'>", unsafe_allow_html=True)
    st.subheader("📈 Dataset Insights")
    
    st.write("### 🔍 Data Preview")
    st.dataframe(df, use_container_width=True)
    
    st.write("### 📊 Correlation Heatmap (Top Features)")
    corr = df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("### 📉 Distribution of Score")
    if "Score" in df.columns:
        fig2 = px.histogram(df, x="Score", nbins=20)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
