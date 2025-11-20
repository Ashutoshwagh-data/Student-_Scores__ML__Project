import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS for attractive UI ---
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0f7ff 0%, #f7fff2 100%);
        font-family: 'Segoe UI', sans-serif;
    }
    .header {
        text-align: center;
        padding-top: 8px;
        padding-bottom: 0px;
    }
    .card {
        background: white;
        padding: 22px;
        border-radius: 14px;
        box-shadow: 0 8px 30px rgba(13,38,76,0.08);
        border-left: 6px solid #2b6cb0;
    }
    .small {
        color: #5a6b7b;
        font-size: 14px;
    }
    .metric {
        font-weight: 700;
    }
