import streamlit as st
import joblib
import numpy as np
import os

# Page Config
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

# Load Model
model_path = os.path.join('model', 'breast_cancer_model.pkl')

@st.cache_resource
def load_model():
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error("Model file not found. Ensure 'breast_cancer_model.pkl' is in the 'model' directory.")
        return None

model = load_model()

# Header
st.title("🏥 Breast Cancer Prediction System")
st.write("Enter the tumor features below to predict if it is **Benign** or **Malignant**.")
st.info("Note: This tool is for educational purposes only and not for medical diagnosis.")
st.write("---")

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        radius_mean = st.number_input("Radius Mean", min_value=0.0, value=14.0, format="%.2f")
        texture_mean = st.number_input("Texture Mean", min_value=0.0, value=19.0, format="%.2f")
        perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, value=90.0, format="%.2f")
    
    with col2:
        area_mean = st.number_input("Area Mean", min_value=0.0, value=650.0, format="%.2f")
        smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, value=0.1, format="%.4f")
    
    submitted = st.form_submit_button("Analyze Tumor")

# Prediction Logic
if submitted and model:
    # Prepare input array (order must match training features)
    # ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
    input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]])
    
    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # Probability of Malignant
    
    st.write("---")
    if prediction == 1:
        st.error(f"**Prediction: MALIGNANT**")
        st.write(f"Confidence: {probability:.2%}")
        st.warning("Please consult a specialist immediately.")
    else:
        st.success(f"**Prediction: BENIGN**")
        st.write(f"Confidence: {1 - probability:.2%}")