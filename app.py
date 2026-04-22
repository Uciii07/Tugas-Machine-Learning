import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved scaler, encoders, and models
@st.cache_resource
def load_artifacts():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('logistic_regression_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    with open('knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    return scaler, encoders, lr_model, svm_model, knn_model

scaler, encoders, lr_model, svm_model, knn_model = load_artifacts()

# Get class names from encoders for dropdown options
gender_options = list(encoders['Gender'].classes_)
education_options = list(encoders['Education'].classes_)
city_options = list(encoders['City'].classes_)
employment_options = list(encoders['EmploymentType'].classes_)

# Streamlit UI
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")
st.title("🏦 Loan Approval Prediction System")
st.markdown("---")

st.header("Applicant Information")

# Input fields
age = st.slider("Age", min_value=18, max_value=70, value=25)
income = st.number_input("Income", min_value=0.0, value=50000.0, step=1000.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0, value=20000.0, step=1000.0)
credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=600)
years_experience = st.slider("Years Experience", min_value=0, max_value=40, value=2)

gender_input = st.selectbox("Gender", options=gender_options)
education_input = st.selectbox("Education", options=education_options)
city_input = st.selectbox("City", options=city_options)
employment_input = st.selectbox("Employment Type", options=employment_options)

st.markdown("---")
st.header("Select Prediction Model")
model_choice = st.radio(
    "Choose a Machine Learning Model for Prediction:",
    ('Logistic Regression', 'SVM', 'KNN')
)

if st.button("🔍 Predict Loan Approval"):
    # Encode categorical features
    gender_encoded = encoders['Gender'].transform([gender_input])[0]
    education_encoded = encoders['Education'].transform([education_input])[0]
    city_encoded = encoders['City'].transform([city_input])[0]
    employment_encoded = encoders['EmploymentType'].transform([employment_input])[0]

    # Create a DataFrame for the input
    input_df = pd.DataFrame([[age, income, loan_amount, credit_score, years_experience,
                                gender_encoded, education_encoded, city_encoded, employment_encoded]],
                            columns=['Age', 'Income', 'LoanAmount', 'CreditScore', 'YearsExperience',
                                     'Gender', 'Education', 'City', 'EmploymentType'])

    # Scale numerical features
    input_scaled = scaler.transform(input_df)

    # Make prediction based on selected model
    if model_choice == 'Logistic Regression':
        prediction = lr_model.predict(input_scaled)[0]
    elif model_choice == 'SVM':
        prediction = svm_model.predict(input_scaled)[0]
    else: # KNN
        prediction = knn_model.predict(input_scaled)[0]

    st.markdown("--- Jardar")
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("✅ Pinjaman DISETUJUI")
    else:
        st.error("❌ Pinjaman DITOLAK")

    st.write(f"Prediction made using **{model_choice}** model.")
