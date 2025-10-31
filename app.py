import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Heart Disease Predictor 💓")

st.title("💓 Cloud-Based Heart Disease Prediction System")
st.markdown("Predict the likelihood of heart disease using ML on AWS Cloud.")

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.sidebar.header("Enter Patient Details")

age = st.sidebar.number_input("Age", 1, 120)
sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.sidebar.number_input("Resting BP (mm Hg)", 80, 200)
chol = st.sidebar.number_input("Cholesterol (mg/dl)", 100, 600)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
restecg = st.sidebar.selectbox("Resting ECG", [0,1,2])
thalach = st.sidebar.number_input("Max Heart Rate", 60, 250)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 10.0)
slope = st.sidebar.selectbox("Slope", [0,1,2])
ca = st.sidebar.selectbox("Major Vessels (0–3)", [0,1,2,3])
thal = st.sidebar.selectbox("Thalassemia (0–3)", [0,1,2,3])

if st.button("Predict"):
    user_data = np.array([[age, 1 if sex=="Male" else 0, cp, trestbps, chol,
                           fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)
    st.success("💔 Heart Disease Detected" if prediction[0]==1 else "❤️ No Heart Disease")
