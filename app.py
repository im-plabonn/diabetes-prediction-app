import streamlit as st
import joblib
import numpy as np

# Title & Description
st.title("🩺 Disease Diagnosis - Diabetes Prediction")
st.write("Enter patient details to predict Diabetes likelihood using SVM Model.")

# Form Input Fields
st.header("Patient Data Input")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=99, value=20)
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0)
pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)

# Load Trained Model
model = joblib.load('diabetes_model.pkl')

# Predict Button
if st.button("Predict Outcome"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ Result: High Risk of Diabetes (Positive)")
    else:
        st.success("✅ Result: Low Risk / No Diabetes (Negative)")