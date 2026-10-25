import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Page Config
st.set_page_config(page_title="Diabetes Diagnosis & Mining Pipeline", layout="wide")

st.title("🩺 Disease Diagnosis - Diabetes Prediction & Data Mining")
st.write("End-to-End Data Mining Pipeline based on Pima Indians Diabetes Dataset")

# Sidebar / Navigation
st.sidebar.header("Navigation")
options = st.sidebar.radio("Go to section:", ["1. Prediction Model", "2. EDA & Visualizations", "3. Dataset Overview"])

# Load Dataset for Visualizations
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, names=columns)
    return df

df = load_data()

# ----------------- SECTION 1: PREDICTION MODEL -----------------
if options == "1. Prediction Model":
    st.header("Patient Data Input & Prediction")
    st.write("Enter patient metrics to test Diabetes likelihood using trained SVM Model.")
    
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=99, value=20)
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0)
        pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)

    # Load Model and Scaler
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')

        if st.button("Predict Outcome", type="primary"):
            raw_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]])
            scaled_input = scaler.transform(raw_input)
            prediction = model.predict(scaled_input)
            
            st.divider()
            if prediction[0] == 1:
                st.error("⚠️ Result: High Risk of Diabetes (Positive)")
            else:
                st.success("✅ Result: Low Risk / No Diabetes (Negative)")
    except Exception as e:
        st.warning("Please ensure 'diabetes_model.pkl' and 'scaler.pkl' are present in the repository.")

# ----------------- SECTION 2: EDA & VISUALIZATIONS -----------------
elif options == "2. EDA & Visualizations":
    st.header("Exploratory Data Analysis (EDA)")
    st.write("Visualizing feature distributions, correlations, and relationships from Lab 2 & 3.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader("2. Target Distribution (Outcome)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Outcome', data=df, palette="Set2", ax=ax)
        plt.xticks([0, 1], ['Non-Diabetic (0)', 'Diabetic (1)'])
        st.pyplot(fig)
        
    st.divider()
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("3. Glucose vs Outcome Boxplot")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='Outcome', y='Glucose', data=df, palette="Set1", ax=ax)
        plt.xticks([0, 1], ['Non-Diabetic', 'Diabetic'])
        st.pyplot(fig)
        
    with col4:
        st.subheader("4. Glucose vs Age Scatter Plot")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x='Age', y='Glucose', hue='Outcome', data=df, palette="viridis", ax=ax)
        st.pyplot(fig)

# ----------------- SECTION 3: DATASET OVERVIEW -----------------
elif options == "3. Dataset Overview":
    st.header("Pima Indians Diabetes Dataset Overview")
    st.dataframe(df.head(10))
    
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())
