import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="Diabetes Data Mining & Prediction Pipeline",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Disease Diagnosis - Diabetes Prediction & Mining Pipeline")
st.write("An Interactive End-to-End Data Mining Web Application using the Pima Indians Diabetes Dataset.")

# ----------------- DATA LOADING & PREPROCESSING -----------------
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, names=columns)
    
    # Impute zero values (missing data) with column mean
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_clean = df.copy()
    df_clean[zero_cols] = df_clean[zero_cols].replace(0, np.nan)
    df_clean[zero_cols] = df_clean[zero_cols].fillna(df_clean[zero_cols].mean())
    
    return df, df_clean

df_raw, df = load_and_preprocess_data()

# Prepare features and target for classification
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

# Scale features
scaler_eval = StandardScaler()
X_scaled = scaler_eval.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVM Model for evaluation tab
svm_eval = SVC(kernel='rbf', C=1.0, random_state=42)
svm_eval.fit(X_train, y_train)

# ----------------- NAVIGATION SIDEBAR -----------------
st.sidebar.header("📌 Navigation Pipeline")
options = st.sidebar.radio(
    "Select Section:",
    [
        "1. Patient Prediction",
        "2. EDA & Visualizations",
        "3. K-Means Clustering Analysis",
        "4. SVM Model Evaluation",
        "5. Dataset Overview & Cleaning"
    ]
)

# ----------------- SECTION 1: PATIENT PREDICTION -----------------
if options == "1. Patient Prediction":
    st.header("🔮 Patient Data Input & Prediction")
    st.write("Enter patient metrics to test Diabetes likelihood using the trained SVM Model.")
    
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

    st.markdown("---")
    
    # Load Model and Scaler from files
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')

        if st.button("Predict Outcome", type="primary"):
            raw_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]])
            scaled_input = scaler.transform(raw_input)
            prediction = model.predict(scaled_input)
            
            if prediction[0] == 1:
                st.error("⚠️ **Result: High Risk of Diabetes (Positive)**")
            else:
                st.success("✅ **Result: Low Risk / No Diabetes (Negative)**")
    except Exception as e:
        st.warning("⚠️ Could not load `diabetes_model.pkl` or `scaler.pkl`. Defaulting to in-memory trained model.")
        if st.button("Predict Outcome (In-Memory Model)", type="primary"):
            raw_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]])
            scaled_input = scaler_eval.transform(raw_input)
            prediction = svm_eval.predict(scaled_input)
            
            if prediction[0] == 1:
                st.error("⚠️ **Result: High Risk of Diabetes (Positive)**")
            else:
                st.success("✅ **Result: Low Risk / No Diabetes (Negative)**")

# ----------------- SECTION 2: EDA & VISUALIZATIONS -----------------
elif options == "2. EDA & Visualizations":
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.write("Visualizing feature distributions, correlations, and key relationships.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader("2. Target Distribution (Outcome)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Outcome', data=df, palette="Set2", ax=ax)
        plt.xticks([0, 1], ['Non-Diabetic (0)', 'Diabetic (1)'])
        st.pyplot(fig)
        
    st.markdown("---")
    
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

# ----------------- SECTION 3: CLUSTERING ANALYSIS -----------------
elif options == "3. K-Means Clustering Analysis":
    st.header("📌 K-Means Clustering Analysis")
    st.write("Unsupervised pattern discovery using K-Means algorithm to group patient profiles.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Optimal K Selection (Elbow Method)")
        wcss = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_tmp.fit(X_scaled)
            wcss.append(kmeans_tmp.inertia_)
            
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(k_range, wcss, marker='o', linestyle='--', color='b')
        ax.set_title("Elbow Method For Optimal k")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("WCSS (Within-Cluster Sum of Square)")
        ax.grid(True)
        st.pyplot(fig)
        
    with col2:
        st.subheader("2. Cluster Visualization")
        selected_k = st.slider("Select Number of Clusters (k):", min_value=2, max_value=5, value=2)
        
        kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        df_clustered = df.copy()
        df_clustered['Cluster'] = clusters
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x='Glucose', y='BMI', hue='Cluster', data=df_clustered, palette='Set1', ax=ax, s=60)
        ax.set_title(f"K-Means Clustering (k={selected_k}) [Glucose vs BMI]")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Cluster Distribution & Summary")
    st.dataframe(df_clustered.groupby('Cluster')[['Glucose', 'BMI', 'Age', 'Outcome']].mean())

# ----------------- SECTION 4: MODEL EVALUATION -----------------
elif options == "4. SVM Model Evaluation":
    st.header("📈 SVM Classification Performance Evaluation")
    st.write("Evaluating the Support Vector Machine (SVM) model using standard metrics.")
    
    y_pred = svm_eval.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Display Score Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc * 100:.2f}%")
    col2.metric("Precision", f"{prec * 100:.2f}%")
    col3.metric("Recall", f"{rec * 100:.2f}%")
    col4.metric("F1-Score", f"{f1 * 100:.2f}%")
    
    st.markdown("---")
    
    col_cm, col_rep = st.columns(2)
    
    with col_cm:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['Non-Diabetic', 'Diabetic'], 
                    yticklabels=['Non-Diabetic', 'Diabetic'])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)
        
    with col_rep:
        st.subheader("Detailed Classification Report")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))

# ----------------- SECTION 5: DATASET OVERVIEW -----------------
elif options == "5. Dataset Overview & Cleaning":
    st.header("📋 Dataset Overview & Preprocessing")
    
    st.subheader("1. Cleaned Dataset Preview (First 10 Rows)")
    st.dataframe(df.head(10))
    
    st.subheader("2. Descriptive Statistics")
    st.dataframe(df.describe())
    
    st.subheader("3. Missing Values Imputation Summary")
    st.write("Original dataset contained zeros in biological metrics. These zero values were imputed using Column Means during preprocessing.")
    
    null_summary = pd.DataFrame({
        'Raw Dataset Zero Counts': (df_raw[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] == 0).sum(),
        'Cleaned Dataset Zero Counts': (df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] == 0).sum()
    })
    st.table(null_summary)
