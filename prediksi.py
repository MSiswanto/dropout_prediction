import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Dropout", layout="wide")

# Load model dan data
model = joblib.load("dropout_model.pkl")
df = pd.read_csv("cleaned_data_new.csv")

st.title("🔎 Prediksi Risiko Dropout Mahasiswa")
st.markdown("Masukkan data mahasiswa di sidebar untuk memprediksi kemungkinan dropout.")

# Sidebar Input
st.sidebar.header("Masukkan Fitur Mahasiswa")

selected_features = [
    "Age at enrollment",
    "Previous qualification (grade)",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)"
]

input_data = {}
for feature in selected_features:
    if df[feature].dtype in ["int64", "float64"]:
        input_data[feature] = st.sidebar.number_input(f"{feature}", value=float(df[feature].mean()))
    else:
        input_data[feature] = st.sidebar.selectbox(f"{feature}", sorted(df[feature].unique()))

input_df = pd.DataFrame([input_data])

if st.sidebar.button("🚀 Prediksi Dropout"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("❌ Mahasiswa diprediksi berisiko **DROPOUT**.")
    else:
        st.success("✅ Mahasiswa diprediksi **TIDAK dropout**.")
