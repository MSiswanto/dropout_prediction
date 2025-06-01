import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Visualisasi Data", layout="wide")

# Load data dan model
model = joblib.load("dropout_model.pkl")
df = pd.read_csv("cleaned_data_new.csv")

st.title("📊 Visualisasi Fitur dan Data Mahasiswa")

# Fitur penting
st.subheader("🔥 Top 10 Fitur Terpenting Menurut Model")
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

importances = pd.Series(model.feature_importances_, index=selected_features).sort_values(ascending=False)
top_features = importances.head(10)

fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette='viridis', ax=ax1)
ax1.set_title("Top 10 Fitur yang Mempengaruhi Dropout")
ax1.set_xlabel("Importance Score")
st.pyplot(fig1)

# Visualisasi lainnya
st.subheader("📈 Visualisasi Tambahan")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Distribusi Status Mahasiswa")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x='Target', palette='Set2', ax=ax2)
    ax2.set_xlabel("Status")
    ax2.set_ylabel("Jumlah")
    st.pyplot(fig2)

with col2:
    st.markdown("#### Boxplot Admission Grade by Status")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, x='Target', y='Admission grade', palette='Pastel1', ax=ax3)
    ax3.set_xlabel("Status")
    ax3.set_ylabel("Admission Grade")
    st.pyplot(fig3)
