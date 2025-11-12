import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder

# ======================================
# 🔧 CLASS TAMBAHAN: SafeOrdinalEncoder
# ======================================
class SafeOrdinalEncoder(OrdinalEncoder):
    """OrdinalEncoder yang otomatis ubah -1 jadi max+1 agar tidak error di CategoricalNB"""
    def transform(self, X):
        X_enc = super().transform(X)
        for i in range(X_enc.shape[1]):
            mask = X_enc[:, i] < 0
            if np.any(mask):
                max_val = np.nanmax(X_enc[:, i][~mask]) if np.any(~mask) else 0
                X_enc[:, i][mask] = max_val + 1
        return X_enc

# ======================================
# 1️⃣ Konfigurasi Halaman Streamlit
# ======================================
st.set_page_config(page_title="Naive Bayes Predictor", page_icon="🧮", layout="centered")
st.title("🧮 Naive Bayes Exam Score Predictor")
st.markdown("""
Aplikasi ini menggunakan **model Naive Bayes** yang sudah dilatih.
Silakan pilih kategori pada setiap fitur, kemudian klik **Run Prediction** untuk melihat hasil prediksi.
""")

# ======================================
# 2️⃣ Load Model & Dataset
# ======================================
@st.cache_resource
def load_model():
    model_path = "naive_bayes_balanced_model.pkl"
    if not os.path.exists(model_path):
        st.stop()
        raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")

    try:
        model = joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    return model

@st.cache_data
def load_dataset():
    return pd.read_excel("dataset_final.xlsx")

model = load_model()
df = load_dataset()

target_col = "Exam_Score"
feature_cols = [c for c in df.columns if c != target_col]

# ======================================
# 3️⃣ Input Form Dinamis
# ======================================
st.subheader("Masukkan Nilai Kategori Setiap Fitur")

user_input = {}
col1, col2 = st.columns(2)
for i, col in enumerate(feature_cols):
    unique_values = sorted(df[col].dropna().unique().tolist())
    with (col1 if i % 2 == 0 else col2):
        user_input[col] = st.selectbox(f"{col}", options=unique_values)

# ======================================
# 4️⃣ Prediksi
# ======================================
if st.button("Run Prediction 🚀"):
    input_df = pd.DataFrame([user_input])
    input_df = input_df[feature_cols]

    try:
        y_pred = model.predict(input_df)[0]
        if y_pred == 1:
            y_pred = "High (1)"
        elif y_pred == 0:
            y_pred = "Medium (0)"
        
        st.success(f"🎯 **Prediksi Exam_Score:** {y_pred}")

        # tampilkan probabilitas jika tersedia
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            prob_df = pd.DataFrame({
                "Kelas": model.classes_,
                "Probabilitas (%)": [round(p * 100, 2) for p in probs]
            })
            st.markdown("#### 🔍 Confidence Level:")
            st.dataframe(prob_df, use_container_width=True)

    except Exception as e:
        st.error("❌ Terjadi kesalahan saat memproses prediksi:")
        st.code(str(e))

# ======================================
# 5️⃣ Tentang Aplikasi
# ======================================
with st.expander("ℹ️ Tentang Aplikasi"):
    st.write("""
    - **Model:** Naive Bayes (Pipeline dengan SafeOrdinalEncoder)
    - **File model:** `naive_bayes_balanced_model.pkl`
    - **Dataset:** `dataset_final.xlsx`
    - **Output:** Prediksi nilai *Exam_Score* berdasarkan kategori input
    """)
