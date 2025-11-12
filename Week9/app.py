# =============================
# 📦 Import Library
# =============================
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score
)

# =============================
# ⚙️ CLASS: Machine Learning Manager
# =============================
class SVMWineModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.best_kernel = "rbf"

    def load_data(self, path):
        df = pd.read_csv(path)
        if "Id" in df.columns:
            df = df.drop(columns=["Id"])
        return df

    def preprocess(self, df):
        # Pisahkan fitur dan target (pakai logika yang sama seperti di kelola.ipynb)
        df["quality_label"] = df["quality"].apply(lambda x: 1 if x >= 6 else 0)
        X = df.drop(["quality", "quality_label"], axis=1)
        y = df["quality_label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Gunakan scaler yang sama untuk semua model
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X, y, X_train_scaled, X_test_scaled, y_train, y_test

    def train_all_kernels(self, X_train_scaled, X_test_scaled, y_train, y_test):
        kernels = ["linear", "poly", "rbf"]
        results = []

        for kernel in kernels:
            svm = SVC(kernel=kernel, random_state=42)  # ❌ hapus class_weight
            svm.fit(X_train_scaled, y_train)
            y_pred = svm.predict(X_test_scaled)

            results.append({
                "Kernel": kernel,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred),
                "Confusion Matrix": confusion_matrix(y_test, y_pred)
            })
        return pd.DataFrame(results)

    def train_best_model(self, X, y):
        # Gunakan kernel terbaik berdasarkan hasil cross-eval
        self.model = SVC(kernel=self.best_kernel, random_state=42)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self.model

    def predict(self, input_df):
        input_scaled = self.scaler.transform(input_df)
        return self.model.predict(input_scaled)

# =============================
# 📊 FUNCTION: Visualizer
# =============================
class Visualizer:
    @staticmethod
    def plot_histograms(df):
        st.subheader("📊 Distribusi Tiap Fitur")
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))
        axes = axes.flatten()

        for i, col in enumerate(num_cols[:12]):
            sns.histplot(df[col], ax=axes[i], kde=True, color='skyblue', bins=20)
            axes[i].set_title(col)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("""
        **Penjelasan:**
        - Grafik ini menunjukkan distribusi tiap fitur numerik.
        - Berguna untuk melihat outlier dan sebaran data.
        """)

    @staticmethod
    def plot_correlation(df):
        st.subheader("📈 Korelasi Antar Fitur")
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Korelasi Antar Fitur", fontsize=14, fontweight='bold')
        st.pyplot(fig)

    @staticmethod
    def plot_kernel_comparison(results_df):
        st.subheader("📊 Perbandingan Kinerja Tiap Kernel")
        fig, ax = plt.subplots(figsize=(8, 5))
        results_df.set_index("Kernel")[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(kind="bar", ax=ax)
        plt.ylim(0.6, 0.9)
        plt.title("Perbandingan Kernel SVM")
        st.pyplot(fig)

    @staticmethod
    def plot_confusion_matrix(cm, kernel):
        st.subheader("📉 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix ({kernel} Kernel)")
        st.pyplot(fig)


# =============================
# 🧠 STREAMLIT APP (Main Logic)
# =============================
st.set_page_config(page_title="SVM Wine Quality", page_icon="🍷", layout="wide")
st.title("🍷 Aplikasi Analisis Kualitas Wine (SVM)")
menu = st.sidebar.radio("Pilih Mode:", ["📂 Analisis Dataset", "🧮 Prediksi Manual", "📈 Prediksi File CSV"])

# Inisialisasi model
ml = SVMWineModel()

# =============================
# MODE 1: Analisis Dataset
# =============================
if menu == "📂 Analisis Dataset":
    uploaded_file = st.file_uploader("📂 Unggah Dataset (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("🔍 Pratinjau Dataset")
        st.dataframe(df.head())
        
        if 'quality' not in df.columns:
            st.error("Kolom 'quality' tidak ditemukan!")
        else:
            X, y, X_train_scaled, X_test_scaled, y_train, y_test = ml.preprocess(df)
            results_df = ml.train_all_kernels(X_train_scaled, X_test_scaled, y_train, y_test)
            results_df_display = results_df.drop(columns=["Confusion Matrix"])
            st.dataframe(results_df_display.round(4))
            Visualizer.plot_kernel_comparison(results_df_display)

            selected_kernel = st.selectbox("Pilih Kernel:", results_df["Kernel"])
            selected_cm = results_df.loc[results_df["Kernel"] == selected_kernel, "Confusion Matrix"].values[0]
            Visualizer.plot_confusion_matrix(selected_cm, selected_kernel)

            st.success(f"Kernel terbaik: {results_df.loc[results_df['Accuracy'].idxmax(), 'Kernel']}")

        Visualizer.plot_histograms(df)
        Visualizer.plot_correlation(df)

        

# =============================
# MODE 2: Prediksi Manual
# =============================
elif menu == "🧮 Prediksi Manual":
    st.subheader("🧮 Input Manual Data Wine")
    with st.form("input_form"):
        features = {
            'fixed acidity': st.slider("Fixed Acidity", 4.0, 16.0, 7.0),
            'volatile acidity': st.slider("Volatile Acidity", 0.1, 1.5, 0.5),
            'citric acid': st.slider("Citric Acid", 0.0, 1.0, 0.3),
            'residual sugar': st.slider("Residual Sugar", 0.5, 15.0, 2.5),
            'chlorides': st.slider("Chlorides", 0.01, 0.6, 0.08),
            'free sulfur dioxide': st.slider("Free Sulfur Dioxide", 1.0, 70.0, 15.0),
            'total sulfur dioxide': st.slider("Total Sulfur Dioxide", 6.0, 290.0, 46.0),
            'density': st.slider("Density", 0.9900, 1.0050, 0.9960),
            'pH': st.slider("pH", 2.8, 4.0, 3.3),
            'sulphates': st.slider("Sulphates", 0.3, 2.0, 0.7),
            'alcohol': st.slider("Alcohol", 8.0, 15.0, 10.0)
        }
        submitted = st.form_submit_button("🔍 Prediksi")

    if submitted:
        df_default = ml.load_data("WineQT.csv")
        X, y, *_ = ml.preprocess(df_default)
        ml.train_best_model(X, y)
        input_df = pd.DataFrame([features])
        prediction = ml.predict(input_df)[0]
        if prediction == 1:
            st.markdown(
                """
                <div style="
                    background-color: #4CAF50;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    color: white;
                    font-size: 20px;
                    font-weight: bold;">
                    Wine berkualitas BAIK
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="
                    background-color: #f44336;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    color: white;
                    font-size: 20px;
                    font-weight: bold;">
                    Wine berkualitas BURUK
                </div>
                """,
                unsafe_allow_html=True
            )


# =============================
# MODE 3: Prediksi File CSV
# =============================
elif menu == "📈 Prediksi File CSV":
    st.subheader("📂 Upload File untuk Prediksi (tanpa kolom quality)")
    uploaded_pred = st.file_uploader("Unggah file predict.csv", type=["csv"])
    if uploaded_pred:
        df_pred = pd.read_csv(uploaded_pred)
        st.dataframe(df_pred.head())

        df_train = ml.load_data("WineQT.csv")
        X, y, *_ = ml.preprocess(df_train)
        ml.train_best_model(X, y)

        y_pred = ml.predict(df_pred)
        df_pred["Predicted_Quality"] = ["Baik (≥6)" if val == 1 else "Buruk (<6)" for val in y_pred]

        st.success("✅ Prediksi berhasil dilakukan!")
        st.dataframe(df_pred)
        csv = df_pred.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Unduh Hasil Prediksi", csv, "predicted_results.csv", "text/csv")
