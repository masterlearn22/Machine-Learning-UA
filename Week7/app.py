import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn import tree

# ----------------------------
# Konfigurasi awal Streamlit
# ----------------------------
st.set_page_config(page_title="Decision Tree Classifier", layout="wide")
st.title("🌳 Decision Tree Classifier - Evaluasi & Visualisasi")

# ----------------------------
# Load model
# ----------------------------
with open("model.pkl", "rb") as f:
    clf = pickle.load(f)

# ----------------------------
# Upload dataset test
# ----------------------------
uploaded = st.file_uploader("Upload dataset_test.csv", type=["csv"])
if uploaded is not None:
    df_test = pd.read_csv(uploaded)

    st.subheader("📄 Cuplikan Data Uji")
    st.dataframe(df_test.head())

    # Kolom target (ubah sesuai datasetmu)
    target_col = "penyakit"
    if target_col not in df_test.columns:
        st.error(f"Kolom target '{target_col}' tidak ada di dataset!")
        st.stop()

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # ----------------------------
    # Prediksi & Evaluasi
    # ----------------------------
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.metric("Accuracy (Test)", f"{acc:.3f}")

    st.write("**Classification Report**")
    st.text(classification_report(y_test, y_pred))

    st.write("**Confusion Matrix**")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax=ax)
    st.pyplot(fig)

    # ----------------------------
    # Visualisasi Pohon
    # ----------------------------
    st.subheader("🌳 Visualisasi Pohon Keputusan")

    max_depth_vis = st.slider("Pilih kedalaman visualisasi pohon", 1, 10, 3)

    fig, ax = plt.subplots(figsize=(20, 10))
    tree.plot_tree(
        clf.named_steps["model"],
        filled=True,
        feature_names=X_test.columns,
        class_names=[str(c) for c in sorted(y_test.unique())],
        fontsize=10,
        max_depth=max_depth_vis  # kendali kedalaman visualisasi
    )
    st.pyplot(fig)
