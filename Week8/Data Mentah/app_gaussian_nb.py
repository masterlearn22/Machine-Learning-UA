
import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Gaussian Naive Bayes", page_icon="🧮", layout="centered")
st.title("🧮 Gaussian Naive Bayes – Kategorikal (Encoded)")

BUNDLE_PATH = "gaussian_nb_model.pkl"
bundle = joblib.load(BUNDLE_PATH)

gnb = bundle["model"]
X_encoder = bundle["X_encoder"]
y_encoder = bundle["y_encoder"]
feature_names = bundle["feature_names"]
categories = bundle["categories"]
data_path = bundle.get("data_path", "")

st.caption(f"Model dimuat dari **{BUNDLE_PATH}** · Dataset training: **{data_path}**")

st.header("Input x_test")
user_vals = {}
cols = st.columns(2)
for i, feat in enumerate(feature_names):
    with cols[i % 2]:
        opts = categories[feat]
        user_vals[feat] = st.selectbox(feat, opts, index=0)

if st.button("🔍 Klasifikasikan (GaussianNB)"):
    X_df = pd.DataFrame([[user_vals[f] for f in feature_names]], columns=feature_names)
    X_enc = X_encoder.transform(X_df)
    y_pred_enc = gnb.predict(X_enc)[0]
    y_pred = y_encoder.inverse_transform([y_pred_enc])[0]
    proba = gnb.predict_proba(X_enc)[0]
    proba_dict = {cls: float(p) for cls, p in zip(y_encoder.classes_, proba)}

    st.subheader("Hasil Prediksi")
    st.json({"Predicted Class": y_pred, "Posterior Probabilities": proba_dict})
    st.success(f"Prediksi kelas: **{y_pred}**")

st.divider()
st.caption("Fitur dikodekan dengan OrdinalEncoder; target dengan LabelEncoder. Model: scikit-learn GaussianNB.")
