import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Naive Bayes (Categorical)", page_icon="🧮", layout="centered")

st.title("🧮 Naive Bayes Kategorikal – Demo")
st.markdown("Dataset di-load dari file: **hasil_dataset_kategorisasi_final_v2.xlsx**")

# Load data
df = pd.read_excel("hasil_dataset_kategorisasi_final_v2.xlsx")
for c in df.columns:
    df[c] = df[c].astype(str)

target_col = "Exam_Score"
features = [c for c in df.columns if c != target_col]

# Priors (Laplace)
class_counts = df[target_col].value_counts().sort_index()
classes = class_counts.index.tolist()
n = len(df)
priors = (class_counts + 1) / (n + len(classes))

# Conditional probabilities with Laplace
value_spaces = {feat: df[feat].unique().tolist() for feat in features}
cond_probs = {}
for feat in features:
    vals = value_spaces[feat]
    ct = pd.crosstab(df[feat], df[target_col]).reindex(index=vals, columns=classes, fill_value=0)
    smoothed = (ct + 1).div(class_counts + len(vals), axis=1)
    cond_probs[feat] = smoothed

# UI to pick feature values
st.header("Input x_test")
user_vals = {}
cols = st.columns(2)
for i, feat in enumerate(features):
    with cols[i % 2]:
        options = value_spaces[feat]
        user_vals[feat] = st.selectbox(feat, options, index=0)

def posterior_for_x(x_dict):
    log_post = {}
    for y in classes:
        lp = float(np.log(priors[y]))
        for feat, v in x_dict.items():
            table = cond_probs[feat]
            if v in table.index:
                p = float(table.loc[v, y])
            else:
                p = float(1.0 / (class_counts[y] + len(value_spaces[feat])))
            lp += float(np.log(p))
        log_post[y] = lp
    maxlog = max(log_post.values())
    s = sum(np.exp(v - maxlog) for v in log_post.values())
    post = {y: float(np.exp(lp - maxlog) / s) for y, lp in log_post.items()}
    return log_post, post

if st.button("🔍 Klasifikasikan"):
    log_post, post = posterior_for_x(user_vals)
    y_pred = max(post, key=post.get)
    st.subheader("Hasil Prediksi")
    st.write({ "posterior": post, "predicted_class": y_pred })
    st.success(f"Prediksi kelas: **{y_pred}**")

st.divider()
st.caption("Model menggunakan Naive Bayes kategorikal dengan Laplace smoothing.")
