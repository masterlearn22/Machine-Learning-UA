import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ========= CONFIGURASI DASAR =========
st.set_page_config(page_title="ResNet50 Image Classifier", page_icon="🧠", layout="centered")

# ========= LOAD MODEL =========
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("resnet50_noaugment.h5")
    return model

model = load_model()
st.success("✅ Model ResNet50 berhasil dimuat!")

# ========= PARAMETER =========
IMG_SIZE = 200  # harus sama dengan ukuran waktu training
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']  # ubah sesuai kelas datasetmu

# ========= FUNGSI PREPROCESS =========
def preprocess_image(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))        # ubah ukuran sesuai training
    img = np.array(img) / 255.0                     # normalisasi 0–1
    img = np.expand_dims(img, axis=0)               # tambah dimensi batch
    return img

# ========= UI STREAMLIT =========
st.title("🌍 Image Classification dengan ResNet50")
st.write("Upload gambar dari kategori: **buildings, forest, glacier, mountain, sea, street**")

uploaded_file = st.file_uploader("📸 Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    if st.button("🔍 Prediksi Kelas"):
        with st.spinner("Model sedang memproses gambar..."):
            processed_img = preprocess_image(image)
            preds = model.predict(processed_img, verbose=0)
            pred_class = CLASS_NAMES[np.argmax(preds)]
            confidence = np.max(preds) * 100

        st.success(f"✅ Prediksi: **{pred_class.upper()}** ({confidence:.2f}%)")

        # tampilkan probabilitas semua kelas
        st.subheader("📊 Probabilitas Tiap Kelas")
        for i, cls in enumerate(CLASS_NAMES):
            st.write(f"- {cls.capitalize()}: {preds[0][i]*100:.2f}%")

