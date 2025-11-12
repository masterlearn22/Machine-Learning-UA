import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# ========= LOAD DATASET TESTING =========
# Pastikan path dan ukuran gambar sama persis seperti saat training!
IMG_SIZE = 200
BATCH_SIZE = 32

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='Week10/seg_train2/seg_train',  # ubah sesuai folder test kamu
    labels='inferred',
    label_mode='int',
    image_size=(IMG_SIZE, IMG_SIZE),
    shuffle=False,
    batch_size=BATCH_SIZE
)

class_names = test_ds.class_names
print("✅ Dataset testing dimuat, kelas:", class_names)

# ========= LOAD MODEL =========
model = tf.keras.models.load_model("Week10/resnet50_noaugment.h5")
print("✅ Model berhasil dimuat!")

# ========= EVALUASI CEPAT (Loss & Accuracy) =========
# loss, acc = model.evaluate(test_ds)
# print(f"\n📊 Evaluasi dasar: Loss={loss:.4f} | Accuracy={acc:.4f}")

# ========= PREDIKSI DAN METRIK LENGKAP =========
# Ambil label asli
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# Prediksi model
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# ========= CLASSIFICATION REPORT =========
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("\n===== 📊 CLASSIFICATION REPORT =====")
print(report_df[['precision', 'recall', 'f1-score']])

# ========= CONFUSION MATRIX =========
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Model Evaluation')
plt.show()

# ========= ROC-AUC SCORE =========
try:
    y_true_bin = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
    auc = roc_auc_score(y_true_bin, y_pred_probs, multi_class='ovr')
    print(f"\nROC-AUC Score (macro): {auc:.4f}")
except Exception as e:
    print("ROC-AUC tidak dapat dihitung:", e)
