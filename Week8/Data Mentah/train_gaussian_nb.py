
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

DATA_PATH = "hasil_dataset_kategorisasi_final_v2.xlsx"
TARGET_COL = "Exam_Score"
MODEL_PATH = "gaussian_nb_model.pkl"

# 1) Load
df = pd.read_excel(DATA_PATH)
for c in df.columns:
    df[c] = df[c].astype(str)

# 2) Split features/target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

cat_features = X.columns.tolist()

# 3) Encoders
X_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
y_encoder = LabelEncoder()

y_enc = y_encoder.fit_transform(y)
X_enc = X_encoder.fit_transform(X)

# 4) Train GaussianNB
gnb = GaussianNB()
gnb.fit(X_enc, y_enc)

# 5) Simple train accuracy for sanity check
y_pred = gnb.predict(X_enc)
acc = accuracy_score(y_enc, y_pred)
print(f"Trained on {X.shape[0]} rows, acc(train)={acc:.3f}")
print("Classes:", list(y_encoder.classes_))

# 6) Persist bundle
bundle = {
    "model": gnb,
    "X_encoder": X_encoder,
    "y_encoder": y_encoder,
    "feature_names": cat_features,
    "target_col": TARGET_COL,
    "categories": {feat: X_encoder.categories_[i].tolist() for i, feat in enumerate(cat_features)},
    "data_path": DATA_PATH,
}

joblib.dump(bundle, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
