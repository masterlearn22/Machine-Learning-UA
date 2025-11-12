# In[ ]:
# ========= IMPORT LIBRARY YANG DIBUTUHKAN =========
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report, roc_auc_score
import time



# In[ ]:
# ========= PARAMETER UTAMA =========
IMG_SIZE = 200
BATCH_SIZE = 32
SEED = 123

# ========= LOAD DATASET =========
dataset = image_dataset_from_directory(
    directory='seg_train/seg_train',
    labels='inferred',
    label_mode='int',
    image_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',
    shuffle=True,
    batch_size=BATCH_SIZE,
    seed=SEED
)




# In[ ]:
# ========= SPLIT DATASET =========
train_size = int(0.8 * len(dataset))
train_ds = dataset.take(train_size)
test_ds = dataset.skip(train_size)

val_size = int(0.2 * len(train_ds))
val_ds = train_ds.take(val_size)
train_ds = train_ds.skip(val_size)

# In[ ]:
# ========= PREPROCESSING (ResNet50) =========
def process(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(process, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.map(process,   num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
test_ds  = test_ds.map(process,  num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

# In[ ]:
#========= DATA AUGMENTATION =========
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),        # Membalik gambar secara horizontal
    layers.RandomRotation(0.2),             # Rotasi gambar ±20%
    layers.RandomZoom((-0.2, 0.0)),         # Zoom OUT maksimal 20% (nilai negatif)
    layers.RandomShear(0.2),                # Geser (shear) gambar sebesar 20%
], name="data_augmentation")

# In[ ]:
# ========= MODEL RESNET50 =========
base_model = tf.keras.applications.ResNet50(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(6, activation='softmax')
])



# In[ ]:
# ========= KOMPILE MODEL =========
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)



# In[ ]:
# ========= CALLBACKS =========
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, verbose=1)
]
model.summary()

# In[ ]:
# ========= TRAINING =========
start_train = time.time()
history = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=1, callbacks=callbacks)
end_train = time.time()
train_time = end_train - start_train

# In[ ]:
# ========= EVALUASI =========
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n🕒 Training Time: {train_time:.2f} detik")
print(f"✅ Test Accuracy: {test_acc:.4f}")

# In[ ]:
# ========= TESTING AKHIR =========
fine_test_loss, fine_test_acc = model.evaluate(test_ds)
print(f"✅ Fine-tuned Test Accuracy: {fine_test_acc:.4f}")

# ========= METRIK LENGKAP =========
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df[['precision','recall','f1-score']])

try:
    y_true_bin = tf.keras.utils.to_categorical(y_true, num_classes=6)
    auc = roc_auc_score(y_true_bin, y_pred_probs, multi_class='ovr')
    print(f"ROC-AUC (macro): {auc:.4f}")
except:
    print("ROC-AUC tidak dapat dihitung")

# In[ ]:
# ========= VISUALISASI =========
sns.lineplot(x=history.epoch, y=history.history['loss'], label='Training Loss')
sns.lineplot(x=history.epoch, y=history.history['val_loss'], label='Validation Loss')
plt.show()

sns.lineplot(x=history.epoch, y=history.history['accuracy'], label='Training Accuracy')
sns.lineplot(x=history.epoch, y=history.history['val_accuracy'], label='Validation Accuracy')
plt.show()

