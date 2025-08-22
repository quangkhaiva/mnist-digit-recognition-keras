import os, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from data_utils import load_mnist_keras, load_mnist_kaggle_csv

# ====== cấu hình nguồn dữ liệu ======
USE_KAGGLE_CSV = False  # Bật True nếu dùng train.csv Kaggle

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

# ====== nạp dữ liệu ======
if not USE_KAGGLE_CSV:
    (x_train, y_train), (x_test, y_test) = load_mnist_keras()
else:
    x_all, y_all = load_mnist_kaggle_csv("data/train.csv", "data/test.csv")
    n = len(x_all)
    n_test = int(0.1*n)
    x_test, y_test = x_all[:n_test], y_all[:n_test]
    x_train, y_train = x_all[n_test:], y_all[n_test:]

# tách validation từ train
val_ratio = 0.1
n_val = int(len(x_train)*val_ratio)
x_val, y_val = x_train[:n_val], y_train[:n_val]
x_trn, y_trn = x_train[n_val:], y_train[n_val:]

# ====== online augmentation ======
augment = keras.Sequential([
    layers.RandomRotation(0.25),
    layers.RandomTranslation(0.2, 0.2),
    layers.RandomZoom(0.25, 0.25),
    layers.RandomContrast(0.3),
    layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x), stddev=0.04)),
])

BATCH = 256
AUTO = tf.data.AUTOTUNE

def make_ds(x, y, training=True):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(8192)
        ds = ds.map(lambda a,b: (augment(a, training=True), b), num_parallel_calls=AUTO)
    return ds.batch(BATCH).prefetch(AUTO)

train_ds = make_ds(x_trn, y_trn, True)
val_ds   = make_ds(x_val, y_val, False)
test_ds  = make_ds(x_test, y_test, False)

# ====== model ======
def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

inputs = keras.Input((28,28,1))
x = conv_block(inputs, 32)
x = conv_block(x, 32)
x = layers.MaxPool2D()(x); x = layers.Dropout(0.25)(x)
x = conv_block(x, 64)
x = conv_block(x, 64)
x = layers.MaxPool2D()(x); x = layers.Dropout(0.25)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

os.makedirs("models", exist_ok=True)
cbs = [
    keras.callbacks.ModelCheckpoint("models/mnist_tf_best.keras",
                                    monitor="val_accuracy", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
]

hist = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=cbs)

# ====== đánh giá ======
loss, acc = model.evaluate(test_ds, verbose=0)
print(f"Test accuracy: {acc:.4f}")

y_prob = model.predict(test_ds, verbose=0)
y_pred = y_prob.argmax(axis=1)
print(classification_report(y_test, y_pred, digits=4))
print(confusion_matrix(y_test, y_pred))