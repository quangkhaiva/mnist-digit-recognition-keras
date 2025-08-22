import numpy as np
import pandas as pd
from tensorflow import keras

def load_mnist_keras():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train[..., None].astype("float32") / 255.0
    x_test = x_test[..., None].astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)

def load_mnist_kaggle_csv(train_csv_path, test_csv_path=None):
    """Dùng nếu tải train.csv/test.csv từ Kaggle Digit Recognizer"""
    df = pd.read_csv(train_csv_path)
    y = df["label"].values.astype("int64")
    x = df.drop(columns=["label"]).values.reshape(-1,28,28,1).astype("float32")/255.0
    return (x, y)