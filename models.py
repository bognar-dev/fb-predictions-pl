import numpy as np
from sklearn.datasets import make_classification
from tensorflow import keras


def get_LSTM(input_dim=29, units=64, output_size=3, allow_cudnn_kernel=True):
    if allow_cudnn_kernel:
        lstm_layer = keras.layers.LSTM(units, input_shape=(input_dim, 1), return_sequences=True)
    else:
        lstm_layer = keras.layers.RNN(
            keras.layers.LSTMCell(units), input_shape=(input_dim, 1)
        )
    model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.LSTM(32, return_sequences=False, activation='linear'),
            keras.layers.Dense(output_size),
        ]
    )
    return model
