import numpy as np
from sklearn.datasets import make_classification
from tensorflow import keras


def get_LSTM(input_dim=29, units=64, output_size=3, allow_cudnn_kernel=True):
    model = keras.models.Sequential(name="lstm_model")
    if allow_cudnn_kernel:
        model.add(keras.layers.LSTM(units, input_shape=(input_dim, 1), return_sequences=False))
    else:
        model.add(keras.layers.RNN(keras.layers.LSTMCell(units), input_shape=(input_dim, 1)))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(output_size, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    print(model.summary())
    return model


def get_Dense(input_dim=29, output_size=3):
    model = keras.models.Sequential(name="dense_model")
    model.add(keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(output_size, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    print(model.summary())
    return model


def get_CNN(input_dim=29, output_size=3):
    model = keras.models.Sequential(name="cnn_model")
    model.add(keras.layers.Conv1D(32, 3, activation='relu', input_shape=(input_dim, 1)))
    model.add(keras.layers.MaxPooling1D(2))
    model.add(keras.layers.Conv1D(64, 3, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(output_size, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    print(model.summary())
    return model
