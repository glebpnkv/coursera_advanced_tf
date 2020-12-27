"""
Description: Using a custom Lambda layer in a tf model

Author: Skif Pankov (ス)
Created: 27/12/2020
"""

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Lambda, Input

import tensorflow.keras.backend as K


def get_fashion_mnist_data():

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test


def threshold_relu(threshold: float = 0.0):

    def threshold_relu_func(x):
        return K.maximum(threshold, x)

    return threshold_relu_func


def fmnist_model(threshold: float = 0.0):

    input = Input(shape=(28, 28), name='input')
    threshold_relu_input = threshold_relu(threshold)

    x = Flatten(input_shape=(28, 28))(input)
    x = Dense(128)(x)
    x = Lambda(threshold_relu_input, name='lambda_threshold_relu')(x)
    x = Dense(10, activation='softmax')(x)

    return Model(inputs=input, outputs=x)


def train_model(threshold: float = 0.0):

    x_train, y_train, x_test, y_test = get_fashion_mnist_data()
    model = fmnist_model(threshold)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10)
    model.evaluate(x_test, y_test)


if __name__ == "__main__":

    THRESHOLD = -0.2
    train_model(THRESHOLD)
