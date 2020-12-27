"""
Description: Using a custom Huber loss to train a model

Author: Skif Pankov (ス)
Created: 27/12/2020
"""

from C1.W2.huber_loss import HuberLoss
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import numpy as np


def train_model():
    # inputs
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)

    # labels
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    model = Sequential([Dense(units=1, input_shape=[1])])

    # model.compile(optimizer='sgd', loss=HuberLoss(threshold=1.02))
    model.compile(optimizer='sgd', loss=HuberLoss(threshold=1.02))
    model.fit(xs, ys, epochs=500, verbose=0)

    out = (model.predict([10.0]))

    return out


if __name__ == "__main__":

    out = train_model()

    print(out)
