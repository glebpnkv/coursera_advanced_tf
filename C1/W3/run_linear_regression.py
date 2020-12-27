"""
Description: Trains a linear regression example with the custom layer

Author: Skif Pankov (ス)
Created: 27/12/2020
"""

from typing import Tuple
from C1.W3.linear_regression import LinearRegression

import numpy as np


def data_generation_process(x: np.ndarray) -> np.ndarray:
    """
    Linear function defining the data generation process

    :param x: numpy array

    :return: numpy array of the same shape as x
    """

    out = 2.0 * x - 1.0

    return out


def generate_data(x_min=-1.0,
                  x_max=4.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates data from data generation process

    :param x_min: Minimum value of x
    :param x_max: Maximum value of x

    :return: a Tuple of 2 numpy arrays: (x, y)
    """

    x = np.arange(start=x_min,
                  stop=x_max + 1,
                  step=1.0,
                  dtype='float')

    y = data_generation_process(x)

    return x, y


def _main() -> None:
    """
    Trains the model and calculates its forecast for a new value.

    This model implementation will become unstable (reach nan values) if "too much" data is used for training

    :return: None
    """

    x_train, y_train = generate_data(-8.0, 8.0)

    x_test = np.array([10.0])

    linear_regresion = LinearRegression()

    linear_regresion.compile(
        optimizer='sgd',
        loss='mse'
    )

    linear_regresion.fit(x_train, y_train, epochs=500, verbose=0)

    print(f"Model's Forecast: {linear_regresion.predict(x_test).flatten()}")
    print(f"True Value: {data_generation_process(x_test)}")
    print(f"Current Parameters: {linear_regresion.simple_dense.variables}")


if __name__ == "__main__":

    _main()
