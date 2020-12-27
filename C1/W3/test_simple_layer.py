"""
Description: Testing a simple custom tf Layer

Author: Skif Pankov (ス)
Created: 27/12/2020
"""

import tensorflow as tf

from C1.W3.simple_dense import SimpleDense


def test_simple_layer():

    layer = SimpleDense(units=1)

    x = tf.ones((1, 1))

    y = layer(x)

    print(layer.variables)
    print(f'y: {y}')


if __name__ == "__main__":

    test_simple_layer()
