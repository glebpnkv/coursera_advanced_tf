"""
Description: Creating a custom tf Layer

Author: Skif Pankov (ス)
Created: 27/12/2020
"""


from tensorflow.python.keras.engine.base_layer import Layer

import tensorflow as tf


class SimpleDense(Layer):
    """
    A copy of SimpleDense layer - a barebone replica of Dense layer
    """

    def __init__(self,
                 units: int = 32):

        super(SimpleDense, self).__init__()

        self.units = units
        self.w = None
        self.b = None

    def build(self, input_shape) -> None:
        """
        Initialises layer's trainable parameters (weights and biases)

        :param input_shape: Tuple of shapes of input Tensors

        :return: None
        """

        # defining initialisers for weights and biases
        w_init = tf.random_normal_initializer()
        b_init = tf.zeros_initializer()

        # initialising the layer's weight
        self.w = tf.Variable(
            name='kernel',
            initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'),
            trainable=True
        )

        self.b = tf.Variable(
            name='bias',
            initial_value=b_init(shape=self.units, dtype='float32'),
            trainable=True
        )

    def call(self, inputs):
        """
        Applies layer on inputs

        :param inputs: Tensor of inputs

        :return: Tensor of the same type as self.w and self.b
        """

        out = tf.matmul(inputs, self.w) + self.b

        return out


def test_custom_layer():

    layer = SimpleDense(units=1)

    x = tf.ones((1, 1))

    y = layer(x)

    print(layer.variables)
    print(f'y: {y}')


if __name__ == "__main__":

    test_custom_layer()
