"""
Description: Creating a simple custom tf Layer

Author: Skif Pankov (ス)
Created: 27/12/2020
"""

from typing import Optional
from tensorflow.python.keras.engine.base_layer import Layer

import tensorflow as tf


class SimpleDense(Layer):
    """
    A copy of SimpleDense layer - a barebone replica of Dense layer
    """

    def __init__(self,
                 units: int = 32,
                 activation: Optional[str] = None):
        """
        Initialisation

        :param units: Number of units
        :param activation: Optional keras-recognised name of the activation function used by the layer
        """

        super(SimpleDense, self).__init__()

        self.units = units
        self.activation = tf.keras.activations.get(activation)
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

        super().build(input_shape)

    def call(self, inputs):
        """
        Applies layer on inputs

        :param inputs: Tensor of inputs

        :return: Tensor of the same type as self.w and self.b
        """

        out = self.activation(tf.matmul(inputs, self.w) + self.b)

        return out
