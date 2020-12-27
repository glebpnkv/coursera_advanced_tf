"""
Description: Linear regression model implemented with a simple custom tf Layer

Author: Skif Pankov (ス)
Created: 27/12/2020
"""

from tensorflow.keras.models import Model
from C1.W3.simple_dense import SimpleDense


class LinearRegression(Model):

    def __init__(self,
                 units: int = 1):

        super(LinearRegression, self).__init__()
        self.simple_dense = SimpleDense(units)

    def call(self, inputs):
        x = self.simple_dense(inputs)

        return x
