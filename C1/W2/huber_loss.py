"""
Description: Implementing a Huber loss as a Loss class

Author: Skif Pankov (ス)
Created: 27/12/2020
"""

from tensorflow.keras.losses import Loss
from typing import Optional

import tensorflow as tf


class HuberLoss(Loss):

    THRESHOLD = 1.0

    def __init__(self,
                 threshold: Optional[float] = THRESHOLD):

        super().__init__()

        if threshold is None:
            threshold = self.THRESHOLD

        self.threshold = threshold

    def call(self, y_true, y_pred):

        error = y_true - y_pred
        is_small_error = tf.abs(error) <= self.threshold

        small_error_loss = tf.square(error) / 2.0
        big_error_loss = self.threshold * (tf.abs(error) - (0.5 * self.threshold))

        out = tf.where(is_small_error, small_error_loss, big_error_loss)

        return out
