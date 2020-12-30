"""
Description:

Author: Skif Pankov (ス)
Created: 30/12/2020
"""

from C1.W4.models import ResNet

import tensorflow as tf
import tensorflow_datasets as tfds

BATCH_SIZE = 128
NUM_EPOCHS = 1


def preprocess(features):

    out = tf.cast(features['image'], tf.float32) / 255.0, features['label']
    return out


def _main(batch_size=BATCH_SIZE,
          num_epochs=NUM_EPOCHS):

    resnet = ResNet(10)
    resnet.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    dataset = tfds.load('mnist', split=tfds.Split.TRAIN)
    dataset = dataset.map(preprocess).batch(batch_size)

    resnet.fit(dataset, epochs=num_epochs)


if __name__ == "__main__":
    _main()
