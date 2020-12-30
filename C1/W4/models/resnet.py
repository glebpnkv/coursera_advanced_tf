"""
Description:

Author: Skif Pankov (ス)
Created: 30/12/2020
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, MaxPool2D, GlobalAveragePooling2D


class IdentityBlock(Model):

    def __init__(self,
                 filters,
                 kernel_size,
                 **kwargs):

        super(IdentityBlock, self).__init__(**kwargs)

        self.conv1 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')
        self.conv2 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')

        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()

        self.act = Activation(activation='relu')
        self.add = Add()

    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.add([x, inputs])
        x = self.act(x)

        return x


class ResNet(Model):

    def __init__(self, num_classes: int, **kwargs):

        super(ResNet, self).__init__(**kwargs)

        # hidden / other layers
        self.conv = Conv2D(filters=64, kernel_size=7, padding='same')
        self.bn = BatchNormalization()
        self.act = Activation('relu')
        self.max_pool = MaxPool2D(pool_size=(3, 3))
        self.id1a = IdentityBlock(64, 3)
        self.id1b = IdentityBlock(64, 3)
        self.global_pool = GlobalAveragePooling2D()
        self.classifier = Dense(num_classes, activation='softmax')

    def call(self, inputs):

        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.id1a(x)
        x = self.id1b(x)

        x = self.global_pool(x)

        x = self.classifier(x)

        return x
