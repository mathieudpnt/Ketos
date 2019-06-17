# ================================================================================ #
#   Authors: Fabio Frazao and Oliver Kirsebom                                      #
#   Contact: fsfrazao@dal.ca, oliver.kirsebom@dal.ca                               #
#   Organization: MERIDIAN (https://meridian.cs.dal.ca/)                           #
#   Team: Data Analytics                                                           #
#   Project: ketos                                                                 #
#   Project goal: The ketos library provides functionalities for handling          #
#   and processing acoustic data and applying deep neural networks to sound        #
#   detection and classification tasks.                                            #
#                                                                                  #
#   License: GNU GPLv3                                                             #
#                                                                                  #
#       This program is free software: you can redistribute it and/or modify       #
#       it under the terms of the GNU General Public License as published by       #
#       the Free Software Foundation, either version 3 of the License, or          #
#       (at your option) any later version.                                        #
#                                                                                  #
#       This program is distributed in the hope that it will be useful,            #
#       but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#       GNU General Public License for more details.                               # 
#                                                                                  #
#       You should have received a copy of the GNU General Public License          #
#       along with this program.  If not, see <https://www.gnu.org/licenses/>.     #
# ================================================================================ #

""" inception sub-module within the ketos.neural_networks module

    This module provides classes that implement Inception Neural Networks.

    Contents:
        ConvBatchNormRelu class:
        InceptionBlock class:
        Inception class:

"""

import tensorflow as tf

class ConvBatchNormRelu(tf.keras.Model):
    def __init__(self, ch, kernel_size=3, strides=1, padding='same'):
        super(ConvBatchNormRelu, self).__init__()

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(ch, kernel_size, strides=strides, padding=padding),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, x, training=None):
        x = self.model(x, training=training)

        return x


class InceptionBlock(tf.keras.Model):
    def __init__(self, ch, strides=1):
        super(InceptionBlock, self).__init__()

        self.ch = ch
        self.strides = strides

        self.conv1 = ConvBatchNormRelu(ch, strides=self.strides)
        self.conv2 = ConvBatchNormRelu(ch, kernel_size=3, strides=self.strides)
        self.conv3_1 = ConvBatchNormRelu(ch, kernel_size=3, strides=self.strides)
        self.conv3_2 = ConvBatchNormRelu(ch, kernel_size=3, strides=1)

        self.pool = tf.keras.layers.MaxPooling2D(3, strides=1, padding='same')
        self.pool_conv = ConvBatchNormRelu(ch, strides=self.strides)

    def call(self, x, training=None):
        x1 = self.conv1(x, training=training)
        x2 = self.conv2(x, training=training)
        x3_1 = self.conv3_1(x, training=training)
        x3_2 = self.conv3_2(x3_1, training=training)
        x4 = self.pool(x)
        x4 = self.pool_conv(x4, training=True)

        out = tf.concat([x1, x2, x3_2, x4], axis=3)

        return out

class Inception(tf.keras.Model):

    def __init__(self, num_layers, num_classes, init_channels=16, **kwargs):
        super(Inception, self).__init__(**kwargs)

        self.input_channels = init_channels
        self.output_channels = init_channels
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.init_channels = init_channels

        self.conv1 = ConvBatchNormRelu(self.init_channels)

        self.blocks = tf.keras.models.Sequential(name='dynamic-blocks')

        for block_id in range(self.num_layers):
            for layer_id in range(2):

                if layer_id == 0:
                    block = InceptionBlock(self.output_channels, strides=2)
                else:
                    block = InceptionBlock(self.output_channels, strides=1)

                self.blocks.add(block)

            self.output_channels *= 2

        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(self.num_classes)

    def call(self, x, training=None):
        out = self.conv1(x, training=training)
        out = self.blocks(out, training=training)
        out = self.avg_pool(out)
        out = self.dense(out)

        return out

        

        

