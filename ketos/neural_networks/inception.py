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
    """ Convolutional layer with batch normalization and relu activation.

        Used in Inception  Blocks

        Args: 
            n_filters: int
                Number of filters in the convolutional layer
            filter_shape: int
                The filter (i.e.: kernel) shape. 
            strides: int
                Strides to be used for the convolution operation
            padding:str
                Type of padding: 'same' or 'valid'
        
    """


    def __init__(self, n_filters, filter_shape=3, strides=1, padding='same'):
        super(ConvBatchNormRelu, self).__init__()

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(n_filters, filter_shape, strides=strides, padding=padding),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, x, training=None):
        x = self.model(x, training=training)

        return x


class InceptionBlock(tf.keras.Model):
    """ Inception Block for the Inception Architecture

        Args:
            n_filters:int
               The number of filters (i.e.: channels) to be used in each convolutional layer of the block
            strides: int
                Strides used in the first 3 and and 5th convolutional layers of the block

    """

    def __init__(self, n_filters, strides=1):
        super(InceptionBlock, self).__init__()

        self.n_filters = n_filters
        self.strides = strides

        self.conv1 = ConvBatchNormRelu(self.n_filters, strides=self.strides)
        self.conv2 = ConvBatchNormRelu(self.n_filters, filter_shape=3, strides=self.strides)
        self.conv3_1 = ConvBatchNormRelu(self.n_filters, filter_shape=3, strides=self.strides)
        self.conv3_2 = ConvBatchNormRelu(self.n_filters, filter_shape=3, strides=1)

        self.pool = tf.keras.layers.MaxPooling2D(3, strides=1, padding='same')
        self.pool_conv = ConvBatchNormRelu(self.n_filters, strides=self.strides)

    def call(self, x, training=None):
        x1 = self.conv1(x, training=training)
        x2 = self.conv2(x, training=training)
        x3_1 = self.conv3_1(x, training=training)
        x3_2 = self.conv3_2(x3_1, training=training)
        x4 = self.pool(x)
        x4 = self.pool_conv(x4, training=True)

        out = tf.concat([x1, x2, x3_2, x4], axis=3)

        return out



        
