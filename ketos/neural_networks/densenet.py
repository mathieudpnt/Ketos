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

""" densenet sub-module within the ketos.neural_networks module

    This module provides classes to implement Residual Networks (ResNets).

    Contents:
        

"""

import os
import tensorflow as tf
import numpy as np
from .dev_utils.nn_interface import RecipeCompat, NNInterface
import json


class ConvBlock(tf.keras.Model):
    """ Convolutional Blocks used in the Dense Blocks.

        Args:
            growth_rate:int
                The growth rate for the number of filters (i.e.: channels) between convolutional layers
    
    """
    def __init__(self, growth_rate):
        super(ConvBlock, self).__init__()

        self.growth_rate = growth_rate

        self.batch_norm1 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.relu1 = tf.keras.layers.Activation('relu')
        self.conv1 = tf.keras.layers.Conv2D(4 * self.growth_rate, kernel_size=1, strides=1, use_bias=False, padding="same")

        self.batch_norm2 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.relu2 = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv2D(self.growth_rate, kernel_size=3, strides=1, use_bias=False, padding="same")

    def call(self, inputs, training=False):
        outputs = self.batch_norm1(inputs, training=training)
        outputs = self.relu1(outputs)
        outputs = self.conv1(outputs)

        outputs = self.batch_norm2(outputs, training=training)
        outputs = self.relu2(outputs)
        outputs = self.conv2(outputs)

        return outputs




class DenseBlock(tf.keras.Model):
    """ Dense block for DenseNet architectures

        Args:
            growth_rate: int
                The growth rate between blocks
            n_blocks:
                The number of convolutional blocks within the dense block
    
    """
    def __init__(self, growth_rate, n_blocks):
        super(DenseBlock,self).__init__()

        self.n_blocks = n_blocks
        self.blocks = tf.keras.Sequential()
        for i_block in range(self.n_blocks):
            self.blocks.add(ConvBlock(growth_rate=growth_rate))
        

    def call(self, inputs, training=False):
        outputs = self.blocks(inputs, training=training)
        outputs = tf.keras.layers.concatenate([inputs, outputs])

        return outputs


class TransitionBlock(tf.keras.Model):
    """ Transition Blocks for the DenseNet architecture

        Args:
            n_filters:int
                Number of filters (i,e,: channels)
            compression_factor: float
                The compression factor used within the transition block
                (i.e.: the reduction of filters/channels from the previous dense block to the next)
            dropout_rate:float
                Dropout rate for the convolutional layer (between 0 and 1, use 0 for no dropout)

    """
    def __init__(self, n_channels, compression_factor, dropout_rate=0.2):
        super(TransitionBlock, self).__init__()
        
        self.n_channels = n_channels
        self.compression_factor = compression_factor
        self.dropout_rate = dropout_rate

        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.conv = tf.keras.layers.Conv2D(int(self.n_channels * self.compression_factor), kernel_size=1, strides=1, padding="same")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.relu = tf.keras.layers.Activation('relu')
        self.avg_pool = tf.keras.layers.AveragePooling2D((2,2), strides=2)

    def call(self, inputs, training=False):
        outputs = self.batch_norm(inputs, training=training)
        outputs = self.relu(outputs)
        outputs = self.conv(outputs)
        outputs = self.dropout(outputs, training=training)
        outputs = self.avg_pool(outputs)

        return outputs