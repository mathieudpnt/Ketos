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