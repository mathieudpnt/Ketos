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

""" seq2seq sub-module within the ketos.neural_networks module

    This module provides a sequence to sequence neural network.

    Contents:
        seq2seq object:
        
"""

import tensorflow as tf


INPUT_SHAPE = (500,40)

seq2seq = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(196, kernel_size=15, strides=4, input_shape = INPUT_SHAPE),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Conv1D(196, kernel_size=15, strides=2, padding ='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Conv1D(196, kernel_size=4, strides=2, padding ='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.GRU(units=128, return_sequences=True),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GRU(units=128, return_sequences=True),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, activation='sigmoid'),
        input_shape=(296,128)
    )
     
])
