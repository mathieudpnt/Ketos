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

class Seq2Seq(tf.keras.Model):
    def __init__(self, input_shape,
            temporal_conv_settings = [{'filters':196,'kernel_size':15,'strides':4},
                                      {'filters':196,'kernel_size':15,'strides':2},
                                      {'filters':196,'kernel_size':15,'strides':2}],
            seq_settings = [{'n_units':128}, {'n_units':128}]):
        super(Seq2Seq, self).__init__()

        self.in_shape = input_shape
        self.temporal_conv_settings = temporal_conv_settings
        self.seq_settings = seq_settings
        
        self.conv_block = tf.keras.models.Sequential(name="conv_block")
        for i,layer in enumerate(self.temporal_conv_settings):
            
            self.conv_block.add(tf.keras.layers.Conv1D(filters=layer['filters'], kernel_size=layer['kernel_size'], strides=layer['strides']))
            self.conv_block.add(tf.keras.layers.BatchNormalization())
            self.conv_block.add(tf.keras.layers.Activation('relu'))
            self.conv_block.add(tf.keras.layers.Dropout(0.8))
            
        self.seq_block = tf.keras.models.Sequential(name="seq_block")
        for layer in self.seq_settings:
            self.seq_block.add(tf.keras.layers.GRU(units=layer['n_units'], return_sequences=True))
            self.seq_block.add(tf.keras.layers.Dropout(0.8))
            self.seq_block.add(tf.keras.layers.BatchNormalization())

        self.final_dropout = tf.keras.layers.Dropout(0.8)
        self.final_block = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'),

        input_shape=(295,self.seq_settings[-1]['n_units']))

    def call(self, inputs, training=True):
        output = self.conv_block(inputs)
        output = self.seq_block(output)
        output = self.final_dropout(output)
        output = self.final_block(output)

        return output




