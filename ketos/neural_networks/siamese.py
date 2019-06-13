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

""" siamese sub-module within the ketos.neural_networks module

    This module provides classes that implement Siamese Convolutional Neural Networks.

    Contents:
        SiameseCNNBranch class:
        SiameseCNN class:
"""

import tensorflow as tf

class SiameseCNNBranch(tf.keras.Model):
    def __init__(self):
        super(SiameseCNNBranch, self).__init__()
        self.seq = tf.keras.models.Sequential(name="siamese_branch")
        self.conv_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(10,10), use_bias=False,
                                            kernel_initializer=tf.random_normal_initializer(),
                                            kernel_regularizer=tf.keras.regularizers.l2(2e-4))
        self.max_pool_1 = tf.keras.layers.MaxPooling2D()


        self.conv_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(7,7),
                                            kernel_initializer=tf.random_normal_initializer(),
                                            kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                                            bias_initializer=tf.random_normal_initializer())
        self.max_pool_2 = tf.keras.layers.MaxPooling2D()

        self.conv_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4),
                                            kernel_initializer=tf.random_normal_initializer(),
                                            kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                                            bias_initializer=tf.random_normal_initializer())
        self.max_pool_3 = tf.keras.layers.MaxPooling2D()

        self.conv_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4,4),
                                            kernel_initializer=tf.random_normal_initializer(),
                                            kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                                            bias_initializer=tf.random_normal_initializer())
        self.max_pool_3 = tf.keras.layers.MaxPooling2D()

        self.flatten = tf.keras.layers.Flatten()
        self.fully_connected = tf.keras.layers.Dense(512,kernel_initializer=tf.random_normal_initializer(),
                                            kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                            bias_initializer=tf.random_normal_initializer())

    def call(self,inputs):
        branch = self.seq(inputs)
        branch = self.conv_1(branch)
        branch = tf.nn.relu(branch)
        branch = self.max_pool_1(branch)
        branch = self.conv_2(branch)
        branch = tf.nn.relu(branch)
        branch = self.max_pool_2(branch)
        branch = self.conv_3(branch)
        branch = tf.nn.relu(branch)
        branch = self.max_pool_3(branch)
        branch = self.flatten(branch)
        branch = self.fully_connected(branch)
        branch = tf.keras.activations.sigmoid(branch)

        return branch

class SiameseCNN(tf.keras.Model):
    def __init__(self):
        super(SiameseCNN, self).__init__()
        #self.input_layer = tf.keras.Input(shape = input_shape)
        
        self.branch1 = SiameseCNNBranch()
        self.branch2 = SiameseCNNBranch()
        
        
        self.L1_layer = tf.keras.layers.Lambda(lambda tensors:tf.keras.backend.abs(tensors[0] - tensors[1]))
        self.final_fully_connected = tf.keras.layers.Dense(1,activation='sigmoid',bias_initializer=tf.random_normal_initializer())


    def call(self, input1, input2):
       
        branch1_in = self.branch1(input1)
        branch2_in = self.branch1(input2)
        #branch2_in = self.branch2(input2)
        
        out = self.L1_layer([branch1_in, branch2_in])
        out = self.final_fully_connected(out)
        
        return out