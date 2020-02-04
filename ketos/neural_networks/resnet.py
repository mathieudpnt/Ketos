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

""" resnet sub-module within the ketos.neural_networks module

    This module provides classes that to implement Residual Networks (ResNets).

    Contents:
        ResNetBlock class:
        ResNet class:
"""

import os
import tensorflow as tf
import numpy as np
from .losses import FScoreLoss
from .metrics import precision_recall_accuracy_f
from .nn_interface import RecipeCompat, NNInterface
import json
from zipfile import ZipFile
from glob import glob
from shutil import rmtree

class ResNetBlock(tf.keras.Model):
    def __init__(self, channels, strides=1, residual_path=False):
        super(ResNetBlock, self).__init__()

        self.channels = channels
        self.strides = strides
        self.residual_path = residual_path
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=(3,3), strides=self.strides,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=(3,3), strides=1,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()

        if residual_path == True:
            self.conv_down = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=(1,1), strides=self.strides,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())
            self.batch_norm_down = tf.keras.layers.BatchNormalization()
        
        self.dropout = tf.keras.layers.Dropout(0.0)

    def call(self,inputs, training=None):
        residual = inputs

        x = self.batch_norm_1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv_1(x)
        x = self.dropout(x)
        x = self.batch_norm_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.dropout(x)

        if self.residual_path:
            residual = self.batch_norm_down(inputs, training=training)
            residual = tf.nn.relu(residual)
            residual = self.conv_down(residual)
            x = self.dropout(x)

        x = x + residual
        return x


class ResNetArch(tf.keras.Model):

    def __init__(self, block_list, n_classes, initial_filters=16, **kwargs):
        super(ResNetArch, self).__init__(**kwargs)

        self.n_blocks = len(block_list)
        self.n_classes = n_classes
        self.block_list = block_list
        self.input_channels = initial_filters
        self.output_channels = initial_filters
        self.conv_initial = tf.keras.layers.Conv2D(filters=self.output_channels, kernel_size=(3,3), strides=1,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())

        self.blocks = tf.keras.models.Sequential(name="dynamic_blocks")

        for block_id in range(self.n_blocks):
            for layer_id in range(self.block_list[block_id]):
                #Frst layer of every block except the first
                if block_id != 0 and layer_id == 0:
                    block = ResNetBlock(self.output_channels, strides=2, residual_path=True)
                
                else:
                    if self.input_channels != self.output_channels:
                        residual_path = True
                    else:
                        residual_path = False
                    block = ResNetBlock(self.output_channels, residual_path=residual_path)

                self.input_channels = self.output_channels

                self.blocks.add(block)
            
            self.output_channels *= 2

        self.batch_norm_final = tf.keras.layers.BatchNormalization()
        self.average_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fully_connected = tf.keras.layers.Dense(self.n_classes)
        self.softmax = tf.keras.layers.Softmax()
    
    def call(self, inputs, training=None):

        output = self.conv_initial(inputs)

        output = self.blocks(output, training=training)
        output = self.batch_norm_final(output, training=training)
        output = tf.nn.relu(output)
        output = self.average_pool(output)
        output = self.fully_connected(output)
        output = self.softmax(output)

        return output



class ResNetInterface(NNInterface):

    @classmethod
    def build_from_recipe(cls, recipe):
        block_list = recipe['block_list']
        n_classes = recipe['n_classes']
        initial_filters = recipe['initial_filters']
        optimizer = recipe['optimizer']
        loss_function = recipe['loss_function']
        metrics = recipe['metrics']
        if 'secondary_metrics' in recipe.keys():
            secondary_metrics = recipe['secondary_metrics']
        else:
            secondary_metrics = None

        instance = cls(block_list=block_list, n_classes=n_classes, initial_filters=initial_filters, optimizer=optimizer, loss_function=loss_function, metrics=metrics, secondary_metrics=secondary_metrics)

        return instance

    @classmethod
    def read_recipe_file(cls, json_file, return_recipe_compat=True):
        with open(json_file, 'r') as json_recipe:
            recipe_dict = json.load(json_recipe)

        optimizer = cls.optimizer_from_recipe(recipe_dict['optimizer'])
        loss_function = cls.loss_function_from_recipe(recipe_dict['loss_function'])
        metrics = cls.metrics_from_recipe(recipe_dict['metrics'])
        if 'secondary_metrics' in recipe_dict.keys():
                secondary_metrics = cls.metrics_from_recipe(recipe_dict['secondary_metrics'])
            else:
                 secondary_metrics = None

        if return_recipe_compat == True:
            recipe_dict['optimizer'] = optimizer
            recipe_dict['loss_function'] = loss_function
            recipe_dict['metrics'] = metrics
            if 'secondary_metrics' in recipe_dict.keys():
                recipe_dict['secondary_metrics'] = secondary_metrics
            
        else:
            recipe_dict['optimizer'] = cls.optimizer_to_recipe(optimizer)
            recipe_dict['loss_function'] = cls.loss_function_to_recipe(loss_function)
            recipe_dict['metrics'] = cls.metrics_to_recipe(metrics)
            if 'secondary_metrics' in recipe_dict.keys():
                recipe_dict['secondary_metrics'] = cls.metrics_to_recipe(secondary_metrics)

        recipe_dict['block_list'] = recipe_dict['block_list']
        recipe_dict['n_classes'] = recipe_dict['n_classes']
        recipe_dict['initial_filters'] = recipe_dict['initial_filters']

        return recipe_dict

    def __init__(self, block_list, n_classes, initial_filters, optimizer, loss_function, metrics, secondary_metrics):
        self.block_list = block_list
        self.n_classes = n_classes
        self.initial_filters = initial_filters
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.secondary_metrics = secondary_metrics

        self.model=ResNetArch(block_list=block_list, n_classes=n_classes, initial_filters=initial_filters)
        self.compile_model()
        #self.metrics_names = self.model.metrics_names

        
        self.log_dir = None
        self.checkpoint_dir = None
        self.tensorboard_callback = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None

    def write_recipe(self):
        recipe = {}
        recipe['block_list'] = self.block_list
        recipe['n_classes'] = self.n_classes
        recipe['initial_filters'] = self.initial_filters
        recipe['optimizer'] = self.optimizer_to_recipe(self.optimizer)
        recipe['loss_function'] = self.loss_function_to_recipe(self.loss_function)
        recipe['metrics'] = self.metrics_to_recipe(self.metrics)
        if self.secondary_metrics is not None:
                recipe['secondary_metrics'] = cls.metrics_to_recipe(self.secondary_metrics)

        return recipe



   