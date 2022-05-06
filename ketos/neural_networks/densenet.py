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

"""densenet sub-module within the ketos.neural_networks module

    This module provides classes to implement Dense Networks (DenseNets).

    Contents: ConvBlock class
              DenseBlock class
              TransitionBlock class
              DenseNetArch class
              DenseNetInterface class
        
"""

import tensorflow as tf
import numpy as np
from .dev_utils.nn_interface import RecipeCompat, NNInterface, NNArch
import json


default_densenet_recipe =  {'dense_blocks':[ 6, 12, 24, 16],
                    'growth_rate':32,
                    'compression_factor':0.5,
                    'n_classes':2,
                    'dropout_rate':0.2,
                    'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
                    'loss_function': RecipeCompat('CategoricalCrossentropy', tf.keras.losses.CategoricalCrossentropy),  
                    'metrics': [RecipeCompat('BinaryAccuracy',tf.keras.metrics.BinaryAccuracy),
                                RecipeCompat('Precision',tf.keras.metrics.Precision, class_id=1),
                                RecipeCompat('Recall',tf.keras.metrics.Recall, class_id=1)],
                    }





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
        """Calls the model on new inputs.

        In this case call just reapplies all ops in the graph to the new inputs (e.g. build a new computational graph from the provided inputs).

        Args:
            inputs: Tensor or list of tensors
                A tensor or list of tensors
            
            training: Bool
                Boolean or boolean scalar tensor, indicating whether to run the Network in training mode or inference mode.

        Returns:
                A tensor if there is a single output, or a list of tensors if there are more than one outputs.
        """
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
        """Calls the model on new inputs.

        In this case call just reapplies all ops in the graph to the new inputs (e.g. build a new computational graph from the provided inputs).

        Args:
            inputs: Tensor or list of tensors
                A tensor or list of tensors
            
            training: Bool
                Boolean or boolean scalar tensor, indicating whether to run the Network in training mode or inference mode.

        Returns:
                A tensor if there is a single output, or a list of tensors if there are more than one outputs.
        """
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
        """Calls the model on new inputs.

        In this case call just reapplies all ops in the graph to the new inputs (e.g. build a new computational graph from the provided inputs).

        Args:
            inputs: Tensor or list of tensors
                A tensor or list of tensors
            
            training: Bool
                Boolean or boolean scalar tensor, indicating whether to run the Network in training mode or inference mode.

        Returns:
                A tensor if there is a single output, or a list of tensors if there are more than one outputs.
        """
        outputs = self.batch_norm(inputs, training=training)
        outputs = self.relu(outputs)
        outputs = self.conv(outputs)
        outputs = self.dropout(outputs, training=training)
        outputs = self.avg_pool(outputs)

        return outputs


class DenseNetArch(NNArch):
    """Implements a DenseNet architecture, building on top of Dense and tansition blocks

        Args:
            block_sets: list of ints
                A list specifying the block sets and how many blocks each set contains.
                Example: [6, 12, 24, 16]  will create a DenseNet with 4 block sets containing 6, 12, 24 and 16
                dense blocks, with a total of 58 blocks.
            growth_rate:int
                The factor by which the number of filters (i.e.: channels) within each dense block grows.
            compression_factor: float
                The factor by which transition blocks reduce the number of filters (i.e.: channels) between dense blocks (between 0 and 1).
            dropout_rate: float
                The droput rate (between 0 and 1) used in each transition block. Use 0 for no dropout.
            n_classes:int
                The number of classes. The output layer uses a Softmax activation and
                will contain this number of nodes, resulting in model outputs with this
                many values summing to 1.0.
            pre_trained_base: instance of DenseNetArch
                A pre-trained densenet model from which the residual blocks will be taken. 
                Use by the the clone_with_new_top method when creating a clone for transfer learning
        """

        

    def __init__(self, dense_blocks, growth_rate, compression_factor, n_classes, dropout_rate, pre_trained_base=None):
        super(DenseNetArch, self).__init__()

        self.dense_blocks = dense_blocks
        self.growth_rate = growth_rate
        self.compression_factor = compression_factor
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate

        if pre_trained_base:
            self.initial_conv = pre_trained_base[0]
            self.initial_batch_norm = pre_trained_base[1]
            self.initial_relu = pre_trained_base[2]
            self.initial_pool = pre_trained_base[3]

            self.dense_blocks_seq = pre_trained_base[4]

        else:
            self.initial_conv = tf.keras.layers.Conv2D(2 * self.growth_rate, kernel_size=7, strides=2, padding="same")
            self.initial_batch_norm = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
            self.initial_relu = tf.keras.layers.Activation('relu')
            self.initial_pool = tf.keras.layers.MaxPool2D((2,2), strides=2)

            self.n_channels = 2 * self.growth_rate
            self.dense_blocks_seq = tf.keras.Sequential()
            for n_layers in self.dense_blocks:
                self.dense_blocks_seq.add(DenseBlock(growth_rate=self.growth_rate, n_blocks=n_layers))
                self.n_channels += n_layers * self.growth_rate
                self.dense_blocks_seq.add(TransitionBlock(n_channels=self.n_channels, compression_factor=self.compression_factor, dropout_rate=self.dropout_rate))

        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(self.n_classes)
        self.softmax = tf.keras.layers.Softmax()

    
    def call(self, inputs, training=False):
        """Calls the model on new inputs.

        In this case call just reapplies all ops in the graph to the new inputs (e.g. build a new computational graph from the provided inputs).

        Args:
            inputs: Tensor or list of tensors
                A tensor or list of tensors
            
            training: Bool
                Boolean or boolean scalar tensor, indicating whether to run the Network in training mode or inference mode.

        Returns:
                A tensor if there is a single output, or a list of tensors if there are more than one outputs.
        """
        outputs = self.call_frontend(inputs)
        outputs = self.initial_conv(outputs)
        outputs = self.initial_batch_norm(outputs, training=training)
        outputs = self.initial_relu(outputs)
        outputs = self.initial_pool(outputs)

        outputs = self.dense_blocks_seq(outputs)

        outputs = self.global_avg_pool(outputs)
        outputs = self.flatten(outputs)
        outputs = self.dense(outputs)
        outputs = self.softmax(outputs)

        return outputs


    def freeze_init_layer(self):
        """Freeze the initial convolutional layer"""
        self.layers[0].trainable = False

    def unfreeze_init_layer(self):
        """Unfreeze the initial convolutional layer"""
        self.layers[0].trainable = True
    
    def freeze_block(self, block_ids):
        """ Freeze specific dense blocks

            Args:
                blocks_ids: list of ints
                    The block numbers to be freezed (starting from zero)
        """

        for block_id in block_ids:
            self.layers[4].layers[block_id].trainable = False

    def unfreeze_block(self, block_ids):
        """ Unfreeze specific dense blocks

             Args:
                blocks_ids: list of ints
                    The block numbers to be freezed (starting from zero)
        """
        for block_id in block_ids:
            self.layers[4].layers[block_id].trainable = True
    
    def freeze_top(self):
        """Freeze the classification block"""
        for layer in self.layers[5:]:
            layer.trainable = False
    
    def unfreeze_top(self):
        """Unfreeze the classification block"""
        for layer in self.layers[5:]:
            layer.trainable = True


    def get_feature_extraction_base(self):
        """ Retrive the feature extraction base (initial convolutional layer + dense blocks)
        
            Returns:
                list containing the feature extraction layers
        """
        return [ self.initial_conv,
                self.initial_batch_norm,
                self.initial_relu,
                self.initial_pool,
                self.dense_blocks_seq]

    def clone_with_new_top(self, n_classes=None, freeze_base=True):
        """ Clone this instance but replace the original classification top with a new (untrained) one
        
            Args:
                n_classes:int
                    The number of classes the new classification top should output.
                    If None(default), the original number of classes will be used.
                freeze_base:bool
                    If True, the weights of the feature extraction base will be froze (untrainable) in the new model.
                
            Returns:
                cloned_model: instance of DenseNetArch
                    The new model with the old feature extraction base and new classification top.
         """
        if freeze_base == True:
            self.trainable = False

        if n_classes is None:
            n_classes = self.n_classes

        pre_trained_base = self.get_feature_extraction_base()
        cloned_model = type(self)(n_classes=n_classes, pre_trained_base=pre_trained_base)

        return cloned_model


class DenseNetInterface(NNInterface):


    @classmethod
    def _build_from_recipe(cls, recipe, recipe_compat=True):
        """ Build a DenseNet model from a recipe.

            Args:
                recipe: dict
                    A recipe dictionary. optimizer, loss function
                    and metrics must be instances of ketos.neural_networks.RecipeCompat.
                    
                    Example recipe:
                    
                    >>> {'dense_blocks': [6, 12, 24, 16], # doctest: +SKIP
                    ...    'growth_rate':32,
                    ...    'compression_factor':0.5,
                    ...    'n_classes':2,
                    ...    'dropout_rate':0.2,        
                    ...    'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
                    ...    'loss_function': RecipeCompat('BinaryCrossentropy', tf.keras.losses.BinaryCrossentropy),  
                    ...    'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)],
                    }
                     

            Returns:
                An instance of ResNetInterface.

        """

        dense_blocks = recipe['dense_blocks']
        growth_rate = recipe['growth_rate']
        compression_factor = recipe['compression_factor']
        n_classes = recipe['n_classes']
        dropout_rate = recipe['dropout_rate']
        
        
        if recipe_compat == True:
            optimizer = recipe['optimizer']
            loss_function = recipe['loss_function']
            metrics = recipe['metrics']
            
        else:
            optimizer = cls._optimizer_from_recipe(recipe['optimizer'])
            loss_function = cls._loss_function_from_recipe(recipe['loss_function'])
            metrics = cls._metrics_from_recipe(recipe['metrics'])
            

        instance = cls(dense_blocks=dense_blocks, growth_rate=growth_rate, compression_factor=compression_factor, n_classes=n_classes, dropout_rate=dropout_rate, optimizer=optimizer, loss_function=loss_function, metrics=metrics)

        return instance

    @classmethod
    def _read_recipe_file(cls, json_file, return_recipe_compat=True):
        """ Read a DenseNet recipe saved in a .json file.

            Args:
                json_file:string
                    Full path (including filename and extension) to the .json file containing the recipe.
                return_recipe_compat:bool
                    If True, returns a dictionary where the optimizer, loss_function, metrics and 
                    secondary_metrics (if available) values are instances of the ketos.neural_networks.nn_interface.RecipeCompat.
                    The returned dictionary will be equivalent to:
                            
                            >>> {'dense_blocks': [6, 12, 24, 16], # doctest: +SKIP
                            ...  'growth_rate':32,
                            ...  'compression_factor':0.5,
                            ...  'n_classes':2,
                            ...  'dropout_rate':0.2,        
                            ...  'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
                            ...  'loss_function': RecipeCompat('BinaryCrossentropy', tf.keras.losses.BinaryCrossentropy),  
                            ...  'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)],
                            }

                    If False, the optimizer, loss_function, metrics and secondary_metrics (if available) values will contain a
                    dictionary representation of such fields instead of the RecipeCompat objects:
                            >>> {'dense_blocks': [6, 12, 24, 16], # doctest: +SKIP
                            ...  'growth_rate':32,
                            ...  'compression_factor':0.5,
                            ...  'n_classes':2,
                            ...  'dropout_rate':0.2,        
                            ...  'optimizer': {'name':'Adam', 'parameters': {'learning_rate':0.005}},
                            ...  'loss_function': {'name':'BinaryCrossentropy', 'parameters':{}},  
                            ...  'metrics': [{'name':'CategoricalAccuracy', 'parameters':{}}]}

                Returns:
                    recipe, according to 'return_recipe_compat.

        """

        with open(json_file, 'r') as json_recipe:
            recipe_dict = json.load(json_recipe)

        optimizer = cls._optimizer_from_recipe(recipe_dict['optimizer'])
        loss_function = cls._loss_function_from_recipe(recipe_dict['loss_function'])
        metrics = cls._metrics_from_recipe(recipe_dict['metrics'])
        
        if return_recipe_compat == True:
            recipe_dict['optimizer'] = optimizer
            recipe_dict['loss_function'] = loss_function
            recipe_dict['metrics'] = metrics
            
        else:
            recipe_dict['optimizer'] = cls._optimizer_to_recipe(optimizer)
            recipe_dict['loss_function'] = cls._loss_function_to_recipe(loss_function)
            recipe_dict['metrics'] = cls._metrics_to_recipe(metrics)
        
        # recipe_dict['block_sets'] = recipe_dict['block_sets']
        # recipe_dict['n_classes'] = recipe_dict['n_classes']
        # recipe_dict['initial_filters'] = recipe_dict['initial_filters']

        return recipe_dict

    def __init__(self, dense_blocks=default_densenet_recipe['dense_blocks'], growth_rate=default_densenet_recipe['growth_rate'], compression_factor=default_densenet_recipe['compression_factor'], n_classes=default_densenet_recipe['n_classes'], dropout_rate=default_densenet_recipe['dropout_rate'], optimizer=default_densenet_recipe['optimizer'], loss_function=default_densenet_recipe['loss_function'], metrics=default_densenet_recipe['metrics']):
        super(DenseNetInterface, self).__init__(optimizer=optimizer, loss_function=loss_function, metrics=metrics)

        self.dense_blocks = dense_blocks
        self.growth_rate = growth_rate
        self.compression_factor = compression_factor
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate

        self.model = DenseNetArch(dense_blocks = self.dense_blocks, growth_rate=self.growth_rate, compression_factor=self.compression_factor, n_classes=self.n_classes, dropout_rate=self.dropout_rate)

    def _extract_recipe_dict(self):
        """ Create a recipe dictionary from a DenseNetInterface instance.

            The resulting recipe contains all the fields necessary to build the same network architecture used by the instance calling this method.
            
            Returns:
                recipe:dict
                    A dictionary containing the recipe fields necessary to build the same network architecture.
                    The output is equivalent to:
                       >>> {'dense_blocks': [6, 12, 24, 16], # doctest: +SKIP
                       ...  'growth_rate':32,
                       ...  'compression_factor':0.5,
                       ...  'n_classes':2,
                       ...  'dropout_rate':0.2,        
                       ...  'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
                       ...  'loss_function': RecipeCompat('BinaryCrossentropy', tf.keras.losses.BinaryCrossentropy),  
                       ...  'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)],
                        }
      
        """

        recipe = {}
        recipe['interface'] = type(self).__name__
        recipe['dense_blocks'] = self.dense_blocks
        recipe['growth_rate'] = self.growth_rate
        recipe['compression_factor'] = self.compression_factor
        recipe['n_classes'] = self.n_classes
        recipe['dropout_rate'] = self.dropout_rate
        recipe['optimizer'] = self._optimizer_to_recipe(self.optimizer)
        recipe['loss_function'] = self._loss_function_to_recipe(self.loss_function)
        recipe['metrics'] = self._metrics_to_recipe(self.metrics)
        
        return recipe


