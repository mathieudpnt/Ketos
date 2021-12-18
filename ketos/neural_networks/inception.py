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
        ConvBatchNormRelu class
        InceptionBlock class
        Inception class
        InceptionInterface

"""

import tensorflow as tf
from .dev_utils.nn_interface import RecipeCompat, NNInterface
import json




default_inception_recipe =  {'n_blocks':3,
                    'n_classes':2,
                    'initial_filters':16,        
                    'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
                    'loss_function': RecipeCompat('BinaryCrossentropy', tf.keras.losses.BinaryCrossentropy),  
                    'metrics': [RecipeCompat('BinaryAccuracy',tf.keras.metrics.BinaryAccuracy),
                                RecipeCompat('Precision',tf.keras.metrics.Precision),
                                RecipeCompat('Recall',tf.keras.metrics.Recall)],
                    }

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
        x1 = self.conv1(x, training=training)
        x2 = self.conv2(x, training=training)
        x3_1 = self.conv3_1(x, training=training)
        x3_2 = self.conv3_2(x3_1, training=training)
        x4 = self.pool(x)
        x4 = self.pool_conv(x4, training=True)

        out = tf.concat([x1, x2, x3_2, x4], axis=3)

        return out


class InceptionArch(tf.keras.Model):
    """ Implements an Inception network, building on InceptionBlocks

        Args:
            n_blocks:int
                Number of Inception Blocks
            n_classes:int
                Number of possible classes 
            initial_filters:int
                Number of filters (i.e.: channels) in the first block
            pre_trained_base: instance of InceptionArch
                A pre-trained inception model from which the residual blocks will be taken. 
                Use by the the clone_with_new_top method when creating a clone for transfer learning
   
    
    """

    def __init__(self, n_blocks, n_classes, pre_trained_base=None, initial_filters=16, **kwargs):
        super(InceptionArch, self).__init__(**kwargs)

        self.input_channels = initial_filters
        self.output_channels = initial_filters
        self.n_blocks = n_blocks
        self.n_classes = n_classes
        self.initial_filters = initial_filters

        if pre_trained_base:
            self.conv1 = pre_trained_base[0]
            self.blocks = pre_trained_base[1]
        else:

            self.conv1 = ConvBatchNormRelu(self.initial_filters)

            self.blocks = tf.keras.models.Sequential(name='dynamic-blocks')

            for block_id in range(self.n_blocks):
                for layer_id in range(2):

                    if layer_id == 0:
                        block = InceptionBlock(self.output_channels, strides=2)
                    else:
                        block = InceptionBlock(self.output_channels, strides=1)

                    self.blocks.add(block)

                self.output_channels *= 2

        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(self.n_classes)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=None):
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
        output = self.conv1(inputs, training=training)
        output = self.blocks(output, training=training)
        output = self.avg_pool(output)
        output = self.dense(output)
        output = self.softmax(output)

        return output

    def freeze_init_layer(self):
        """Freeze the initial convolutional layer"""
        self.layers[0].trainable = False

    def unfreeze_init_layer(self):
        """Unfreeze the initial convolutional layer"""
        self.layers[0].trainable = True
    
    def freeze_block(self, block_ids):
        """ Freeze specific inception blocks

            Args:
                blocks_ids: list of ints
                    The block numbers to be freezed (starting from zero)
        """

        for block_id in block_ids:
            self.layers[1].layers[block_id].trainable = False

    def unfreeze_block(self, block_ids):
        """ Unfreeze specific inception blocks

            Args:
                blocks_ids: list of ints
                    The block numbers to be freezed (starting from zero)
        """
        for block_id in block_ids:
            self.layers[1].layers[block_id].trainable = True
    
    def freeze_top(self):
        """Freeze the classification block"""
        for layer in self.layers[2:]:
            layer.trainable = False
    
    def unfreeze_top(self):
        """Unfreeze the classification block"""
        for layer in self.layers[2:]:
            layer.trainable = True


    def get_feature_extraction_base(self):
        """ Retrive the feature extraction base (initial convolutional layer + residual blocks)
        
            Returns:
                list containing the feature extraction layers
        """

        return [self.conv1, self.blocks]

    def clone_with_new_top(self, n_classes=None, freeze_base=True):
        """ Clone this instance but replace the original classification top with a new (untrained) one
        
            Args:
                n_classes:int
                    The number of classes the new classification top should output.
                    If None(default), the original number of classes will be used.
                freeze_base:bool
                    If True, the weights of the feature extraction base will be froze (untrainable) in the new model.
                
            Returns:
                cloned_model: instance of InceptionArch
                    The new model with the old feature extraction base and new classification top.
         """
        if freeze_base == True:
            self.trainable = False

        if n_classes is None:
            n_classes = self.n_classes

        pre_trained_base = self.get_feature_extraction_base()
        cloned_model = type(self)(n_classes=n_classes, pre_trained_base=pre_trained_base)

        return cloned_model
    
        
class InceptionInterface(NNInterface):
    """ Creates an Inception model with the standardized Ketos interface.

        Args:
            num_blocks: int
                The number of inception blocks to be used. 
            
            n_classes:int
                The number of classes. The output layer uses a Softmax activation and
                will contain this number of nodes, resulting in model outputs with this
                many values summing to 1.0.

            initial_filters:int
                The number of filters used in the first ResNetBlock. Subsequent blocks 
                will have two times more filters than their previous block.

            optimizer: ketos.neural_networks.RecipeCompat object
                A recipe compatible optimizer (i.e.: wrapped by the ketos.neural_networksRecipeCompat class)

            loss_function: ketos.neural_networks.RecipeCompat object
                A recipe compatible loss_function (i.e.: wrapped by the ketos.neural_networksRecipeCompat class)

            metrics: list of ketos.neural_networks.RecipeCompat objects
                A list of recipe compatible metrics (i.e.: wrapped by the ketos.neural_networksRecipeCompat class).
                These metrics will be computed on each batch during training.
                
    """

    @classmethod
    def _build_from_recipe(cls, recipe, recipe_compat=True):
        """ Build an Inception model from a recipe.

            Args:
                recipe: dict
                    A recipe dictionary. optimizer, loss function
                    and metrics must be instances of ketos.neural_networks.RecipeCompat.
                    
                    Example recipe:
                    
                    >>> {{'n_blocks':3, # doctest: +SKIP
                    ...    'n_classes':2,
                    ...    'initial_filters':16,        
                    ...    'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
                    ...    'loss_function': RecipeCompat('BinaryCrossentropy', tf.keras.losses.BinaryCrossentropy),  
                    ...    'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)],
                    }


            Returns:
                An instance of InceptionInterface.

        """

        n_blocks = recipe['n_blocks']
        n_classes = recipe['n_classes']
        initial_filters = recipe['initial_filters']
        
        if recipe_compat == True:
            optimizer = recipe['optimizer']
            loss_function = recipe['loss_function']
            metrics = recipe['metrics']
            
        else:
            optimizer = cls._optimizer_from_recipe(recipe['optimizer'])
            loss_function = cls._loss_function_from_recipe(recipe['loss_function'])
            metrics = cls._metrics_from_recipe(recipe['metrics'])
            

        instance = cls(n_blocks=n_blocks, n_classes=n_classes, initial_filters=initial_filters, optimizer=optimizer, loss_function=loss_function, metrics=metrics)

        return instance

    @classmethod
    def _read_recipe_file(cls, json_file, return_recipe_compat=True):
        """ Read an Inception recipe saved in a .json file.

            Args:
                json_file:string
                    Full path (including filename and extension) to the .json file containing the recipe.
                return_recipe_compat:bool
                    If True, returns a dictionary where the optimizer, loss_function, metrics and 
                    secondary_metrics (if available) values are instances of the ketos.neural_networks.nn_interface.RecipeCompat.
                    The returned dictionary will be equivalent to:
                            
                            >>> {'n_blocks':3, # doctest: +SKIP
                            ... 'n_classes':2,
                            ... 'initial_filters':16,        
                            ... 'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
                            ... 'loss_function': RecipeCompat('BinaryCrossentropy', tf.keras.losses.BinaryCrossentropy),  
                            ... 'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)]}

                    If False, the optimizer, loss_function, metrics and secondary_metrics (if available) values will contain a
                    dictionary representation of such fields instead of the RecipeCompat objects:
                            >>> {'n_blocks':3, # doctest: +SKIP
                            ... 'n_classes':2,
                            ... 'initial_filters':16,        
                            ... 'optimizer': {'name':'Adam', 'parameters': {'learning_rate':0.005}},
                            ... 'loss_function': {'name':'BinaryCrossentropy', 'parameters':{}},  
                            ... 'metrics': [{'name':'CategoricalAccuracy', 'parameters':{}}]}

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

        return recipe_dict

    def __init__(self, n_blocks=default_inception_recipe['n_blocks'], n_classes=default_inception_recipe['n_classes'], initial_filters=default_inception_recipe['initial_filters'],
                       optimizer=default_inception_recipe['optimizer'], loss_function=default_inception_recipe['loss_function'], metrics=default_inception_recipe['metrics']):
        super(InceptionInterface, self).__init__(optimizer, loss_function, metrics)
        self.n_blocks = n_blocks
        self.n_classes = n_classes
        self.initial_filters = initial_filters

        self.model=InceptionArch(n_blocks=n_blocks, n_classes=n_classes, initial_filters=initial_filters)

    def _extract_recipe_dict(self):
        """ Create a recipe dictionary from an InceptionInterface instance.

            The resulting recipe contains all the fields necessary to build the same network architecture used by the instance calling this method.
            
            Returns:
                recipe:dict
                    A dictionary containing the recipe fields necessary to build the same network architecture.
                    The output is equivalent to:
                        >>> {'n_blocks':3, # doctest: +SKIP
                        ...    'n_classes':2,
                        ...    'initial_filters':16,        
                        ...    'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
                        ...    'loss_function': RecipeCompat('BinaryCrossentropy', tf.keras.losses.BinaryCrossentropy),  
                        ...    'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)]}
        """

        recipe = {}
        recipe['interface'] = type(self).__name__
        recipe['n_blocks'] = self.n_blocks
        recipe['n_classes'] = self.n_classes
        recipe['initial_filters'] = self.initial_filters
        recipe['optimizer'] = self._optimizer_to_recipe(self.optimizer)
        recipe['loss_function'] = self._loss_function_to_recipe(self.loss_function)
        recipe['metrics'] = self._metrics_to_recipe(self.metrics)
        
        return recipe


