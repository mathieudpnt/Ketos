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

""" cnn sub-module within the ketos.neural_networks module

    This module provides classes to implement Convolutional Neural Networks (CNNs).

    Contents:
        CNN class:
        CNNInterface class:
"""
import tensorflow as tf
from .nn_interface import NNInterface


vgg19_recipe = {'convolutional_layers':  [{'n_filters':64, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool': None, 'batch_normalization':True},
                                    {'n_filters':64, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool': {'pool_size':(2,2) , 'strides':(2,2)}, 'batch_normalization':True},
                                    {'n_filters':128, "filter_shape":(3,3), 'strides':1, 'padding':'valid','activation':'relu', 'max_pool':None, 'batch_normalization':True, },
                                    {'n_filters':128, "filter_shape":(3,3), 'strides':1, 'padding':'valid','activation':'relu', 'max_pool':{'pool_size':(2,2) , 'strides':(2,2)}, 'batch_normalization':True},
                                    {'n_filters':256, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True, },
                                    {'n_filters':256, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True, },
                                    {'n_filters':256, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True,},
                                    {'n_filters':256, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':(2,2) , 'strides':(2,2)}, 'batch_normalization':True},
                                    {'n_filters':512, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True, },
                                    {'n_filters':512, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True, },
                                    {'n_filters':512, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True, },
                                    {'n_filters':512, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':(2,2) , 'strides':(2,2)}, 'batch_normalization':True,},
                                    {'n_filters':256, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True,},
                                    {'n_filters':256, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True,},
                                    {'n_filters':256, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True,},
                                    {'n_filters':256, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':(2,2) , 'strides':(2,2)}, 'batch_normalization':True,}],
                 
                  'dense_layers':[{'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    {'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    {'n_hidden':1000, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},]

                    }                


alexnet_recipe = {'convolutional_layers':  [{'n_filters':96, "filter_shape":(11,11), 'strides':4, 'padding':'valid',  'activation':'relu', 'max_pool': {'pool_size':(3,3) , 'strides':(2,2)}, 'batch_normalization':True, },
                                    {'n_filters':256, "filter_shape":(5,5), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool': {'pool_size':(3,3) , 'strides':(2,2)}, 'batch_normalization':True, },
                                    {'n_filters':384, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True,},
                                    {'n_filters':384, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True,},
                                    {'n_filters':256, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':(3,3) , 'strides':(2,2)}, 'batch_normalization':True,},],
                  
                  'dense_layers':[{'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    {'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    {'n_hidden':1000, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5,}]

                    }



class CNNArch(tf.keras.Model):
    """ Implement a Convolutional Neural Network

        Args:
            convolutional_layers: list
                A list of dictionaries containing the detailed specification for the convolutional layers.
                Each layer is specified as a dictionary with the following format:
                {'n_filters':96, "filter_shape":(11,11), 'strides':4, 'padding':'valid', activation':'relu', 'max_pool': {'pool_size':(3,3) , 'strides':(2,2)}, 'batch_normalization':True}

            dense_layers: list
                A list of dictionaries containing the detailed specification for the fully connected layers.
                Each layer is specified as a dictionary with the following format:
                {'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5}

            n_classes:int
                The number of classes the network will be used to classify.
                The output will be this number of values representing the scores for each class. 
                Scores sum to 1.0.

    """


    def __init__(self, convolutional_layers, dense_layers, n_classes, **kwargs):
        super(CNNArch, self).__init__(**kwargs)

        self.convolutional_block = tf.keras.models.Sequential(name="convolutional_block")
        for conv_layer in convolutional_layers:
            self.convolutional_block.add(tf.keras.layers.Conv2D(filters=conv_layer['n_filters'], kernel_size=conv_layer['filter_shape'], strides=conv_layer['strides'], activation=conv_layer['activation'], padding=conv_layer['padding']))
            if conv_layer['max_pool'] is not None:
                self.convolutional_block.add(tf.keras.layers.MaxPooling2D(pool_size=conv_layer['max_pool']['pool_size'], strides=conv_layer['max_pool']['strides'] ))
            if conv_layer['batch_normalization'] == True:
                self.convolutional_block.add(tf.keras.layers.BatchNormalization())
            
        self.dense_block = tf.keras.models.Sequential(name="dense_block")
        for fc_layer in dense_layers:
            self.dense_block.add(tf.keras.layers.Dense(units=fc_layer['n_hidden'], activation=fc_layer['activation']))
            if fc_layer['batch_normalization'] == True:
                self.dense_block.add(tf.keras.layers.BatchNormalization())
            if fc_layer['dropout'] > 0.0:
                self.dense_block.add(tf.keras.layers.Dropout(fc_layer['dropout']))

        self.softmax = tf.keras.layers.Softmax(n_classes)

    def call(self, inputs, training=None):

        output = self.convolutional_block(inputs, training=training)
        output = self.dense_block(output, training=training)
        output = self.softmax(output)

        return output


class CNNInterface(NNInterface):
    """ Creates a CNN model with the standardized Ketos interface.

        Args:
             convolutional_layers: list
                A list of dictionaries containing the detailed specification for the convolutional layers.
                Each layer is specified as a dictionary with the following format:
                {'n_filters':96, "filter_shape":(11,11), 'strides':4, 'padding':'valid', activation':'relu', 'max_pool': {'pool_size':(3,3) , 'strides':(2,2)}, 'batch_normalization':True}

            dense_layers: list
                A list of dictionaries containing the detailed specification for the fully connected layers.
                Each layer is specified as a dictionary with the following format:
                {'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5}

            n_classes:int
                The number of classes the network will be used to classify.
                The output will be this number of values representing the scores for each class. 
                Scores sum to 1.0.

            optimizer: ketos.neural_networks.RecipeCompat object
                A recipe compatible optimizer (i.e.: wrapped by the ketos.neural_networksRecipeCompat class)

            loss_function: ketos.neural_networks.RecipeCompat object
                A recipe compatible loss_function (i.e.: wrapped by the ketos.neural_networksRecipeCompat class)

            metrics: list of ketos.neural_networks.RecipeCompat objects
                A list of recipe compatible metrics (i.e.: wrapped by the ketos.neural_networksRecipeCompat class).
                These metrics will be computed on each batch during training.

            secondary_metrics: list of ketos.neural_networks.RecipeCompat objects
                A list of recipe compatible metrics (i.e.: wrapped by the ketos.neural_networksRecipeCompat class).
                These can be used as additional metrics. Computed at each batch during training but only printed or
                logged as the average at the end of the epoch
                
    """

    
    DEFAULT_FC_LAYER = {'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5}
    @classmethod
    def convolutional_layers_from_conv_set(cls, conv_set):
        default_conv_layer = {'n_filters':64, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':(2,2) , 'strides':(2,2)}, 'batch_normalization':True}
        convolutional_layers = []
        for layer_parameters in conv_set:
            n_filters, max_pool = layer_parameters
            
            layer_details = default_conv_layer
            layer_details['n_filters'] = n_filters
            if max_pool is False:
                layer_details['max_pool'] = None
            convolutional_layers.append(layer_details)

        return convolutional_layers


    @classmethod
    def dense_layers_from_dense_set(cls, dense_set):
        default_dense_layer = {'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5}
        dense_layers = []
        for layer_parameters in dense_set:
            n_hidden = layer_parameters
            layer_details = default_dense_layer
            layer_details['n_hidden'] = n_hidden

            dense_layers.append(dense_layers)
        return dense_layers


    @classmethod
    def build_from_recipe(cls, recipe):
        if 'convolutional_layers' in recipe.keys() and 'dense_layers' in recipe.keys():
            convolutional_layers = recipe['convolutional_layers']
            dense_layers = recipe['dense_layers']
        elif 'conv_set' in recipe.keys() and 'dense_set' in recipe.keys():
            convolutional_layers = cls.convolutional_layers_from_conv_set(recipe['conv_set'])
            dense_layers = cls.dense_layers_from_dense_set(recipe['dense_set'])
        n_classes = recipe['n_classes']
        optimizer = recipe['optimizer']
        loss_function = recipe['loss_function']
        metrics = recipe['metrics']

        instance = cls(convolutional_layers=convolutional_layers, dense_layers=dense_layers, n_classes=n_classes, optimizer=optimizer, loss_function=loss_function, metrics=metrics)

        return instance
   
    def __init__(self, convolutional_layers, dense_layers, n_classes, optimizer, loss_function, metrics):
        self.convolutional_layers = convolutional_layers
        self.dense_layers = dense_layers
        self.n_classes = n_classes
        self.initial_filters = initial_filters
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics

        self.model=CNNArch(convolutional_layers=self.convolutional_layers, dense_layers=self.dense_layers, n_classes=n_classes)
        self.compile_model()
        #self.metrics_names = self.model.metrics_names

        
        self.log_dir = None
        self.checkpoint_dir = None
        self.tensorboard_callback = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
