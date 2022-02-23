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
import numpy as np
from .dev_utils.nn_interface import NNInterface, RecipeCompat
import json



vgg_like_recipe = {'convolutional_layers':  [{'n_filters':64, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool': None, 'batch_normalization':True},
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
                                    {'n_filters':512, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True,},
                                    {'n_filters':512, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True,},
                                    {'n_filters':512, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True,},
                                    {'n_filters':512, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':(2,2) , 'strides':(2,2)}, 'batch_normalization':True,}],
                 
                  'dense_layers':[{'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    {'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    ],
                  'n_classes': 2 ,
                  'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
                  'loss_function': RecipeCompat('BinaryCrossentropy', tf.keras.losses.BinaryCrossentropy),  
                  'metrics': [RecipeCompat('BinaryAccuracy',tf.keras.metrics.BinaryAccuracy)]
                  

}


                                    


alexnet_like_recipe = {'convolutional_layers':  [{'n_filters':96, "filter_shape":(11,11), 'strides':4, 'padding':'valid',  'activation':'relu', 'max_pool': {'pool_size':(3,3) , 'strides':(2,2)}, 'batch_normalization':True, },
                                    {'n_filters':256, "filter_shape":(5,5), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool': {'pool_size':(3,3) , 'strides':(2,2)}, 'batch_normalization':True, },
                                    {'n_filters':384, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True,},
                                    {'n_filters':384, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True,},
                                    {'n_filters':256, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':(3,3) , 'strides':(2,2)}, 'batch_normalization':True,},],
                  
                  'dense_layers':[{'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    {'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    ],

                  'n_classes': 2 ,
                  'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
                  'loss_function': RecipeCompat('BinaryCrossentropy', tf.keras.losses.BinaryCrossentropy),  
                  'metrics': [RecipeCompat('BinaryAccuracy',tf.keras.metrics.BinaryAccuracy)]                 


                    }

default_cnn_recipe = {'convolutional_layers':  [{'n_filters':32, "filter_shape":(8,8), 'strides':4, 'padding':'valid',  'activation':'relu', 'max_pool': {'pool_size':(3,3) , 'strides':(2,2)}, 'batch_normalization':True, },
                                    {'n_filters':64, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool': {'pool_size':(3,3) , 'strides':(2,2)}, 'batch_normalization':True, },],
                  
                  'dense_layers':[{'n_hidden':512, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    {'n_hidden':128, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},],

                  'n_classes': 2 ,
                  'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
                  'loss_function': RecipeCompat('BinaryCrossentropy', tf.keras.losses.BinaryCrossentropy),  
                  'metrics': [RecipeCompat('BinaryAccuracy',tf.keras.metrics.BinaryAccuracy)]
                  
                    }


default_cnn_1d_recipe = {'convolutional_layers':  [{'n_filters':8, "filter_shape":128, 'strides':2, 'padding':'causal', 'activation':'relu', 'max_pool': None, 'batch_normalization':True},
                                    {'n_filters':16, "filter_shape":64, 'strides':2, 'padding':'causal', 'activation':'relu', 'max_pool': {'pool_size': 8 , 'strides':8}, 'batch_normalization':True},
                                    {'n_filters':32, "filter_shape":32, 'strides':2, 'padding':'causal', 'activation':'relu', 'max_pool': {'pool_size': 8 , 'strides':8}, 'batch_normalization':True},
                                    {'n_filters':64, "filter_shape":16, 'strides':2, 'padding':'causal','activation':'relu', 'max_pool':None, 'batch_normalization':True, },
                                    {'n_filters':128, "filter_shape":8, 'strides':2, 'padding':'causal','activation':'relu', 'max_pool':None, 'batch_normalization':True},
                                    {'n_filters':256, "filter_shape":4, 'strides':2, 'padding':'causal', 'activation':'relu', 'max_pool':{'pool_size': 4 , 'strides': 4}, 'batch_normalization':True, },
                                    ],

                  'dense_layers':[{'n_hidden':512, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    {'n_hidden':128, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    ],

                  'n_classes': 2 ,
                  'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01),
                  'loss_function': RecipeCompat('CategoricalCrossentropy', tf.keras.losses.CategoricalCrossentropy, from_logits=True),  
                  'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy),   
                              RecipeCompat('Precision',tf.keras.metrics.Precision, class_id=1),
                              RecipeCompat('Recall',tf.keras.metrics.Recall, class_id=1)],  
                    }


class CNNArch(tf.keras.Model):
    """ Implement a Convolutional Neural Network

        Note: in addition to the dense layers specified in the 'dense_layers' argument, an extra dense
              layer will always be added to the end. The output of this layer is determined by the 'n_classes'
              parameter. 

        Args:
            convolutional_layers: list
                A list of dictionaries containing the detailed specification for the convolutional layers.
                Each layer is specified as a dictionary with the following format:

                >>> {'n_filters':96, "filter_shape":(11,11), 'strides':4, 'padding':'valid', activation':'relu', 'max_pool': {'pool_size':(3,3) , 'strides':(2,2)}, 'batch_normalization':True} # doctest: +SKIP

            dense_layers: list
                A list of dictionaries containing the detailed specification for the fully connected layers.
                Each layer is specified as a dictionary with the following format:
                
                >>> {'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5} # doctest: +SKIP
                
                This list should not include the output layr, which will be automatically added based on the 'n_classes' parameter.

             pre_trained_base: instance of CNNArch
                A pre-trained CNN model from which the residual blocks will be taken. 
                Use by the the clone_with_new_top method when creating a clone for transfer learning

            n_classes:int
                The number of classes the network will be used to classify.
                The output will be this number of values representing the scores for each class. 
                Scores sum to 1.0.
    """


    def __init__(self, convolutional_layers, dense_layers, n_classes, pre_trained_base=None, **kwargs):
        super(CNNArch, self).__init__(**kwargs)

        if pre_trained_base:
            self.convolutional_block = pre_trained_base[0]
        else:

            self.convolutional_block = tf.keras.models.Sequential(name="convolutional_block")
            for conv_layer in convolutional_layers:
                self.convolutional_block.add(tf.keras.layers.Conv2D(filters=conv_layer['n_filters'], kernel_size=conv_layer['filter_shape'], strides=conv_layer['strides'], activation=conv_layer['activation'], padding=conv_layer['padding']))
                if conv_layer['max_pool'] is not None:
                    self.convolutional_block.add(tf.keras.layers.MaxPooling2D(pool_size=conv_layer['max_pool']['pool_size'], strides=conv_layer['max_pool']['strides'] ))
                if conv_layer['batch_normalization'] == True:
                    self.convolutional_block.add(tf.keras.layers.BatchNormalization())
        
        self.flatten = tf.keras.layers.Flatten()

        self.dense_block = tf.keras.models.Sequential(name="dense_block")
        for fc_layer in dense_layers:
            self.dense_block.add(tf.keras.layers.Dense(units=fc_layer['n_hidden'], activation=fc_layer['activation']))
            if fc_layer['batch_normalization'] == True:
                self.dense_block.add(tf.keras.layers.BatchNormalization())
            if fc_layer['dropout'] > 0.0:
                self.dense_block.add(tf.keras.layers.Dropout(fc_layer['dropout']))

        
        self.dense_block.add(tf.keras.layers.Dense(n_classes))
        self.dense_block.add(tf.keras.layers.Softmax())

    
    def freeze_conv_block(self):
        """Freeze the convolutional block"""
        self.layers[0].trainable = False

    def unfreeze_conv_block(self):
        """Unfreeze the convolutional block"""
        self.layers[0].trainable = True
    
    def freeze_top(self):
        """Freeze the classification (dense) block"""
        for layer in self.layers[1:]:
            layer.trainable = False
    
    def unfreeze_top(self):
        """Unfreeze the classification (dense) block"""
        for layer in self.layers[1:]:
            layer.trainable = True

    def get_feature_extraction_base(self):
        """ Retrive the feature extraction base (initial convolutional layer + residual blocks)
        
            Returns:
                list containing the feature extraction layers
        """
        return [self.convolutional_block]

    def clone_with_new_top(self, n_classes=None, freeze_base=True):
        """ Clone this instance but replace the original classification top with a new (untrained) one
        
            Args:
                n_classes:int
                    The number of classes the new classification top should output.
                    If None(default), the original number of classes will be used.
                freeze_base:bool
                    If True, the weights of the feature extraction base will be froze (untrainable) in the new model.
                
            Returns:
                cloned_model: instance of CNNArch
                    The new model with the old feature extraction base and new classification top.
         """
        if freeze_base == True:
            self.trainable = False

        if n_classes is None:
            n_classes = self.n_classes

        pre_trained_base = self.get_feature_extraction_base()
        cloned_model = type(self)(n_classes=n_classes, pre_trained_base=pre_trained_base)

        return cloned_model

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
        output = self.convolutional_block(inputs, training=training)
        output = self.flatten(output)
        output = self.dense_block(output, training=training)
        
        return output




class CNN1DArch(tf.keras.Model):
    """ Implement an 1D (temporal) Convolutional Neural Network.

        Note: in addition to the dense layers specified in the 'dense_layers' argument, an extra dense \
              layer will always be added to the end. The output of this layer is determined by the 'n_classes' \
              parameter. 

        Args:
            convolutional_layers: list
                A list of dictionaries containing the detailed specification for the convolutional layers.
                Each layer is specified as a dictionary with the following format:

                >>> {'n_filters':96, "filter_shape":(11,11), 'strides':4, 'padding':'valid', activation':'relu', 'max_pool': {'pool_size':(3,3) , 'strides':(2,2)}, 'batch_normalization':True} # doctest: +SKIP

            dense_layers: list
                A list of dictionaries containing the detailed specification for the fully connected layers.
                Each layer is specified as a dictionary with the following format:

                >>> {'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5} # doctest: +SKIP

                This list should not include the output layr, which will be automatically added based on the 'n_classes' parameter.

            n_classes:int
                The number of classes the network will be used to classify.
                The output will be this number of values representing the scores for each class. 
                Scores sum to 1.0.

            pre_trained_base: instance of CNN1DArch
                A pre-trained CNN 1D model from which the residual blocks will be taken. 
                Use by the the clone_with_new_top method when creating a clone for transfer learning
    """

    def __init__(self, dense_layers, n_classes, pre_trained_base=None, convolutional_layers=None, **kwargs): 

        self.convolutional_layers = convolutional_layers
        self.dense_layers = dense_layers
        self.n_classes = n_classes
        super(CNN1DArch, self).__init__(**kwargs)

        if pre_trained_base:
            self.convolutional_block = pre_trained_base[0]
        else:
            self.convolutional_block = tf.keras.models.Sequential(name="convolutional_block")
            for conv_layer in self.convolutional_layers:
                self.convolutional_block.add(tf.keras.layers.Conv1D(filters=conv_layer['n_filters'], kernel_size=conv_layer['filter_shape'], strides=conv_layer['strides'], activation=conv_layer['activation'], padding=conv_layer['padding']))
                if conv_layer['max_pool'] is not None:
                    self.convolutional_block.add(tf.keras.layers.MaxPooling1D(pool_size=conv_layer['max_pool']['pool_size'], strides=conv_layer['max_pool']['strides'] ))
                if conv_layer['batch_normalization'] == True:
                    self.convolutional_block.add(tf.keras.layers.BatchNormalization())

        self.flatten = tf.keras.layers.Flatten()
        
        self.dense_block = tf.keras.models.Sequential(name="dense_block")
        for fc_layer in self.dense_layers:
            self.dense_block.add(tf.keras.layers.Dense(units=fc_layer['n_hidden'], activation=fc_layer['activation']))
            if fc_layer['batch_normalization'] == True:
                self.dense_block.add(tf.keras.layers.BatchNormalization())
            if fc_layer['dropout'] > 0.0:
                self.dense_block.add(tf.keras.layers.Dropout(fc_layer['dropout']))
        self.dense_block.add(tf.keras.layers.Dense(self.n_classes))
        self.dense_block.add(tf.keras.layers.Softmax())
        
    def freeze_conv_block(self):
        """Freeze the convolutional block"""
        self.layers[0].trainable = False

    def unfreeze_conv_block(self):
        """Unfreeze the convolutional block"""
        self.layers[0].trainable = True
    
    def freeze_top(self):
        """Freeze the classification (dense) block"""
        for layer in self.layers[1:]:
            layer.trainable = False
    
    def unfreeze_top(self):
        """Unfreeze the classification (dense) block"""
        for layer in self.layers[1:]:
            layer.trainable = True

    def get_feature_extraction_base(self):
        """ Retrive the feature extraction base (initial convolutional layer + residual blocks)
        
            Returns:
                list containing the feature extraction layers
        """
        return [self.convolutional_block]

    def clone_with_new_top(self, n_classes=None, freeze_base=True):
        """ Clone this instance but replace the original classification top with a new (untrained) one
        
            Args:
                n_classes:int
                    The number of classes the new classification top should output.
                    If None(default), the original number of classes will be used.

                freeze_base:bool
                    If True, the weights of the feature extraction base will be froze (untrainable) in the new model.
                
            Returns:
                cloned_model: instance of CNN1DArch
                    The new model with the old feature extraction base and new classification top.
        """
        if freeze_base == True:
            self.trainable = False

        if n_classes is None:
            n_classes = self.n_classes

        pre_trained_base = self.get_feature_extraction_base()
        cloned_model = type(self)(n_classes=n_classes, pre_trained_base=pre_trained_base)

        return cloned_model


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
        output = self.convolutional_block(inputs, training=training)
        output = self.flatten(output)
        output = self.dense_block(output, training=training)

        print("output shape: ", output.shape)

        return output


class CNNInterface(NNInterface):
    """ Creates a CNN model with the standardized Ketos interface.

        Args:
            convolutional_layers: list
                A list of dictionaries containing the detailed specification for the convolutional layers.
                Each layer is specified as a dictionary with the following format:

                >>> {'n_filters':96, "filter_shape":(11,11), 'strides':4, 'padding':'valid', activation':'relu', 'max_pool': {'pool_size':(3,3) , 'strides':(2,2)}, 'batch_normalization':True} # doctest: +SKIP

            dense_layers: list
                A list of dictionaries containing the detailed specification for the fully connected layers.
                Each layer is specified as a dictionary with the following format:
                
                >>> {'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5} # doctest: +SKIP

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

           
        Examples:

            >>> # Most users will create a model based on a Ketos recipe 
            >>> # The one below, specifies a CNN with 3 convolutional layers and 2 dense layers
            >>>
            >>> recipe = {'conv_set':[[64, False], [128, True], [256, True]], # doctest: +SKIP
            ...   'dense_set': [512, ],
            ...   'n_classes':2,
            ...   'optimizer': {'recipe_name':'Adam', 'parameters': {'learning_rate':0.005}},
            ...   'loss_function': {'recipe_name':'FScoreLoss', 'parameters':{}},  
            ...   'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]
            ... }
            >>> # To create the CNN, simply  use the  'build_from_recipe' method:
            >>> cnn = CNNInterface._build_from_recipe(recipe, recipe_compat=False) # doctest: +SKIP         
    """
    
    @classmethod
    def _convolutional_layers_from_conv_set(cls, conv_set):
        """ Create a detailed description of the convolutional layers based on the simplified description in 'conv_set'

            The resulting detailed description can then be used to build the convolutional layers in the model

            Args:
                conv_set:list
                    A list describing the convolutional layers in a CNN.
                    each layer is represented by a list of 2 elements: The number of filters (int) and
                    whether or not that layer is followed by a max_pooling operation (boolean).
                    Example:
                   
                    >>> [[64,False], [128, True]] # doctest: +SKIP
                   
                    This conv_set would describe two convolutional layers, with 64 and 128 filters, respectively.
                    Only the second would have max_pooling.

            Returns:
                convolutional_layers: list
                    A list of detailed layer description dictionaries.
                    Example: 
                   
                    >>> [{'n_filters':64, "filter_shape":[3,3], 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True}, # doctest: +SKIP
                    ...          {'n_filters':128, "filter_shape":[3,3], 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':[2,2] , 'strides':[2,2]}, 'batch_normalization':True},
                    ...          ]
                              

        """
        
        convolutional_layers = []
        for layer_parameters in conv_set:
            n_filters, max_pool = layer_parameters
            
            #default layer details
            layer_details = {'n_filters':64, "filter_shape":[3,3], 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':[2,2] , 'strides':[2,2]}, 'batch_normalization':True}
            layer_details['n_filters'] = n_filters
                       
            if max_pool is False:
                layer_details['max_pool'] = None
            convolutional_layers.append(layer_details)
            

        return convolutional_layers


    @classmethod
    def _dense_layers_from_dense_set(cls, dense_set):
        """ Create a detailed description of the dense layers based on the simplified description in 'dense_set'

            The resulting detailed description can then be used to build the convolutional layers in the model.

            Args:
                dense_set:list
                    A list describing the dense layers in a CNN.
                    Each layer is represented by a one integer describing the number of output nodes in that layer.
                    The number of input nodes is automatically determined from the previous layer.
                    Example: [512, 256] 
                    This cdense_set would describe two dense layers, with 512 and 256 nodes, respectively.
                    Note that, the last layer of a CNN does not need to be especified in the dense_set, as the output layer
                    is automatically created according with the number of classes to be classified.

            Returns:
                dense_layers: list
                    A list of detailed layer description dictionaries.
                    Example: 

                    >>> [{'n_hidden':512, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5}, # doctest: +SKIP
                    ...       {'n_hidden':256, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                    ...       ]  

        """

        dense_layers = []
        for layer_parameters in dense_set:
            n_hidden = layer_parameters
            layer_details = {'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5}
            layer_details['n_hidden'] = n_hidden

            dense_layers.append(layer_details)
        return dense_layers


    @classmethod
    def _build_from_recipe(cls, recipe, recipe_compat=True ):
        """ Build a CNN model from a recipe.

            Args:
                recipe: dict
                    A recipe dictionary. The optimizer, loss function
                    and metrics must be instances of ketos.neural_networks.RecipeCompat.
                    Example recipe:

                        >>> {'conv_set':[[64, False], [128, True], [256, True]], # doctest: +SKIP
                        ...  'dense_set': [512, ],
                        ...  'n_classes':2,
                        ...  'optimizer': {'name':'Adam', 'parameters': {'learning_rate':0.005}},
                        ...  'loss_function': {'name':'FScoreLoss', 'parameters':{}},  
                        ...  'metrics': [{'name':'CategoricalAccuracy', 'parameters':{}}]
                        ...  ]

                        The only optional field is 'secondary_metrics'.

                    Alternatively, the 'conv_set' and 'dense_set' can be replaced by detailed descriptions in
                    'convolutional_layers' and 'dense_layers'.
                    Note that these need to be provided in pairs ('conv_set' + 'dense_set' OR 'convolutional_layers' and 'dense_layers')
                    Example:

                        >>> {'conv_set': [{'n_filters':64, "filter_shape":[3,3], 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True}, # doctest: +SKIP
                        ...              {'n_filters':128, "filter_shape":[3,3], 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':[2,2] , 'strides':[2,2]}, 'batch_normalization':True},
                        ...              ],
                        ...  'dense_set': [{'n_hidden':512, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                        ...                {'n_hidden':256, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                        ...                ],
                        ...  'n_classes':2,
                        ...  'optimizer': {'name':'Adam', 'parameters': {'learning_rate':0.005}},
                        ...  'loss_function': {'name':'FScoreLoss', 'parameters':{}},  
                        ...  'metrics': [{'name':'CategoricalAccuracy', 'parameters':{}}],
                        ...  ]


            Returns:
                An instance of CNNInterface.
        """

        conv_set = None
        dense_set = None        
        if 'convolutional_layers' in recipe.keys() and 'dense_layers' in recipe.keys():
            convolutional_layers = recipe['convolutional_layers']
            dense_layers = recipe['dense_layers']
        elif 'conv_set' in recipe.keys() and 'dense_set' in recipe.keys():
            conv_set = recipe['conv_set']
            dense_set = recipe['dense_set']
            convolutional_layers = cls._convolutional_layers_from_conv_set(conv_set)
            dense_layers = cls._dense_layers_from_dense_set(dense_set)
            
        n_classes = recipe['n_classes']
        
        if recipe_compat == True:
            optimizer = recipe['optimizer']
            loss_function = recipe['loss_function']
            metrics = recipe['metrics']
            
        else:
            optimizer = cls._optimizer_from_recipe(recipe['optimizer'])
            loss_function = cls._loss_function_from_recipe(recipe['loss_function'])
            metrics = cls._metrics_from_recipe(recipe['metrics'])
            
        

        instance = cls(convolutional_layers=convolutional_layers, dense_layers=dense_layers, n_classes=n_classes, optimizer=optimizer, loss_function=loss_function, metrics=metrics)
        instance.conv_set = conv_set
        instance.dense_set = dense_set

        return instance
   
    @classmethod
    def _read_recipe_file(cls, json_file, return_recipe_compat=True):
        """ Read a CNN recipe saved in a .json file.

            Args:
                json_file:string
                    Full path (including silename and extension) to the .json file containing the recipe.
                return_recipe_compat:bool
                    If True, returns a dictionary where the optimizer, loss_function, metrics and 
                    secondary_metrics (if available) values are instances of the ketos.neural_networks.nn_interface.RecipeCompat.
                        The returned dictionary will be equivalent to:

                           >>> {'conv_set':[(64, False), (128, True), (256, True)], # doctest: +SKIP
                           ...  'dense_set: [512, 256],
                           ...  'convolutional_layers: ,
                           ...  'dense_layers: ,
                           ... 'n_classes': 2 ,
                           ... 'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
                           ... 'loss_function': RecipeCompat('FScoreLoss', FScoreLoss),  
                           ... 'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)]}

                    If False, the optimizer, loss_function, metrics and secondary_metrics (if available) values will contain a
                    dictionary representation of such fields instead of the RecipeCompat objects:
                        
                            >>> {'conv_set':[(64, False), (128, True), (256, True)], # doctest: +SKIP
                            ...    'dense_set: [512, 256],
                            ...    'convolutional_layers: ,
                            ...    'dense_layers: ,
                            ...    'n_classes': 2 ,
                            ...    'optimizer': {'name':'Adam', 'parameters': {'learning_rate':0.005}},
                            ...    'loss_function': {'name':'FScoreLoss', 'parameters':{}},  
                            ...    'metrics': [{'name':'CategoricalAccuracy', 'parameters':{}}]}

                Returns:
                    recipe, according to 'return_recipe_compat'.

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
            
        if 'convolutional_layers' in recipe_dict.keys() and 'dense_layers' in recipe_dict.keys():
            convolutional_layers = recipe_dict['convolutional_layers']
            dense_layers = recipe_dict['dense_layers']
        elif 'conv_set' in recipe.keys() and 'dense_set' in recipe_dict.keys():
            conv_set = recipe_dict['conv_set']
            dense_set = recipe_dict['dense_set']
            convolutional_layers = cls._convolutional_layers_from_conv_set(conv_set)
            dense_layers = cls._dense_layers_from_dense_set(dense_set)
            
        recipe_dict['conv_set'] = recipe_dict['conv_set']
        recipe_dict['dense_set'] = recipe_dict['dense_set']
        recipe_dict['convolutional_layers'] = recipe_dict['convolutional_layers']
        recipe_dict['dense_layers'] = recipe_dict['dense_layers']
        recipe_dict['n_classes'] = recipe_dict['n_classes']
        

        return recipe_dict


    def __init__(self, convolutional_layers=default_cnn_recipe['convolutional_layers'], dense_layers=default_cnn_recipe['dense_layers'],
                 n_classes=default_cnn_recipe['n_classes'], optimizer=default_cnn_recipe['optimizer'], loss_function=default_cnn_recipe['loss_function'], 
                 metrics=default_cnn_recipe['metrics']):
        super(CNNInterface, self).__init__(optimizer, loss_function, metrics)
        self.conv_set = None
        self.dense_det = None
        self.convolutional_layers = convolutional_layers
        self.dense_layers = dense_layers
        self.n_classes = n_classes
       
        self.model=CNNArch(convolutional_layers=self.convolutional_layers, dense_layers=self.dense_layers, n_classes=n_classes)
       


    def _extract_recipe_dict(self):
        """ Create a recipe dictionary from a CNNInterface instance.

            The resulting recipe contains all the fields necessary to build the same network architecture used by the instance calling this method.
            
            Returns:
                recipe:dict
                    A dictionary containing the recipe fields necessary to build the same network architecture.
                    Example:

                        >>> {'conv_set':[(64, False), (128, True), (256, True)], # doctest: +SKIP
                        ...  'dense_set: [512, 256],
                        ...  'convolutional_layers: ,
                        ...  'dense_layers: ,
                        ...  'n_classes':2,
                        ...  'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
                        ...  'loss_function': RecipeCompat('FScoreLoss', FScoreLoss),  
                        ...  'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)]}
        """

        recipe = {}
        recipe['interface'] = type(self).__name__
        recipe['conv_set'] = self.conv_set
        recipe['dense_set'] = self.dense_set
        recipe['convolutional_layers'] = self.convolutional_layers
        recipe['dense_layers'] = self.dense_layers
        recipe['n_classes'] = self.n_classes
        recipe['optimizer'] = self._optimizer_to_recipe(self.optimizer)
        recipe['loss_function'] = self._loss_function_to_recipe(self.loss_function)
        recipe['metrics'] = self._metrics_to_recipe(self.metrics)
        
        return recipe


class CNN1DInterface(CNNInterface):
    """ Create an 1D (temporal) CNN model with the standardized Ketos interface.

        Args:
            convolutional_layers: list
                A list of dictionaries containing the detailed specification for the convolutional layers.
                Each layer is specified as a dictionary with the following format:

                >>> {'n_filters':96, "filter_shape":(11,11), 'strides':4, 'padding':'valid', activation':'relu', 'max_pool': {'pool_size':(3,3) , 'strides':(2,2)}, 'batch_normalization':True} # doctest: +SKIP

            dense_layers: list
                A list of dictionaries containing the detailed specification for the fully connected layers.
                Each layer is specified as a dictionary with the following format:

                >>> {'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5} # doctest: +SKIP

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

         
        Examples:

            >>> # Most users will create a model based on a Ketos recipe 
            >>> # The one below, specifies a CNN with 3 convolutional layers and 2 dense layers
            >>>
            >>> recipe = {'conv_set':[[64, False], [128, True], [256, True]], # doctest: +SKIP
            ...   'dense_set': [512, ],
            ...   'n_classes':2,
            ...   'optimizer': {'name':'Adam', 'parameters': {'learning_rate':0.005}},
            ...   'loss_function': {'name':'FScoreLoss', 'parameters':{}},  
            ...   'metrics': [{'name':'CategoricalAccuracy', 'parameters':{}}]
            ... }
            >>> # To create the CNN, simply  use the  'build_from_recipe' method:
            >>> cnn = CNNInterface.build_from_recipe(recipe, recipe_compat=False) # doctest: +SKIP
                
    """

    @classmethod
    def _convolutional_layers_from_conv_set(cls, conv_set):
        """ Create a detailed description of the convolutional layers based on the simplified description in 'conv_set'

            The resulting detailed description can then be used to build the convolutional layers in the model

            Args:
                conv_set:list
                    A list describing the convolutional layers in a CNN.
                    each layer is represented by a list of 2 elements: The number of filters (int) and
                    whether or not that layer is followed by a max_pooling operation (boolean).
                    Example:
                   
                    >>> [[64,False], [128, True]] # doctest: +SKIP
                   
                    This conv_set would describe two convolutional layers, with 64 and 128 filters, respectively.
                    Only the second would have max_pooling.

            Returns:
                convolutional_layers: list
                    A list of detailed layer description dictionaries.
                    Example: 
                   
                    >>> [{'n_filters':64, "filter_shape":[3,3], 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True}, # doctest: +SKIP
                    ...          {'n_filters':128, "filter_shape":[3,3], 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':[2,2] , 'strides':[2,2]}, 'batch_normalization':True},
                    ...          ]
                              

        """
        
        convolutional_layers = []
        for layer_parameters in conv_set:
            n_filters, max_pool = layer_parameters
            
            #default layer details
            layer_details = {'n_filters':64, "filter_shape":64, 'strides':2, 'padding':'causal', 'activation':'relu', 'max_pool':{'pool_size':8 , 'strides':8}, 'batch_normalization':True}
            layer_details['n_filters'] = n_filters
                       
            if max_pool is False:
                layer_details['max_pool'] = None
            convolutional_layers.append(layer_details)
            

        return convolutional_layers


    @classmethod
    def _dense_layers_from_dense_set(cls, dense_set):
        """ Create a detailed description of the dense layers based on the simplified description in 'dense_set'

            The resulting detailed description can then be used to build the convolutional layers in the model.

            Args:
                dense_set:list
                    A list describing the dense layers in a CNN.
                    Each layer is represented by a one integer describing the number of output nodes in that layer.
                    The number of input nodes is automatically determined from the previous layer.
                    Example: [512, 256] 
                    This cdense_set would describe two dense layers, with 512 and 256 nodes, respectively.
                    Note that, the last layer of a CNN does not need to be especified in the dense_set, as the output layer
                    is automatically created according with the number of classes to be classified.

            Returns:
                dense_layers: list
                    A list of detailed layer description dictionaries.
                    Example: 

                    >>> [{'n_hidden':512, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5}, # doctest: +SKIP
                    ...       {'n_hidden':256, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                    ...       ]  

        """

        dense_layers = []
        for layer_parameters in dense_set:
            n_hidden = layer_parameters
            layer_details = {'n_hidden':512, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5}
            layer_details['n_hidden'] = n_hidden

            dense_layers.append(layer_details)
        return dense_layers


    @classmethod
    def _build_from_recipe(cls, recipe, recipe_compat=True ):
        """ Build a CNN model from a recipe.

            Args:
                recipe: dict
                    A recipe dictionary. The optimizer, loss function
                    and metrics must be instances of ketos.neural_networks.RecipeCompat.
                    Example recipe:

                        >>> {'conv_set':[[64, False], [128, True], [256, True]], # doctest: +SKIP
                        ...  'dense_set': [512, ],
                        ...  'n_classes':2,
                        ...  'optimizer': {'name':'Adam', 'parameters': {'learning_rate':0.005}},
                        ...  'loss_function': {'name':'FScoreLoss', 'parameters':{}},  
                        ...  'metrics': [{'name':'CategoricalAccuracy', 'parameters':{}}]
                        ...  ]

                        The only optional field is 'secondary_metrics'.

                    Alternatively, the 'conv_set' and 'dense_set' can be replaced by detailed descriptions in
                    'convolutional_layers' and 'dense_layers'.
                    Note that these need to be provided in pairs ('conv_set' + 'dense_set' OR 'convolutional_layers' and 'dense_layers')
                    Example:

                        >>> {'conv_set': [{'n_filters':64, "filter_shape":[3,3], 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True}, # doctest: +SKIP
                        ...              {'n_filters':128, "filter_shape":[3,3], 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':[2,2] , 'strides':[2,2]}, 'batch_normalization':True},
                        ...              ],
                        ...  'dense_set': [{'n_hidden':512, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                        ...                {'n_hidden':256, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                        ...                ],
                        ...  'n_classes':2,
                        ...  'optimizer': {'name':'Adam', 'parameters': {'learning_rate':0.005}},
                        ...  'loss_function': {'name':'FScoreLoss', 'parameters':{}},  
                        ...  'metrics': [{'name':'CategoricalAccuracy', 'parameters':{}}],
                        ...  ]


            Returns:
                An instance of CNNInterface.
        """

        conv_set = None
        dense_set = None        
        if 'convolutional_layers' in recipe.keys() and 'dense_layers' in recipe.keys():
            convolutional_layers = recipe['convolutional_layers']
            dense_layers = recipe['dense_layers']
        elif 'conv_set' in recipe.keys() and 'dense_set' in recipe.keys():
            conv_set = recipe['conv_set']
            dense_set = recipe['dense_set']
            convolutional_layers = cls._convolutional_layers_from_conv_set(conv_set)
            dense_layers = cls._dense_layers_from_dense_set(dense_set)
            
        n_classes = recipe['n_classes']
        
        if recipe_compat == True:
            optimizer = recipe['optimizer']
            loss_function = recipe['loss_function']
            metrics = recipe['metrics']
            
        else:
            optimizer = cls._optimizer_from_recipe(recipe['optimizer'])
            loss_function = cls._loss_function_from_recipe(recipe['loss_function'])
            metrics = cls._metrics_from_recipe(recipe['metrics'])
            
        

        instance = cls(convolutional_layers=convolutional_layers, dense_layers=dense_layers, n_classes=n_classes, optimizer=optimizer, loss_function=loss_function, metrics=metrics)
        instance.conv_set = conv_set
        instance.dense_set = dense_set

        return instance

    @classmethod
    def transform_batch(cls, x, y, n_classes=2):
        """ Transforms a training batch into the format expected by the network.

            When this interface is subclassed to make new neural_network classes, this method can be overwritten to
            accomodate any transformations required. Common operations are reshaping of input arrays and parsing or one hot encoding of the labels.

            Args:
                x:numpy.array
                    The batch of inputs with shape (batch_size, width, height)
                y:numpy.array
                    The batch of labels.
                    Each label must be represented as an integer, ranging from zero to n_classes
                    The array is expected to have a field named 'label'.
                n_classes:int
                    The number of possible classes for one hot encoding.
                    
                

            Returns:
                X:numpy.array
                    The transformed batch of inputs
                Y:numpy.array
                    The transformed batch of labels

            Examples:

                >>> import numpy as np
                >>> # Create a batch of 10 5x5 arrays
                >>> inputs = np.random.rand(10,5,5)
                >>> inputs.shape
                (10, 5, 5)

                    
                >>> # Create a batch of 10 labels (0 or 1)
                >>> labels = np.random.choice([0,1], size=10)
                >>> labels.shape
                (10,)

                >>> transformed_inputs, transformed_labels = NNInterface.transform_batch(inputs, labels, n_classes=2)
                >>> transformed_inputs.shape
                (10, 5, 5, 1)

                >>> transformed_labels.shape
                (10, 2)
                
        """

        X = cls._transform_input(x)
        if y.dtype.names is not None:
            Y = np.array([cls._to1hot(class_label=label, n_classes=n_classes) for label in y['label']])
        else:
            Y = np.array([cls._to1hot(class_label=label, n_classes=n_classes) for label in y])

        return (X,Y)

    @classmethod
    def _transform_input(cls,input):
        """ Transforms a training input to the format expected by the network.

            Similar to :func:`NNInterface.transform_train_batch`, but only acts on the inputs (not labels). Mostly used for inference, rather than training.
            When this interface is subclassed to make new neural_network classes, this method can be overwritten to
            accomodate any transformations required. Common operations are reshaping of an input.

            Args:
                input:numpy.array
                    An input instance. Must be of shape (n,m) or (k,n,m).

            Raises:
                ValueError if input does not have 2 or 3 dimensions.

            Returns:
                tranformed_input:numpy.array
                    The transformed batch of inputs

            Examples:
            
                >>> import numpy as np
                >>> # Create a batch of 10 5x5 arrays
                >>> batch_of_inputs = np.random.rand(10,5,5)
                >>> selected_input = batch_of_inputs[0]
                >>> selected_input.shape
                (5, 5)
                 
                >>> transformed_input = NNInterface._transform_input(selected_input)
                >>> transformed_input.shape
                (1, 5, 5, 1)

                # The input can also have shape=(1,n,m)
                >>> selected_input = batch_of_inputs[0:1]
                >>> selected_input.shape
                (1, 5, 5)
                 
                >>> transformed_input = NNInterface._transform_input(selected_input)
                >>> transformed_input.shape
                (1, 5, 5, 1)

                
        """

        if input.ndim == 1:
            transformed_input = input.reshape(1,input.shape[0],1)
        elif input.ndim == 2:
            transformed_input = input.reshape(input.shape[0],input.shape[1],1)
        else:
            raise ValueError("Expected input to have 1 or 2 dimensions, got {}({}) instead".format(input.ndims, input.shape))

        return transformed_input

    def __init__(self, convolutional_layers=default_cnn_1d_recipe['convolutional_layers'], dense_layers=default_cnn_1d_recipe['dense_layers'],
                 n_classes=default_cnn_1d_recipe['n_classes'], optimizer=default_cnn_1d_recipe['optimizer'], loss_function=default_cnn_1d_recipe['loss_function'], 
                 metrics=default_cnn_1d_recipe['metrics']):
        super(CNN1DInterface, self).__init__(optimizer=optimizer, loss_function=loss_function, metrics=metrics)
        self.conv_set = None
        self.dense_set = None
        self.convolutional_layers = convolutional_layers
        self.dense_layers = dense_layers
        self.n_classes = n_classes
       
        self.model = CNN1DArch(convolutional_layers=self.convolutional_layers, dense_layers=self.dense_layers, n_classes=n_classes)
       
