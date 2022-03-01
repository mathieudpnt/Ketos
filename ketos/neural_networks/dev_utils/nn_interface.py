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

import tensorflow as tf
from .losses import FScoreLoss
from ...data_handling.parsing import parse_audio_representation
from ketos.utils import ensure_dir
from ketos.neural_networks.dev_utils.export import get_export_function
from zipfile import ZipFile
from shutil import rmtree
import numpy as np
import pandas as pd
import json
import os
import warnings


class RecipeCompat():
    """ Makes a loss function, metric or optimizer compatible with the Ketos recipe format.

        The resulting object can be included in a ketos recipe and read by the NNInterface (or it's subclasses)

        Args:
            recipe_name: str
                The name to be used in the recipe
            template: constructor
                The loss function, metric or optimizer constructor 
            kwargs
                Any keyword arguments to be passed to the constructor (func)

        Returns:
             A RecipeCompat object

        Examples:
          >>> # Example Metric
          >>> p = tf.keras.metrics.Precision
          >>> dec_p = RecipeCompat("precision", p)

          >>> # Example Optimizer
          >>> opt = tf.keras.optimizers.Adam
          >>> dec_opt = RecipeCompat("adam", opt, learning_rate=0.001)

          >>> # Example Loss
          >>> loss = tf.keras.losses.BinaryCrossentropy
          >>> dec_loss = RecipeCompat('binary_crossentropy', loss, from_logits=True)
    """
    def __repr__(self):
        return "{0} ketos recipe".format(self.recipe_name)

    def __init__(self, recipe_name, template, **kwargs):
        self.recipe_name = recipe_name
        self.args = kwargs
        self.template = template
        self.instance = self.instantiate_template(**kwargs)
            
    def instantiate_template(self, **template_kwargs):
        args = self.args.copy()
        args.update(template_kwargs)
        inst = self.template(**args)
        return inst

    def __call__(self, *args, **kwargs):
        result = self.instance(*args, **kwargs)
        return result


class NNInterface():
    """ General interface for neural network architectures in the ketos.neural_networks module.

        This class implements common methods for neural network models and is supposed to be subclassed. 
        When implementing new neural network architectures, the interface implemented in this clas can be inherited.

    Args:
        optimizer: RecipeCompat object
            An instance of the RecipeCompat class wrapping a tensorflow(-compatible) optimizer (e.g.:from tensorflow.keras.optimizers)
                
        loss_function: RecipeCompat object
            An instance of the RecipeCompat class wrappinf a tensorflow(-compatible) loss-function (e.g.:from tensorflow.keras.losses)
        
        metrics: list of RecipeCompat objects
          A list of instances of the RecipeCompat class wrapping a tensorflow(-compatible) metric (e.g.:from tensorflow.keras.metrics)

    Examples:     
        The following example shows how a newly defined network architecture could use the interface provided by NNInterface.

        First, the new architecture must be defined. Here, a simple multi-layer perceptron is defined in the following class.

        >>> class MLP(tf.keras.Model): # doctest: +SKIP
        ...    def __init__(self, n_neurons, activation):
        ...        super(MLP, self).__init__()
        ... 
        ...        self.dense = tf.keras.layers.Dense(n_neurons, activation=activation)
        ...        self.final_node = tf.keras.layers.Dense(1)
        ... 
        ...    def call(self, inputs):
        ...        output = self.dense(inputs)
        ...        output = self.dense(output)
        ...        output = self.final_node(output)
        ...        return output

        With the architecture, the interface to the MLP can be created by subclassing NNInterface:
        
        >>> from ketos.neural_networks.dev_utils import RecipeCompat, NNInterface   # doctest: +SKIP
        >>> class MLPInterface(NNInterface):  # doctest: +SKIP
        ...     @classmethod
        ...     def _build_from_recipe(cls, recipe, recipe_compat=True):
        ...         n_neurons = recipe['n_neurons']    # take the n_neurons parameter from the recipe instead of using the default
        ...         activation = recipe['activation']  # take the activation parameter from the recipe instead of using the default
        ...        
        ...         if recipe_compat == True:
        ...             optimizer = recipe['optimizer']
        ...             loss_function = recipe['loss_function']
        ...             metrics = recipe['metrics']
        ...            
        ...         else:
        ...             optimizer = cls._optimizer_from_recipe(recipe['optimizer'])
        ...             loss_function = cls._loss_function_from_recipe(recipe['loss_function'])
        ...             metrics = cls._metrics_from_recipe(recipe['metrics'])
        ...
        ...         instance = cls(n_neurons=n_neurons, activation=activation, optimizer=optimizer, loss_function=loss_function, metrics=metrics)
        ...         return instance
        ... 
        ...     @classmethod
        ...     def _read_recipe_file(cls, json_file, return_recipe_compat=True):
        ...         with open(json_file, 'r') as json_recipe:
        ...            recipe_dict = json.load(json_recipe)
        ...
        ...         optimizer = cls.optimizer_from_recipe(recipe_dict['optimizer'])
        ...         loss_function = cls.loss_function_from_recipe(recipe_dict['loss_function'])
        ...         metrics = cls.metrics_from_recipe(recipe_dict['metrics'])
        ...
        ...         if return_recipe_compat == True:
        ...             recipe_dict['optimizer'] = optimizer
        ...             recipe_dict['loss_function'] = loss_function
        ...             recipe_dict['metrics'] = metrics
        ...         else:
        ...             recipe_dict['optimizer'] = cls._optimizer_to_recipe(optimizer)
        ...             recipe_dict['loss_function'] = cls._loss_function_to_recipe(loss_function)
        ...             recipe_dict['metrics'] = cls._metrics_to_recipe(metrics)
        ...
        ...        recipe_dict['n_neurons'] = recipe_dict['n_neurons']    # read the n_neurons parameter from the recipe file
        ...        recipe_dict['activation'] = recipe_dict['activation']  # read the activation parameter from the recipe file
        ...        
        ...        return recipe_dict
        ...
        ...     def __init__(self, n_neurons, activation, optimizer, loss_function, metrics):
        ...        super(MLPInterface, self).__init__(optimizer, loss_function, metrics)
        ...        self.n_neurons = n_neurons
        ...        self.activation = activation
        ...        self.model = MLP(n_neurons=n_neurons, activation=activation)
        ...
        ...    def _extract_recipe_dict(self):
        ...        recipe = {}
        ...        recipe['optimizer'] = self._optimizer_to_recipe(self.optimizer)
        ...        recipe['loss_function'] = self._loss_function_to_recipe(self.loss_function)
        ...        recipe['metrics'] = self._metrics_to_recipe(self.metrics)
        ...        recipe['n_neurons'] = self.n_neurons
        ...        recipe['activation'] = self.activation
        ...        return recipe
    """
    valid_optimizers = {'Adadelta':tf.keras.optimizers.Adadelta,
                        'Adagrad':tf.keras.optimizers.Adagrad,
                        'Adam':tf.keras.optimizers.Adam,
                        'Adamax':tf.keras.optimizers.Adamax,
                        'Nadam':tf.keras.optimizers.Nadam,
                        'RMSprop':tf.keras.optimizers.RMSprop,
                        'SGD':tf.keras.optimizers.SGD,
                        }

    valid_losses = {'FScoreLoss':FScoreLoss,
                    'BinaryCrossentropy':tf.keras.losses.BinaryCrossentropy,
                    'CategoricalCrossentropy':tf.keras.losses.CategoricalCrossentropy,
                    'CategoricalHinge':tf.keras.losses.CategoricalHinge,
                    'CosineSimilarity':tf.keras.losses.CosineSimilarity,
                    'Hinge':tf.keras.losses.Hinge,
                    'Huber':tf.keras.losses.Huber,
                    'KLD':tf.keras.losses.KLD,
                    'LogCosh':tf.keras.losses.LogCosh,
                    'MAE':tf.keras.losses.MAE,
                    'MAPE':tf.keras.losses.MAPE,
                    'MeanAbsoluteError':tf.keras.losses.MeanAbsoluteError,
                    'MeanAbsolutePercentageError':tf.keras.losses.MeanAbsolutePercentageError,
                    'MeanSquaredError':tf.keras.losses.MeanSquaredError,
                    'MeanSquaredLogarithmicError':tf.keras.losses.MeanSquaredLogarithmicError,
                    'MSE':tf.keras.losses.MSE,
                    'MSLE':tf.keras.losses.MSLE,
                    'Poisson':tf.keras.losses.Poisson,
                    'SparseCategoricalCrossentropy':tf.keras.losses.SparseCategoricalCrossentropy,          
                    }

    valid_metrics = {'Accuracy':tf.keras.metrics.Accuracy,
                     'AUC':tf.keras.metrics.AUC,
                     'BinaryAccuracy':tf.keras.metrics.BinaryAccuracy,
                     'BinaryCrossentropy':tf.keras.metrics.BinaryCrossentropy,
                     'CategoricalAccuracy':tf.keras.metrics.CategoricalAccuracy,
                     'CategoricalCrossentropy':tf.keras.metrics.CategoricalCrossentropy,
                     'CategoricalHinge':tf.keras.metrics.CategoricalHinge,
                     'CosineSimilarity':tf.keras.metrics.CosineSimilarity,
                     'FalseNegatives':tf.keras.metrics.FalseNegatives,
                     'FalsePositives':tf.keras.metrics.FalsePositives,
                     'Hinge':tf.keras.metrics.Hinge,
                     'KLDivergence':tf.keras.metrics.KLDivergence,
                     'LogCoshError':tf.keras.metrics.LogCoshError,
                     'Mean':tf.keras.metrics.Mean,
                     'MeanAbsoluteError':tf.keras.metrics.MeanAbsoluteError,
                     'MeanAbsolutePercentageError':tf.keras.metrics.MeanAbsolutePercentageError,
                     'MeanIoU':tf.keras.metrics.MeanIoU,
                     'MeanRelativeError':tf.keras.metrics.MeanRelativeError,
                     'MeanSquaredError':tf.keras.metrics.MeanSquaredError,
                     'MeanSquaredLogarithmicError':tf.keras.metrics.MeanSquaredLogarithmicError,
                     'Poisson':tf.keras.metrics.Poisson,
                     'Precision':tf.keras.metrics.Precision,
                     'Recall':tf.keras.metrics.Recall,
                     'RootMeanSquaredError':tf.keras.metrics.RootMeanSquaredError,
                     'SensitivityAtSpecificity':tf.keras.metrics.SensitivityAtSpecificity,
                     'SparseCategoricalAccuracy':tf.keras.metrics.SparseCategoricalAccuracy,
                     'SparseCategoricalCrossentropy':tf.keras.metrics.SparseCategoricalCrossentropy,
                     'SparseTopKCategoricalAccuracy':tf.keras.metrics.SparseTopKCategoricalAccuracy,
                     'SpecificityAtSensitivity':tf.keras.metrics.SensitivityAtSpecificity,
                     'SquaredHinge':tf.keras.metrics.SquaredHinge,
                     'Sum':tf.keras.metrics.Sum,
                     'TopKCategoricalAccuracy':tf.keras.metrics.TopKCategoricalAccuracy,
                     'TrueNegatives':tf.keras.metrics.TrueNegatives,
                     'TruePositives':tf.keras.metrics.TruePositives,
                     }


    @classmethod
    def _to1hot(cls, class_label, n_classes=2):
        """ Create the one hot representation of class_label 

            Args:
                class_label: int
                    An integer number representing the class label
                n_class: int
                    The number of classes available
            
            Returns:
                one_hot: numpy.array
                    The one hot representation of the class_label in a 1 x n_classes array.

            Examples:
                >>> NNInterface._to1hot(class_label=0, n_classes=2)
                array([1., 0.])

                >>> NNInterface._to1hot(class_label=1, n_classes=2)
                array([0., 1.])

                >>> NNInterface._to1hot(class_label=1, n_classes=3)
                array([0., 1., 0.])

                >>> NNInterface._to1hot(class_label=1, n_classes=5)
                array([0., 1., 0., 0., 0.])
        """
        one_hot = np.zeros(n_classes)
        one_hot[class_label]=1.0
        return one_hot
    

    @classmethod
    def transform_batch(cls, x, y, n_classes=2):
        """ Transforms a training batch into the format expected by the network.

            When this interface is subclassed to make new neural_network classes, this method 
            can be overwritten to accomodate any transformations required. Common operations 
            are reshaping of input arrays and parsing or one hot encoding of the labels.

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
            # Keeping for now for backwards compatibility
            Y = np.array([cls._to1hot(class_label=label, n_classes=n_classes) for label in y['label']])
            warnings.warn("Deprecation warning: In a future version, the label parameter y will be a numpy aray or list and will not accept ['label'] keyword")
        else:    
            Y = np.array([cls._to1hot(class_label=label, n_classes=n_classes) for label in y])
        return (X,Y)


    @classmethod
    def _transform_input(cls,input):
        """ Transforms a training input to the format expected by the network.

            Similar to :func:`NNInterface.transform_train_batch`, but only acts on the 
            inputs (not labels). Mostly used for inference, rather than training.
            When this interface is subclassed to make new neural_network classes, this 
            method can be overwritten to accomodate any transformations required. Common 
            operations are reshaping of an input.

            Args:
                input:numpy.array
                    An input instance. Must have 2 or 3 dimensions.

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
        if input.ndim == 2:
            transformed_input = input.reshape(1, input.shape[0], input.shape[1], 1)
        elif input.ndim == 3:
            transformed_input = input.reshape(input.shape[0], input.shape[1], input.shape[2], 1)
        else:
            raise ValueError("Expected input to have 2 or 3 dimensions, got {} ({}) instead".format(input.ndim, input.shape))

        return transformed_input


    @classmethod
    def _transform_output(cls,output):
        """ Transforms the network output 

            When this interface is subclassed to make new neural_network classes, this method 
            can be overwritten to accomodate any transformations required. Common operations are 
            reshaping of an input and returning the class wih the highest score instead of a softmax vector.

            Args:
                output:np.array
                    The output neural network output. An array of one or more vectors of float scores 
                    that each add to 1.0.
            Returns:
                transformed_output:tuple
                    The transformed output, where the first value is the integer representing the highest 
                    classs in the rank the second is the respective score.

            Example:
                >>> import numpy as np
                >>> output = np.array([[0.2,0.1,0.7]])  
                >>> NNInterface._transform_output(output)
                (array([2]), array([0.7]))

                >>> output = np.array([[0.2,0.1,0.7],[0.05,0.65,0.3]])  
                >>> NNInterface._transform_output(output)
                (array([2, 1]), array([0.7 , 0.65]))
        """
        max_class = np.argmax(output, axis=-1)
        if output.shape[0] == 1:
            max_class_conf = output[0][max_class]
            transformed_output = (max_class[0], max_class_conf[0])
        elif output.shape[0] > 1:
            max_class_conf = np.array([output[i][c] for i, c in enumerate(max_class)])

        transformed_output = (max_class, max_class_conf)
        
        return transformed_output


    @classmethod
    def _optimizer_from_recipe(cls, optimizer):
        """ Create a recipe-compatible optimizer object from an optimizer dictionary

            Used when building a model from a recipe dictionary.
            
            Args:
                optimizer: optimizer dictionay
                    A dictionary with the following keys: {'name':..., 'parameters':{...}}.
                    The 'name' value must be a valid name as defined in the `valid_optimizers` 
                    class attribute. The 'parameters' value is a dictionary of keyword arguments 
                    to be used when building the optimizer (e.g.: {'learning_rate':0.001, 'momentum': 0.01})

            Returns:
                built_optimizer: 
                    A recipe-compatible optimizer object.

            Raises:
                ValueError if the optimizer name is not included in the valid_optimizers 
                class attribute.
        """
        recipe_name = optimizer['recipe_name']
        kwargs = optimizer['parameters']

        if recipe_name not in cls.valid_optimizers.keys():
            raise ValueError("Invalid optimizer name '{}'".format(recipe_name))
        built_optimizer = RecipeCompat(recipe_name,cls.valid_optimizers[recipe_name],**kwargs)

        return built_optimizer


    @classmethod
    def _optimizer_to_recipe(cls, optimizer):
        """ Create an optimizer dictionary from a recipe-compatible optimizer object

            Used when creating a ketos recipe that can be used to recreate the model.

            Args:
                optimizer: instance of RecipeCompat
                    An optimizer wrapped in a RecipeCompat object
            Returns:
                recipe_optimizer: dict 
                    A dictionary with the 'name' and 'parameters' keys.

            Raises:
                ValueError if the optimizer name is not included in the valid_optimizers 
                class attribute.
        """
        recipe_name = optimizer.recipe_name
        kwargs = optimizer.args

        if recipe_name not in cls.valid_optimizers.keys():
            raise ValueError("Invalid optimizer name '{}'".format(recipe_name))
        recipe_optimizer = {'recipe_name':recipe_name, 'parameters':kwargs}

        return recipe_optimizer


    @classmethod
    def _loss_function_from_recipe(cls, loss_function):
        """ Create a recipe-compatible loss object from a loss function dictionary

            Used when building a model from a recipe dictionary.

            Args:
                loss_function: loss function dictionay
                    A dictionary with the following keys: {'name':..., 'parameters':{...}}.
                    The 'name' value must be a valid name as defined in the `valid_losses` class attribute.
                    The 'parameters' value is a dictionary of keyword arguments to be used when building the loss_function
                    (e.g.: {'from_logits':True, 'label_smoothing':0.5})

            Returns:
                built_loss: 
                    A recipe-compatible loss function object.

            Raises:
                ValueError if the loss function name is not included in the valid_losses class attribute.
        """
        recipe_name = loss_function['recipe_name']
        kwargs = loss_function['parameters']

        if recipe_name not in cls.valid_losses.keys():
            raise ValueError("Invalid loss function name '{}'".format(recipe_name))
        built_loss = RecipeCompat(recipe_name, cls.valid_losses[recipe_name],**kwargs)

        return built_loss


    @classmethod
    def _loss_function_to_recipe(cls, loss_function):
        """ Create a loss function dictionary from a recipe-compatible loss function object

            Used when creating a ketos recipe that can be used to recreate the model.

            Args:
                loss_function: instance of RecipeCompat
                    A loss-function wrapped in a RecipeCompat object
            Returns:
                recipe_optimizer: dict 
                    A dictionary with the 'name' and 'parameters' keys.

            Raises:
                ValueError if the loss_function name is not included in the valid_losses class attribute.
        """
        recipe_name = loss_function.recipe_name
        kwargs = loss_function.args

        if recipe_name not in cls.valid_losses.keys():
            raise ValueError("Invalid loss function name '{}'".format(recipe_name))
        recipe_loss = {'recipe_name':recipe_name, 'parameters':kwargs}

        return recipe_loss


    @classmethod
    def _metrics_from_recipe(cls, metrics):
        """ Create a list of recipe-compatible metric objects from a metrics dictionary

            Used when building a model from a recipe dictionary.

            Args:
                metrics: list of metrics dictionaries
                    a list of dictionaries with the following keys: {'name':..., 'parameters':{...}}.
                    The 'name' value must be a valid name as defined in the `valid_metrics` class attribute.
                    The 'parameters' value is a dictionary of keyword arguments to be used when building the metrics
                    (e.g.: {'from_logits':True})

            Returns:
                built_metrics: 
                    A list of recipe-compatible metric objects.

            Raises:
                ValueError if any of the metric names is not included in the valid_metrics class attribute.
        """
        
        built_metrics = []
        for m in metrics:
            recipe_name = m['recipe_name']
            kwargs = m['parameters']
             
            if recipe_name not in cls.valid_metrics.keys():
                raise ValueError("Invalid metric name '{}'".format(m['recipe_name']))
            built_metrics.append(RecipeCompat(recipe_name, cls.valid_metrics[recipe_name], **kwargs))

        return built_metrics


    @classmethod
    def _metrics_to_recipe(cls, metrics):
        """ Create a metrics dictionary from a list of recipe-compatible metric objects
         
            Used when creating a ketos recipe that can be used to recreate the model

            Args:
                metrics: list of RecipeCompat objects
                    A list of RecipeCompat objects, each wrapping a metric.
            Returns:
                recipe_metrics: list of dicts
                    A list dictionaries, each with 'name' and 'parameters' keys.

            Raises:
                ValueError if any of the metric names is not included in the valid_metrics 
                class attribute.
        """
        recipe_metrics = []
        for m in metrics: 
            if m.recipe_name not in cls.valid_metrics.keys():
                raise ValueError("Invalid metric name '{}'".format(m['recipe_name']))
            recipe_metrics.append({'recipe_name':m.recipe_name, 'parameters':m.args})

        return recipe_metrics


    @classmethod
    def _read_recipe_file(cls, json_file, return_recipe_compat=True):
        """ Read a .json_file containing a ketos recipe and builds a recipe dictionary.

            When subclassing NNInterface to create interfaces to new neural networks, this 
            method can be overwritten to include other recipe fields relevant to the child class.

            Args:
                json_file:str
                    Path to the .json file (e.g.: '/home/user/ketos_recupes/my_recipe.json').
                return_recipe_compat:bool
                    If True, the returns a recipe-compatible dictionary (i.e.: where the values 
                    are RecipeCompat objects). If false, returns a recipe dictionary (i.e.: where 
                    the values are name+parameters dictionaries:  {'name':..., 'parameters':{...}})

            Returns:
                recipe_dict: dict
                    A recipe dictionary that can be used to rebuild a model.
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


    @classmethod
    def _write_recipe_file(cls, json_file, recipe):
        """ Write a recipe dictionary into a .json file

            Args:
                json_file: str
                    Path to the .json file (e.g.: '/home/user/ketos_recipes/my_recipe.json').
                
                recipe:dict
                    A recipe dictionary containing the optimizer, loss function and metrics 
                    in addition to other parameters necessary to build an instance of the neural network.

                    recipe = {"optimizer": RecipeCompat('adam',tf.keras.optimizers.Adam),
                              "loss_function":RecipeCompat('categorical_cross_entropy',tf.keras.losses.CategoricalCrossEntropy),
                              "metrics":[RecipeCompat('categorical_accuracy',tf.keras.metrics.CategoricalAccuracy)],
                              "another_parameter:32}
        """
        with open(json_file, 'w') as json_recipe:
            json.dump(recipe, json_recipe)


    @classmethod
    def _load_model(cls, recipe, weights_path):
        """ Load a model given a recipe dictionary and the saved weights.

            If multiple versions of the model are available in the folder indicated by weights_path 
            the latest will be selected. 

            Args:
                recipe: dict
                    A dictionary containing the recipe
                weights_path:str
                    The path to the folder containing the saved weights.
                    Saved weights are tensorflow chekpoint. The path should not include the checkpoint 
                    files, only the folder containing them. (e.g.: '/home/user/my_saved_models/model_a/')
        """
        instance = cls._build_from_recipe(recipe) 
        latest_checkpoint = tf.train.latest_checkpoint(weights_path)
        instance.model.load_weights(latest_checkpoint)
        instance.checkpoint_dir = weights_path
        return instance


    @classmethod
    def load_model_file(cls, model_file, new_model_folder='kt-tmp', overwrite=True, load_audio_repr=False, 
            replace_top=False, diff_n_classes=None):
        """ Same as :meth:`NNInterface.load`.

            Only included for backward compatibility.
            Will be removed in future versions.
        """
        warnings.warn("load_model_file is deprecated and will be removed in future versions. Use load instead.", category=DeprecationWarning)

        return cls.load(model_file=model_file, new_model_folder=new_model_folder, overwrite=overwrite, 
            load_audio_repr=load_audio_repr, replace_top=replace_top, diff_n_classes=diff_n_classes)


    @classmethod
    def load(cls, model_file, new_model_folder='kt-tmp', overwrite=True, load_audio_repr=False, 
            replace_top=False, diff_n_classes=None):
        """ Load a model from a ketos (.kt) model file.

            Args:
                model_file:str
                    Path to the ketos (.kt) file
                new_model_folder:str
                    Path to folder where files associated with the model will be stored. 
                    By default the files will be saved to a folder named 'kt-tmp' created 
                    in the current working directory.
                overwrite: bool
                    If True, the 'new_model_folder' will be overwritten. Default is True.
                replace_top: bool
                    If True, the classification top of the model will be replaced by a new, 
                    untrained one. What is actually replaced (i.e.: what exactly is the "top") 
                    is defined by the architecture. It is usually a block of dense layers with 
                    the appropriate activations. Default is False.
                diff_n_classes: int
                    Only relevant when 'replace_top' is True.
                    If the new model should have a different number of classes it can be specified 
                    by this parameter. If left as None, the new model will have the same number of 
                    classes as the original model.
                load_audio_repr: bool
                    If True, also return a dictionary with the audio representation. 
                    
            Raises:
                FileExistsError: If the 'new_model_folder' already exists and 'overwrite' is False.

            Returns:
                model_instance: 
                    The loaded model
                audio_repr: 
                    If load_audio_repr is True, also return a dictionary with the audio representation.
        """
        try:
            os.makedirs(new_model_folder)
        
        except FileExistsError:
            if overwrite == True:
                rmtree(new_model_folder)
                os.makedirs(new_model_folder)
        
            else:
                raise FileExistsError("Ketos needs a new folder for this model. Choose a folder name " \
                    + "that does not exist or set 'overwrite' to True to replace the existing folder")

        with ZipFile(model_file, 'r') as zip:
            zip.extractall(path=new_model_folder)
        
        recipe = cls._read_recipe_file(os.path.join(new_model_folder,"recipe.json"))
        model_instance = cls._load_model(recipe,  os.path.join(new_model_folder, "checkpoints"))

        if replace_top == True:
            new_n_classes = recipe['n_classes']
            if diff_n_classes is not None:
                new_n_classes = diff_n_classes
            
            model_with_new_top = model_instance.model.clone_with_new_top(n_classes=new_n_classes)
            model_instance.model = model_with_new_top
               
        if load_audio_repr is True:
            audio_repr = []
            f = open(os.path.join(new_model_folder,"audio_repr.json"), 'r')
            json_content = json.load(f)
            for section, rep in json_content.items():
                audio_repr.append({section:parse_audio_representation(rep)})
            
            return model_instance, audio_repr
        
        return model_instance


    @classmethod
    def _build_from_recipe(cls, recipe):
        """ Build a model from a recipe dictionary

            When subclassing NNInterface to create interfaces for new neural networks, the method
            can be overwritten to include all the recipe fields relevant to the new class.

            Args:
                recipe:dict
                    A recipe dictionary
        """
        optimizer = recipe['optimizer']
        loss_function = recipe['loss_function']
        metrics = recipe['metrics']
        instance = cls(optimizer=optimizer, loss_function=loss_function, metrics=metrics)
        return instance
    

    @classmethod
    def build_from_recipe_file(cls, recipe_file):
        """ Same as :meth:`NNInterface.build`.

            Only included for backward compatibility.
            Will be removed in future versions.
        """
        warnings.warn("build_from_recipe_file is deprecated and will be removed in future versions. Use build instead.", category=DeprecationWarning)

        return cls.build(recipe_file)


    @classmethod
    def build(cls, recipe_file):
        """ Build a model from a recipe file

            Args:
                recipe:str
                    path to .json file containing the recipe

            Returns:
                instance:
                    An instance of the neural network interface 
        """
        recipe = cls._read_recipe_file(recipe_file)
        instance = cls._build_from_recipe(recipe)
        return instance


    def __init__(self, optimizer, loss_function, metrics):
        
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.model = None
        
        self._log_dir = None
        self._checkpoint_dir = None
        self._tensorboard_callback = None
        self._train_generator = None
        self._val_generator = None
        self._test_generator = None

        self._train_loss = tf.keras.metrics.Mean(name='train_loss')
        self._val_loss = tf.keras.metrics.Mean(name='val_loss')
        self._train_metrics = []
        self._val_metrics = []
        for m in self.metrics:
            self._train_metrics.append(m.instantiate_template(name='train_' + m.recipe_name))
            self._val_metrics.append(m.instantiate_template(name='val_' + m.recipe_name))

        self._early_stopping_monitor = {"metric": 'val_loss',
                                        "decreasing": True,
                                        "period":10,
                                        "min_epochs": 5,
                                        "max_epochs": None,
                                        "delta" : 0.1,
                                        "baseline":0.5}

    def _extract_recipe_dict(self):
        """ Create a recipe dictionary from a neural network instance.

            The resulting recipe contains all the fields necessary to build the same network 
            architecture used by the instance calling this method.
            When subclassing NNInterface to create interfaces for new neural networks, this method 
            can be overwritten to match the recipe fields expected by :func:`build_from_recipe`

            Returns:
                recipe:dict
                    A dictionary containing the recipe fields necessary to build the same network 
                    architecture used by the instance calling this method
        """
        recipe = {}
        recipe['optimizer'] = self._optimizer_to_recipe(self.optimizer)
        recipe['loss_function'] = self._loss_function_to_recipe(self.loss_function)
        recipe['metrics'] = self._metrics_to_recipe(self.metrics)
        return recipe


    def save_recipe_file(self, recipe_file):
        """ Creates a recipe from an existing neural network instance and save it into a .json file.

            This method is a convenience method that wraps :func:`write_recipe` and :func:`write_recipe_file`

            Args:
                recipe_file:str
                    Path to .json file in which the recipe will be saved.
        """
        recipe = self._extract_recipe_dict()
        self._write_recipe_file(json_file=recipe_file, recipe=recipe)


    def save_model(self, model_file=None, output_name=None, **kwargs):
        """ Same as :meth:`NNInterface.save`.

            Only included for backward compatibility.
            Will be removed in future versions.
        """
        warnings.warn("save_model is deprecated and will be removed in future versions. Use save instead.", category=DeprecationWarning)

        self.save(model_file=model_file, output_name=output_name, **kwargs)


    def save(self, model_file=None, output_name=None, checkpoint_name=None, **kwargs):
        """ Save the trained model to a file.

            The model can be saved to three different formats: 

             * ketos (.kt)
             * protobuf (.pb)
             * ketos-protobuf (.ktpb), compatible with PAMGuard

            The format will be inferred from the extension given to the output file, as 
            follows

             * .kt (or any other extension): ketos 
             * .pb: protobuf
             * .ktpb: ketos-protobuf (PAMGuard compatible)

            The save method accepts different optional arguments depending on the format.
            See the documentation of export functions for more information:
            
             * ketos: :func:`export.export_to_ketos`
             * protobuf: :func:`export.export_to_protobuf`
             * ketos-protobuf: :func:`export.export_to_ketos_protobuf`

            TODO: remove deprecated argument model_file

            Args:
                output_name: str
                    Path to the output file
                checkpoint_name: str
                    The name of the checkpoint to be saved (e.g.:cp-0015.ckpt).
                    If None, will use the latest checkpoints
                audio_repr: dict
                    Audio representation dictionary. Optional for the ketos format (.kt), 
                    required for the ketos-protobuf (.ktpb) format, not applicable for 
                    the protobuf (.pb) format. It is also possible to specify the path 
                    to a json file containing the audio representation.
        """
        if model_file is not None:
            warnings.warn("model_file is deprecated and will be removed in future versions. Use output_name instead.", category=DeprecationWarning)
            if output_name is None:
                output_name = model_file

        # attempt to load weights from checkpoint
        if isinstance(self.checkpoint_dir, str) and os.path.isdir(self.checkpoint_dir):
            if checkpoint_name == None:
                latest = tf.train.latest_checkpoint(self.checkpoint_dir)
                checkpoint_name = os.path.basename(latest)

            self.model.load_weights(os.path.join(self.checkpoint_dir, checkpoint_name))

        else:
            print(f"Warning: model.checkpoint_dir does not point to a valid directory ({self.model.checkpoint_dir})")

        f = get_export_function(output_name)

        f(model=self, output_name=output_name, checkpoint_name=checkpoint_name, **kwargs)


    @property
    def train_generator(self):
        return self._train_generator

    @train_generator.setter
    def train_generator(self, train_generator):
        """ Link a batch generator (used for training) to this instance.

            Args:
                train_generator: instance of BatchGenerator
                    A batch generator that provides training data during the training loop 
        """
        self._train_generator = train_generator    

    @property
    def val_generator(self):
        return self._val_generator

    @val_generator.setter
    def val_generator(self, val_generator):
        """ Link a batch generator (used for validation) to this instance.

            Args:
                val_generator: instance of BatchGenerator
                    A batch generator that provides validation data during the training loop 
        """
        self._val_generator = val_generator    

    @property
    def test_generator(self):
        return self._test_generator

    @test_generator.setter
    def test_generator(self, test_generator):
        """ Link a batch generator (used for testing) to this instance.

            Args:
                test_generator: instance of BatchGenerator
                    A batch generator that provides test data
        """
        self._test_generator = test_generator    
    
    @property
    def log_dir(self):
        return self._log_dir

    @log_dir.setter
    def log_dir(self, log_dir):
        """ Defines the directory where tensorboard log files and .csv log files can be stored
        
             Note: Creates folder if it does not exist. If it already exists, this method does not delete any content.

             Args:
                 log_dir:str
                     Path to the directory 
        """
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)
    

    def add_learning_rate_scheduler(self, scheduler_type="PiecewiseConstantDecay",**kwargs):
        """ Add a learning rate scheduler to the current neural network interface.

            Notes: - This method must be called before training and after an optimizer has been defined.
                   - Keep in mind that in the schedulers a 'step' corresponds to each time the optmization algorithm is called. 
                     Normally this means that each batch is a step (i.e.: each epoch has several steps). 
                     
            Args:
                scheduler_type:str
                    One of four scheduler types: 
                    `PiecewiseConstantDecay <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PiecewiseConstantDecay>`_ , 
                    `ExponentialDecay, <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay>`_ 
                    `InverseTimeDecay <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/InverseTimeDecay>`_ , or
                    `PolynomialDecay <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PolynomialDecay>`_.

                    Each type also requires additional arguments, as detailed below.

                    PiecewiseConstantDecay:

                    boundaries:list
                        A list of Tensors or ints or floats with strictly increasing entries, and with all elements 
                        having the same type as the optimizer step.
                    values:list
                        A list of Tensors or floats or ints that specifies the values for the intervals defined by 
                        boundaries. It should have one more element than boundaries, and all elements should have the same type.

                    ExponentialDecay:

                    initial_learning_rate:float
                        The initial learning rate.
                    decay_steps: int
                        The decay steps (must be positive). In each step, the learning rate is calculated as 
                        initial_learning_rate * decay_rate ^ (step / decay_steps)
                    decay_rate:float
                        The decay rate.
                    staircase:bool
                        If True decay the learning rate at discrete intervals

                    InverseTimeDecay:

                    initial_learning_rate:float
                        The initial learning rate.
                    decay_steps:int
                        How often to apply decay.
                    decay_rate:float
                            The decay rate.
                    staircase:bool
                        Whether to apply decay in a discrete staircase, as opposed to continuous, fashion.

                    PolynomialDecay:

                    initial_learning_rate:float
                        The initial learning rate.
                    decay_steps:int
                        Must be positive. The number of steps in which the end_learning_rate should be reached.
                    end_learning_rate:float
                        The minimal end learning rate.
                    power:float
                        The power of the polynomial. Defaults to linear, 1.0.
                    cycle:bool
                        Whether or not it should cycle beyond decay_steps.
        """
        scheduler_types = {"PiecewiseConstantDecay":tf.keras.optimizers.schedules.PiecewiseConstantDecay,
                            "ExponentialDecay":tf.keras.optimizers.schedules.ExponentialDecay,
                            "InverseTimeDecay":tf.keras.optimizers.schedules.InverseTimeDecay,
                            "PolynomialDecay":tf.keras.optimizers.schedules.PolynomialDecay}

        assert scheduler_type in scheduler_types.keys(), ValueError("{0} is not a valid scheduler type. Accepted values are: {1}".format(scheduler_type, list(scheduler_types.keys())))  
        scheduler = scheduler_types[scheduler_type](**kwargs)

        self.optimizer.instance = self.optimizer.instantiate_template(learning_rate = scheduler)

    
    @property
    def early_stopping_monitor(self):
        r""" Sets an early stopping monitor.

            An early stopping monitor is a dictionary specifying
            how a target metric should be monitored during training.
            When the conditions are met, the training loop will be stopped
            and the model will keep the set of weights that resulted in the
            best value for the target metric.

            The following parameters are expected:

                metric: str
                    The name of the metric to be monitored. It must be one the metrics
                    defined when creating a neural network interface, either through 
                    the 'metrics' argument of the class constructor or the 'metrics' field in a recipe.
                    The name must be prefixed by 'train\_' or 'val\_', indicating weather the training or
                    validation metric should be monitored.
                decreasing: bool,
                    If True, improvements will be indicated by a decrease in the metric value during training. 
                    If False, improvements will be defined as an increase in the metric value.
                period: int
                    The number of epochs the training loop will continue without any improvement before training is stopped.
                    Example: If period is 5, training will stop if the target metric does not improve for 5 consecutive epochs.
                min_epochs: int
                    The number of epochs to train for before starting to monitor.
                delta: float
                    The minimum difference between the current metric value and the best
                    value recorded since the monitor started. An improvement is only considered if
                    (current value - best value) <= delta (if decreasing is True) or 
                    (current value - best value) >= delta (if decreasing is False)
                baseline: float or None
                    If this value is reached, training will stop immediately.
                    If None, this parameter is ignored.
        """
        return self._early_stopping_monitor


    @early_stopping_monitor.setter
    def early_stopping_monitor(self, parameters):
        valid_metrics = [m.name for m in self._train_metrics] + [m.name for m in self._val_metrics] + [self._train_loss.name] + [self._val_loss.name]
        assert parameters['metric'] in  valid_metrics, "Invalid metric. Must be one of {}".format(str(valid_metrics)) 
        self._early_stopping_monitor = parameters

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @checkpoint_dir.setter
    def checkpoint_dir(self, checkpoint_dir):
        """ Defines the directory where tensorflow checkpoint files can be stored

            Args:
                log_dir:str
                    Path to the directory

        """
        self._checkpoint_dir = checkpoint_dir
        os.makedirs(self._checkpoint_dir, exist_ok=True)

    
    def _set_tensorboard_callback(self):
        """ Link tensorboard callback to this instances model, so that tensorboard logs can be saved
        """

        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        self.tensorboard_callback.set_model(self.model)
        
    def _print_metrics(self, metric_values):
        """ Print the metric values to the screen.

            This method can be overwritten to customize the message.

            Args:
                metric_value:list
                    List of metric values. Usually returned by model.train_on_batch or generated by custom metrics.
        """
        message  = [self.model.metrics_names[i] + ": {:.3f} ".format(metric_values[i]) for i in range(len(self.model.metrics_names))]
        print(''.join(message))


    def _name_logs(self, logs, prefix="train_"):
        """ Attach the prefix string to each log name.

            Args:
                logs:list
                   List of log values
                prefix:str
                    Prefix to be added to the logged metric name
        
            Returns:
                named_log: zip
                    A zip iterator that yields a tuple: (prefix + log metric name, log value)
        """
        named_logs = {}
        for l in zip(self.metrics_names, logs):
            named_logs[prefix+l[0]] = l[1]
        
        return named_logs

    @tf.function
    def _train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_function.instance(labels, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.instance.apply_gradients(zip(gradients, self.model.trainable_variables))
        self._train_loss(loss)
        for train_metric in self._train_metrics:
            train_metric(labels, predictions)

    @tf.function
    def _val_step(self,inputs, labels):
        predictions = self.model(inputs, training=False)
        v_loss = self.loss_function.instance(labels, predictions)
        self._val_loss(v_loss)
        for val_metric in self._val_metrics:
            val_metric(labels, predictions)
            

    def _get_metric_value(self, metric_name):
        if metric_name == 'train_loss':
            return self._train_loss.result()
        elif metric_name == "val_loss":
            return self._val_loss.result()
        elif metric_name.startswith("train_"):
            for m in self._train_metrics:
                if m.name == metric_name:
                    return m.result()
        elif metric_name.startswith("val_"):
            for m in self._val_metrics:
                if m.name == metric_name:
                    return m.result()


    def train_loop(self, n_epochs, verbose=True, validate=True, log_tensorboard=False, 
        tensorboard_metrics_name='tensorboard_metrics', log_csv=False, csv_name='log.csv', 
        checkpoint_freq=5, early_stopping=False):
        """ Train the model

            Typically, before starting the training loop, a few steps will already have been taken:
            
            >>> #Set the batch generator for the training data
            >>> model.train_generator = my_train_generator # doctest: +SKIP
            >>> #Set the batch generator for the validation data (optional; only if the validate option is set to True)
            >>> model.val_generator = my_val_generator # doctest: +SKIP
            >>> # Set the log_dir
            >>> model.log_dir = "./my_logs" # doctest: +SKIP
            >>> # Set the checkpoint_dir
            >>> model.checkpoint_dir = "./my_checkpoints" # doctest: +SKIP
            >>> model.train_loop(n_epochs=50) # doctest: +SKIP

            Args:
                n_epochs:int
                    The number of epochs (i.e.: passes through the entire training dataset, as defined 
                    by the train_generator attribute)
                verbose:bool
                    If True, print summary metrics at the end of each epoch
                validate:bool
                    If True, evaluate the model on the validation data (as defined by the val_generator 
                    attribute) at the end of each epoch
                log_tensorboard:bool
                    If True, log the training and validation (if validate is True) metrics in the tensoraboard 
                    format.
                    See 'tensorboard_metrics_name' below.
                tensorboard_metrics_name:string
                    The name of the directory where the tensorboard metrics will be saved. This directory 
                    will be created within the path specified by the log_dir attribute.
                    Default is 'tensorboard_metrics'. Only relevant if log_tensorboard is True.
                log_csv:bool
                    If True, log the training and validation (if validate is True) metrics in a csv file 
                    (see csv_name).
                    
                    The csv will have the following columns,

                     * epoch: the epoch number, starting from 1
                     * loss: the value of the loss metric
                     * dataset: 'train' or 'val' (only when validate is True)
                    
                    In addition, each metric defined by the metrics attribute will be added as a column.

                    Example:
                    
                    epoch,loss,dataset,CategoricalAccuracy,Precision,Recall
                    1,0.353,train,0.668,0.653,0.796
                    1,0.560,val,0.448,0.448,1.0
                    ...
                    50,0.053,train,0.968,0.953,0.986
                    50,0.160,val,0.848,0.748,0.838

                checkpoint_freq:int
                    The frequency (in epochs) with which checkpoints (i.e.: the model weights) will be 
                    saved to the directory defined by the checkpoint_dir attribute.
                    This number should not exceed 'n_epochs'. If early_stopping  (see below) is used, 
                    this parameter is ignored and every epoch is saved as a checkpoint
                    so that the model state can be set to whichever chekpopint is selected by the 
                    early stopping monitor. 
                    If this value is <=0, no checkpoints will be saved.

                early_stopping: bool
                    If False, train for n_epochs. If True, use the early_stop_monitor to stop training when 
                    the conditions defined there are reached (or n_epochs is reached, whichever 
                    happens first).
                    When training is stopped by the early stopping monitor, an attribute 'last_epoch_with_improvement' 
                    will be added to the object.
                    This attribute holds the epoch number (starting from zero) that had the best metric value 
                    based on the conditions set by the early_stopping_monitor.
                    The 'last_epoch_with_improvement' reflects the current state of the weights when trained 
                    is stopped early.
        """
        assert checkpoint_freq <= n_epochs, "The checkpoint frequency ({0}) cannot be greater than the number of epochs ({1})".format(checkpoint_freq, n_epochs)

        if log_csv == True:
            column_names = ['epoch', 'loss', 'dataset'] + [ m.recipe_name for m in self.metrics]
            log_csv_df = pd.DataFrame(columns = column_names)

        if log_tensorboard == True:
            tensorboard_writer = tf.summary.create_file_writer(os.path.join(self._log_dir, tensorboard_metrics_name))
            tensorboard_writer.set_as_default()

        if early_stopping == True:
            early_stopping_metric = []
            best_metric_value = None
            self.last_epoch_with_improvement = 0
            epochs_without_improvement = 0
            should_stop = False
            checkpoint_freq = 1

        for epoch in range(n_epochs):
            #Reset the metric accumulators
            self._train_loss.reset_states()
            for train_metric in self._train_metrics:
                train_metric.reset_states()
            
            self._val_loss.reset_states()
            for val_metric in self._val_metrics:
                val_metric.reset_states()
                
            for train_batch_id in range(self._train_generator.n_batches):
                train_X, train_Y = next(self._train_generator)  
                self._train_step(train_X, train_Y)
                                
            if verbose == True:
                print("\n====================================================================================")
                print("Epoch: {} \ntrain_loss: {}".format(epoch + 1, self._train_loss.result()))
                print("".join([m.name + ": {:.3f} ".format(m.result().numpy()) for m in self._train_metrics]))

            if log_csv == True:
                log_row = [epoch + 1, self._train_loss.result().numpy(), "train"]
                log_row = log_row + [m.result().numpy() for m in self._train_metrics]
                log_csv_df = log_csv_df.append(pd.Series(log_row, index = log_csv_df.columns), ignore_index=True)
            
            if log_tensorboard == True:
                tf.summary.scalar('train_loss', data=self._train_loss.result().numpy(), step=epoch)
                for m in self._train_metrics:
                    tf.summary.scalar(m.name, data=m.result().numpy(), step=epoch)
            
            if validate == True:
                for val_batch_id in range(self._val_generator.n_batches):
                    val_X, val_Y = next(self._val_generator)
                    self._val_step(val_X, val_Y)                                    
                if verbose == True:
                    print("val_loss: {}".format(self._val_loss.result()))
                    print("".join([m.name + ": {:.3f} ".format(m.result().numpy()) for m in self._val_metrics]))

                if log_csv == True:
                    log_row = [epoch + 1, self._val_loss.result().numpy(), "val"]
                    log_row = log_row + [m.result().numpy() for m in self._val_metrics]
                    log_csv_df = log_csv_df.append(pd.Series(log_row, index = log_csv_df.columns), ignore_index=True)

                if log_tensorboard == True:
                    tf.summary.scalar('val_loss', data=self._val_loss.result().numpy(), step=epoch)
                    for m in self._val_metrics:
                        tf.summary.scalar(m.name, data=m.result().numpy(), step=epoch)

            if verbose == True:
                print("\n====================================================================================")
            
            if (checkpoint_freq > 0) and ((epoch + 1)  % checkpoint_freq == 0):
                checkpoint_name = "cp-{:04d}.ckpt".format(epoch + 1)
                self.model.save_weights(os.path.join(self._checkpoint_dir, checkpoint_name))
            
            if early_stopping == True:
               
                print("\nFocus metric", self._early_stopping_monitor['metric'])
                current_early_stopping_metric = (self._get_metric_value(self._early_stopping_monitor['metric']))
                if best_metric_value is None:
                    best_metric_value = current_early_stopping_metric

                if epoch >= self._early_stopping_monitor['min_epochs']:
                    
                    if self._early_stopping_monitor['decreasing'] == True:
                        if (self._early_stopping_monitor['baseline'] is not None) and (current_early_stopping_metric <= self._early_stopping_monitor['baseline']):
                            should_stop = True
                            self.last_epoch_with_improvement = epoch
                        else:
                            current_delta = current_early_stopping_metric - best_metric_value
                            if current_delta < 0 and (abs(current_delta) > self._early_stopping_monitor['delta']): #metric is decreasing = improvement
                                epochs_without_improvement = 0
                                self.last_epoch_with_improvement = epoch
                                best_metric_value = current_early_stopping_metric
                            else:                 # metric is not decreasing = no improvement
                                epochs_without_improvement += 1
                                
                    elif self._early_stopping_monitor['decreasing'] == False:
                        if (self._early_stopping_monitor['baseline'] is not None) and (current_early_stopping_metric >= self._early_stopping_monitor['baseline']):
                            should_stop = True
                            self.last_epoch_with_improvement = epoch
                        else:
                            current_delta = current_early_stopping_metric - best_metric_value
                            print("abs:", abs(current_delta))
                            if current_delta > 0 and (abs(current_delta) > self._early_stopping_monitor['delta']): #metric is increasing = improvement
                                epochs_without_improvement = 0
                                self.last_epoch_with_improvement = epoch
                                best_metric_value = current_early_stopping_metric
                            else:                 # metric is not increasing = no improvement
                                epochs_without_improvement += 1

                    print("\nEpochs without improvement:", epochs_without_improvement)  
                    print("\nCurrent value:", current_early_stopping_metric)
                    print("\nBest value:", best_metric_value)

                    if epochs_without_improvement > self._early_stopping_monitor['period']:
                        should_stop = True
                    
                    if should_stop == True:
                        break
        
        if log_csv == True:
            log_csv_df.to_csv(os.path.join(self._log_dir, csv_name))
        
        if early_stopping:
            last_checkpoint_with_improvement = "cp-{:04d}.ckpt".format(self.last_epoch_with_improvement + 1)
            self.model.load_weights(os.path.join(self.checkpoint_dir, last_checkpoint_with_improvement))
            return {'checkpoint_name':last_checkpoint_with_improvement}


    def run_on_test_generator(self, return_raw_output=False, compute_val_metrics=True, verbose=True, return_metrics=False):
        """ Run the model on the test generator

            Args:
                return_raw_output:bool
                    If False, the model output will be transformed by :func:`transform_output`.
                    If True, the model output will be returned without any modifications.
                compute_val_metrics: bool
                    If True, compute the same metrics used for validation when running the model on the test generator
                verbose: bool
                    If True and compute_val_metrics is also true, print the results.
                return_metrics: bool
                    Only relevant if 'compute_val_metrics=True'.
                    If True, return a dictionary with the metrics
                
            Returns:
                output
                    The corresponding batch of model outputs
                metrics
                    A dictionary with the validation metrics.
                    Only returned if both 'compute_val_metrics' and 'return_metrics' are True.
        """        
        if compute_val_metrics:
            self._val_loss.reset_states()
            for val_metric in self._val_metrics:
                val_metric.reset_states()

        predictions = []
        for batch_id in range(self._test_generator.n_batches):
                    X, Y = next(self._test_generator)
                    if compute_val_metrics: self._val_step(X, Y)
                    predictions.append(self.model(X, training=False))
                                           
        predictions = np.array(predictions)
        
        if return_raw_output == False:
            reshaped_predictions = predictions.reshape(-1, predictions.shape[2])
            predictions = self._transform_output(reshaped_predictions)

        if compute_val_metrics == True:                                    
            if verbose == True:
                print("loss: {}".format(self._val_loss.result()))
                print("".join([m.name.split('val_')[1] + ": {:.3f} ".format(m.result().numpy()) for m in self._val_metrics]))
            if return_metrics == True:
                metrics_dict ={m.name.split('val_')[1]:m.result().numpy() for m in self._val_metrics}
                metrics_dict["loss"] = self._val_loss.result()
                return predictions, metrics_dict

        return predictions

        
    def run_on_instance(self, input, return_raw_output=False, transform_input=True):
        """ Run the model on one input

            Args:
                input: numpy.array
                    The input in the shape expected by :func:`transform_input`
                return_raw_output:bool
                    If False, the model output will be transformed by :func:`transform_output`.
                    If true, the model output will be returned without any modifications.
                transform_input:bool
                    If True, the input is transformed by the interface's :func:`transform_input` method

            Returns:
                output
                    The model output
                
        """        
        if transform_input == True:
            input = self._transform_input(input)

        output = self.model.predict(input)
        
        if not return_raw_output:
            return self._transform_output(output)
        else:
            return output

    
    def run_on_batch(self, input_batch, return_raw_output=False, transform_input=True):
        """ Run the model on a batch of inputs

            Args:
                input_batch: numpy.array
                    The  batch of inputs 
                transform_input:bool
                    If True, the input_batch is transformed by the interface's :func:`transform_input` method
                return_raw_output:bool
                    If False, the model output will be transformed by :func:`transform_output`.
                    If true, the model output will be returned without any modifications.

            Returns:
                output
                    The corresponding batch of model outputs
        """
        if transform_input == True:
            input_batch = self._transform_input(input_batch)
        output = self.model.predict_on_batch(input_batch)
        
        if not return_raw_output:
            return self._transform_output(output)
        else:
            return output
