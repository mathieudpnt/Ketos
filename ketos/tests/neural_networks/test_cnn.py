import pytest
import numpy as np
import tensorflow as tf
from ketos.neural_networks.nn_interface import RecipeCompat
from ketos.neural_networks.cnn import CNNArch, CNNInterface
from ketos.neural_networks.losses import FScoreLoss
from ketos.neural_networks.metrics import Precision, Recall, Accuracy, FScore
from ketos.data_handling.data_feeding import BatchGenerator
import os
import tables
import json



current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


@pytest.fixture
def recipe_simple_dict():
    recipe = {'conv_set':[(64, False), (128, True), (256, True)],
               'dense_set': [512, 256],
               'n_classes':2,
               'optimizer': {'name':'Adam', 'parameters': {'learning_rate':0.005}},
               'loss_function': {'name':'FScoreLoss', 'parameters':{}},  
               'metrics': [{'name':'CategoricalAccuracy', 'parameters':{}}]

    }

    return recipe


@pytest.fixture
def recipe_simple():
    recipe = {'conv_set':[(64, False), (128, True), (256, True)],
               'dense_set': [512, 256],
               'n_classes':2,        
               'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
               'loss_function': RecipeCompat('FScoreLoss', FScoreLoss),  
               'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)]
        
    }
    return recipe


@pytest.fixture
def recipe_detailed_dict():
    recipe = {'convolutional_layers':  [{'n_filters':64, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True},
                                    {'n_filters':128, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':(2,2) , 'strides':(2,2)}, 'batch_normalization':True},
                                    {'n_filters':256, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':(2,2) , 'strides':(2,2)}, 'batch_normalization':True}],
              'dense_layers':[{'n_hidden':512, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    {'n_hidden':256, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    ],
               'n_classes':2,
               'optimizer': {'name':'Adam', 'parameters': {'learning_rate':0.005}},
               'loss_function': {'name':'FScoreLoss', 'parameters':{}},  
               'metrics': [{'name':'CategoricalAccuracy', 'parameters':{}}]

    }

    return recipe

@pytest.fixture
def recipe_detailed():
    recipe = {'convolutional_layers':  [{'n_filters':64, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True},
                                    {'n_filters':128, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':(2,2) , 'strides':(2,2)}, 'batch_normalization':True},
                                    {'n_filters':256, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':(2,2) , 'strides':(2,2)}, 'batch_normalization':True}],
              'dense_layers':[{'n_hidden':512, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    {'n_hidden':256, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    ],
               'n_classes':2,
               'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
               'loss_function': RecipeCompat('FScoreLoss', FScoreLoss),  
               'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)]
        
    }

    return recipe


def test_convolutional_layers_from_conv_set(recipe_simple, recipe_detailed):
    detailed_layers = CNNInterface.convolutional_layers_from_conv_set(recipe_simple['conv_set'])
    assert detailed_layers == recipe_detailed['convolutional_layers']
    

def test_dense_layers_from_dense_set(recipe_simple, recipe_detailed):
    detailed_layers = CNNInterface.dense_layers_from_dense_set(recipe_simple['dense_set'])
    assert detailed_layers == recipe_detailed['dense_layers']
    
 
def test_CNNInterface_build_from_recipe_simple(recipe_simple, recipe_detailed):
    cnn = CNNInterface.build_from_recipe(recipe_simple)

    assert cnn.optimizer.name == recipe_simple['optimizer'].name
    assert cnn.optimizer.func.__class__ == recipe_simple['optimizer'].func.__class__
    assert cnn.optimizer.args == recipe_simple['optimizer'].args

    assert cnn.loss_function.name == recipe_simple['loss_function'].name
    assert cnn.loss_function.func.__class__ == recipe_simple['loss_function'].func.__class__
    assert cnn.loss_function.args == recipe_simple['loss_function'].args

    assert cnn.metrics[0].name == recipe_simple['metrics'][0].name
    assert cnn.metrics[0].func.__class__ == recipe_simple['metrics'][0].func.__class__
    assert cnn.metrics[0].args == recipe_simple['metrics'][0].args

    assert cnn.conv_set == recipe_simple['conv_set']
    assert cnn.dense_set == recipe_simple['dense_set']
    # assert cnn.convolutional_layers == recipe_detailed['convolutional_layers']
    # assert cnn.dense_layers == recipe_detailed['dense_layers']
    assert cnn.n_classes ==  recipe_simple['n_classes']
    

def test_CNNInterface_build_from_recipe_detailed(recipe_detailed):
    cnn = CNNInterface.build_from_recipe(recipe_detailed)

    assert cnn.optimizer.name == recipe_detailed['optimizer'].name
    assert cnn.optimizer.func.__class__ == recipe_detailed['optimizer'].func.__class__
    assert cnn.optimizer.args == recipe_detailed['optimizer'].args

    assert cnn.loss_function.name == recipe_detailed['loss_function'].name
    assert cnn.loss_function.func.__class__ == recipe_detailed['loss_function'].func.__class__
    assert cnn.loss_function.args == recipe_detailed['loss_function'].args

    assert cnn.metrics[0].name == recipe_detailed['metrics'][0].name
    assert cnn.metrics[0].func.__class__ == recipe_detailed['metrics'][0].func.__class__
    assert cnn.metrics[0].args == recipe_detailed['metrics'][0].args

    assert cnn.convolutional_layers == recipe_detailed['convolutional_layers']
    assert cnn.dense_layers == recipe_detailed['dense_layers']
    assert cnn.n_classes ==  recipe_detailed['n_classes']

def test_write_recipe_simple(recipe_simple, recipe_simple_dict, recipe_detailed):
    cnn = CNNInterface.build_from_recipe(recipe_simple)
    written_recipe = cnn.write_recipe()

    #Even when the model is built from a simplified recipe, the detailed form will still be included when writing the recipe again

    recipe_simple_dict['convolutional_layers'] = recipe_detailed['convolutional_layers']
    recipe_simple_dict['dense_layers'] = recipe_detailed['dense_layers']

    assert written_recipe == recipe_simple_dict

