import pytest
import numpy as np
import tensorflow as tf
from ketos.neural_networks.dev_utils.nn_interface import RecipeCompat
from ketos.neural_networks.inception import ConvBatchNormRelu, InceptionBlock, InceptionArch, InceptionInterface
#from ketos.neural_networks.dev_utils.losses import FScoreLoss
from ketos.data_handling.data_feeding import BatchGenerator
#from ketos.neural_networks.dev_utils.metrics import Precision, Recall, Accuracy, FScore
import os
import tables
import json

current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')

@pytest.fixture
def recipe_dict():
    recipe = {'interface': 'InceptionInterface',
                'n_blocks':3,
                'n_classes':2,
                'initial_filters':16,
                'optimizer': {'recipe_name':'Adam', 'parameters': {'learning_rate':0.005}},
                'loss_function': {'recipe_name':'CategoricalCrossentropy', 'parameters':{}},  
                'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]
    
    }

    return recipe

@pytest.fixture
def recipe():
    recipe = {'interface': 'InceptionInterface',
                'n_blocks':3,
                'n_classes':2,
                'initial_filters':16,
                'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
                'loss_function': RecipeCompat('CategoricalCrossentropy', tf.keras.losses.CategoricalCrossentropy),  
                'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy),
                            ],
                }

    return recipe

def test_ConvBatchNormRelu():
    layer = ConvBatchNormRelu(n_filters=16, filter_shape=3, strides=1, padding='same')

    
    assert len(layer.layers[0].layers) == 3
    assert isinstance(layer.layers[0].layers[0], tf.keras.layers.Conv2D)
    assert isinstance(layer.layers[0].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(layer.layers[0].layers[2], tf.keras.layers.ReLU)


def test_InceptionBlock():
    block = InceptionBlock(n_filters=16, strides=1)

    assert len(block.layers) == 6
    assert isinstance(block.layers[0], ConvBatchNormRelu)
    assert isinstance(block.layers[1], ConvBatchNormRelu)
    assert isinstance(block.layers[2], ConvBatchNormRelu)
    assert isinstance(block.layers[3], ConvBatchNormRelu)
    assert isinstance(block.layers[4], tf.keras.layers.MaxPooling2D)
    assert isinstance(block.layers[5], ConvBatchNormRelu)


def test_InceptionArch():
    inception = InceptionArch(n_blocks=3, n_classes=2, initial_filters=16)

    assert len(inception.layers) == 5
    assert isinstance(inception.layers[0], ConvBatchNormRelu)
    assert len(inception.layers[1].layers) == 6
    for block in inception.layers[1].layers:
        assert isinstance(block, InceptionBlock)
    assert isinstance(inception.layers[2], tf.keras.layers.GlobalAveragePooling2D)
    assert isinstance(inception.layers[3], tf.keras.layers.Dense)
    assert isinstance(inception.layers[4], tf.keras.layers.Softmax)


def test_InceptionInterface_build_from_recipe(recipe):
    inception = InceptionInterface._build_from_recipe(recipe)

    assert inception.optimizer.recipe_name == recipe['optimizer'].recipe_name
    assert inception.optimizer.instance.__class__ == recipe['optimizer'].instance.__class__
    assert inception.optimizer.args == recipe['optimizer'].args

    assert inception.loss_function.recipe_name == recipe['loss_function'].recipe_name
    assert inception.loss_function.instance.__class__ == recipe['loss_function'].instance.__class__
    assert inception.loss_function.args == recipe['loss_function'].args

    assert inception.metrics[0].recipe_name == recipe['metrics'][0].recipe_name
    assert inception.metrics[0].instance.__class__ == recipe['metrics'][0].instance.__class__
    assert inception.metrics[0].args == recipe['metrics'][0].args

    assert inception.n_blocks == recipe['n_blocks']
    assert inception.initial_filters == recipe['initial_filters']
    assert inception.n_classes ==  recipe['n_classes']    

def test_extract_recipe_dict(recipe, recipe_dict):
    inception = InceptionInterface._build_from_recipe(recipe)
    written_recipe = inception._extract_recipe_dict()

    assert written_recipe == recipe_dict

def test_read_recipe_file(recipe, recipe_dict):
    path_to_recipe_file = os.path.join(path_to_tmp, "test_inception_recipe.json")
    inception = InceptionInterface._build_from_recipe(recipe)
    written_recipe = inception._extract_recipe_dict()
    inception.save_recipe_file(path_to_recipe_file)

    #Read recipe as a recipe dict
    read_recipe = inception._read_recipe_file(path_to_recipe_file,return_recipe_compat=False)
    assert read_recipe == recipe_dict

    #Read recipe as a recipe dict with RecipeCompat objects
    read_recipe = inception._read_recipe_file(path_to_recipe_file,return_recipe_compat=True)
    assert read_recipe['optimizer'].recipe_name ==recipe['optimizer'].recipe_name
    assert read_recipe['optimizer'].instance.__class__ == recipe['optimizer'].instance.__class__
    assert read_recipe['optimizer'].args == recipe['optimizer'].args

    assert read_recipe['loss_function'].recipe_name == recipe['loss_function'].recipe_name
    assert read_recipe['loss_function'].instance.__class__ == recipe['loss_function'].instance.__class__
    assert read_recipe['loss_function'].args == recipe['loss_function'].args
    
    assert read_recipe['metrics'][0].recipe_name == recipe['metrics'][0].recipe_name
    assert read_recipe['metrics'][0].instance.__class__ == recipe['metrics'][0].instance.__class__
    assert read_recipe['metrics'][0].args == recipe['metrics'][0].args

    assert read_recipe['n_blocks'] == recipe['n_blocks']
    assert read_recipe['initial_filters'] == recipe['initial_filters']
    assert read_recipe['n_classes'] ==  recipe['n_classes']  
    

def test_train_Inception(sample_data):
    data, labels = sample_data
    inception = InceptionInterface() #default inception network
    train_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)
    val_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)

    inception.train_generator = train_generator
    inception.val_generator = val_generator

    inception.train_loop(2, checkpoint_freq=0)



    



