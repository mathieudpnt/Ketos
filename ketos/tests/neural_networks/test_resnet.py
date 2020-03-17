import pytest
import numpy as np
import tensorflow as tf
from ketos.neural_networks.nn_interface import RecipeCompat
from ketos.neural_networks.resnet import ResNetBlock, ResNetArch, ResNetInterface
from ketos.neural_networks.losses import FScoreLoss
from ketos.neural_networks.metrics import Precision, Recall, Accuracy, FScore
import os
import tables
import json

current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')

@pytest.fixture
def recipe_dict():
    recipe = {'block_sets':[2,2,2],
               'n_classes':2,
               'initial_filters':16,        
               'optimizer': {'name':'Adam', 'parameters': {'learning_rate':0.005}},
               'loss_function': {'name':'FScoreLoss', 'parameters':{}},  
               'metrics': [{'name':'CategoricalAccuracy', 'parameters':{}}]
        
    }
    return recipe
@pytest.fixture
def recipe():
    recipe = {'block_sets':[2,2,2],
               'n_classes':2,
               'initial_filters':16,        
               'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
               'loss_function': RecipeCompat('FScoreLoss', FScoreLoss),  
               'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)]
        
    }
    return recipe


def test_ResNetBlock():
    block = ResNetBlock(filters=1, strides=1, residual_path=False)

    assert len(block.layers) == 5
    assert isinstance(block.layers[0], tf.keras.layers.Conv2D)
    assert isinstance(block.layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[2], tf.keras.layers.Conv2D)
    assert isinstance(block.layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[4], tf.keras.layers.Dropout)



def test_ResNetBlock_residual():
    block = ResNetBlock(filters=1, strides=1, residual_path=True)

    assert len(block.layers) == 7
    assert isinstance(block.layers[0], tf.keras.layers.Conv2D)
    assert isinstance(block.layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[2], tf.keras.layers.Conv2D)
    assert isinstance(block.layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[4], tf.keras.layers.Conv2D)
    assert isinstance(block.layers[5], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[6], tf.keras.layers.Dropout)

def test_ResNetArch():
    resnet = ResNetArch(block_sets=[2,2,2], n_classes=2, initial_filters=16)

    assert len(resnet.layers) == 6
    assert isinstance(resnet.layers[0], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1], tf.keras.models.Sequential)
    assert isinstance(resnet.layers[2], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[3], tf.keras.layers.GlobalAveragePooling2D)
    assert isinstance(resnet.layers[4], tf.keras.layers.Dense)
    assert isinstance(resnet.layers[5], tf.keras.layers.Softmax)

    #ResNet blocks
    assert len(resnet.layers[1].layers) == 6
    assert isinstance(resnet.layers[1].layers[0], ResNetBlock)
    assert isinstance(resnet.layers[1].layers[1], ResNetBlock)
    assert isinstance(resnet.layers[1].layers[2], ResNetBlock)
    assert isinstance(resnet.layers[1].layers[3], ResNetBlock)
    assert isinstance(resnet.layers[1].layers[4], ResNetBlock)
    assert isinstance(resnet.layers[1].layers[5], ResNetBlock)

    #Block 1
    assert isinstance(resnet.layers[1].layers[0].layers[0], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[0].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[0].layers[2], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[0].layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[0].layers[4], tf.keras.layers.Dropout)

    #Block 2
    assert isinstance(resnet.layers[1].layers[1].layers[0], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[1].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[1].layers[2], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[1].layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[1].layers[4], tf.keras.layers.Dropout)

    #Block 3
    assert isinstance(resnet.layers[1].layers[2].layers[0], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[2].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[2].layers[2], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[2].layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[2].layers[4], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[2].layers[5], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[2].layers[6], tf.keras.layers.Dropout)

    #Block 4
    assert isinstance(resnet.layers[1].layers[3].layers[0], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[3].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[3].layers[2], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[3].layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[3].layers[4], tf.keras.layers.Dropout)

    #Block 5
    assert isinstance(resnet.layers[1].layers[4].layers[0], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[4].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[4].layers[2], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[4].layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[4].layers[4], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[4].layers[5], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[4].layers[6], tf.keras.layers.Dropout)

    #Block 6
    assert isinstance(resnet.layers[1].layers[5].layers[0], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[5].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[5].layers[2], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[5].layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[5].layers[4], tf.keras.layers.Dropout)


def test_ResNetInterface_build_from_recipe(recipe):
    resnet = ResNetInterface.build_from_recipe(recipe)

    assert resnet.optimizer.name == recipe['optimizer'].name
    assert resnet.optimizer.func.__class__ == recipe['optimizer'].func.__class__
    assert resnet.optimizer.args == recipe['optimizer'].args

    assert resnet.loss_function.name == recipe['loss_function'].name
    assert resnet.loss_function.func.__class__ == recipe['loss_function'].func.__class__
    assert resnet.loss_function.args == recipe['loss_function'].args

    assert resnet.metrics[0].name == recipe['metrics'][0].name
    assert resnet.metrics[0].func.__class__ == recipe['metrics'][0].func.__class__
    assert resnet.metrics[0].args == recipe['metrics'][0].args

    assert resnet.initial_filters == recipe['initial_filters']
    assert resnet.block_sets == recipe['block_sets']
    assert resnet.n_classes ==  recipe['n_classes']


def test_write_recipe(recipe, recipe_dict):
    resnet = ResNetInterface.build_from_recipe(recipe)
    written_recipe = resnet.write_recipe()

    assert written_recipe == recipe_dict


def test_read_recipe_file(recipe, recipe_dict):
    path_to_recipe_file = os.path.join(path_to_tmp, "test_resnet_recipe.json")
    resnet = ResNetInterface.build_from_recipe(recipe)
    written_recipe = resnet.write_recipe()
    resnet.save_recipe(path_to_recipe_file)

    #Read recipe as a recipe dict
    read_recipe = resnet.read_recipe_file(path_to_recipe_file,return_recipe_compat=False)
    assert read_recipe == recipe_dict

    #Read recipe as a recipe dict with RecipeCompat objects
    read_recipe = resnet.read_recipe_file(path_to_recipe_file,return_recipe_compat=True)
    assert read_recipe['optimizer'].name ==recipe['optimizer'].name
    assert read_recipe['optimizer'].func.__class__ == recipe['optimizer'].func.__class__
    assert read_recipe['optimizer'].args == recipe['optimizer'].args

    assert read_recipe['loss_function'].name == recipe['loss_function'].name
    assert read_recipe['loss_function'].func.__class__ == recipe['loss_function'].func.__class__
    assert read_recipe['loss_function'].args == recipe['loss_function'].args
    
    assert read_recipe['metrics'][0].name == recipe['metrics'][0].name
    assert read_recipe['metrics'][0].func.__class__ == recipe['metrics'][0].func.__class__
    assert read_recipe['metrics'][0].args == recipe['metrics'][0].args

    assert read_recipe['initial_filters'] == recipe['initial_filters']
    assert read_recipe['block_sets'] == recipe['block_sets']
    assert read_recipe['n_classes'] ==  recipe['n_classes']
