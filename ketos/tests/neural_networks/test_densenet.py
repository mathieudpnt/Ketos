import pytest
import numpy as np
import tensorflow as tf
from ketos.neural_networks.dev_utils.nn_interface import RecipeCompat
from ketos.neural_networks.densenet import ConvBlock, DenseBlock, TransitionBlock, DenseNetArch, DenseNetInterface
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
    recipe = {'interface': 'DenseNetInterface',
                    'dense_blocks':[ 6, 12, 24, 16],
                    'growth_rate':32,
                    'compression_factor':0.5,
                    'n_classes':2,
                    'dropout_rate':0.2,
                    'optimizer': {'recipe_name':'Adam', 'parameters': {'learning_rate':0.005}},
                    'loss_function': {'recipe_name':'CategoricalCrossentropy', 'parameters':{}},  
                    'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]
        
    }
    return recipe

@pytest.fixture
def recipe():
    recipe = {'interface': 'DenseNetInterface',
                    'dense_blocks':[ 6, 12, 24, 16],
                    'growth_rate':32,
                    'compression_factor':0.5,
                    'n_classes':2,
                    'dropout_rate':0.2,
                    'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
                    'loss_function': RecipeCompat('CategoricalCrossentropy', tf.keras.losses.CategoricalCrossentropy),  
                    'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy),
                                ],
                    }
    return recipe


def test_ConvBlock():
    block = ConvBlock(growth_rate=32)

    assert len(block.layers) == 6

    assert isinstance(block.layers[0], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[1], tf.keras.layers.Activation)
    assert isinstance(block.layers[2], tf.keras.layers.Conv2D)

    assert isinstance(block.layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[4], tf.keras.layers.Activation)
    assert isinstance(block.layers[5], tf.keras.layers.Conv2D)


def test_DenseBlock():
    block = DenseBlock(growth_rate=32, n_blocks=16)

    assert len(block.layers) == 1
    assert len(block.layers[0].layers) == 16
    for i in range(16):
        assert isinstance(block.layers[0].layers[i], ConvBlock)
        assert len(block.layers[0].layers[i].layers) == 6
        assert isinstance(block.layers[0].layers[i].layers[0], tf.keras.layers.BatchNormalization)
        assert isinstance(block.layers[0].layers[i].layers[1], tf.keras.layers.Activation)
        assert isinstance(block.layers[0].layers[i].layers[2], tf.keras.layers.Conv2D)

        assert isinstance(block.layers[0].layers[i].layers[3], tf.keras.layers.BatchNormalization)
        assert isinstance(block.layers[0].layers[i].layers[4], tf.keras.layers.Activation)
        assert isinstance(block.layers[0].layers[i].layers[5], tf.keras.layers.Conv2D)


def test_TransitionBlock():
    block = TransitionBlock(n_channels=64, compression_factor=0.5)

    assert len(block.layers) == 5
    assert isinstance(block.layers[0], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[1], tf.keras.layers.Conv2D)
    assert isinstance(block.layers[2], tf.keras.layers.Dropout)
    assert isinstance(block.layers[3], tf.keras.layers.Activation)
    assert isinstance(block.layers[4], tf.keras.layers.AveragePooling2D)

def test_DenseNetArch(recipe):
    densenet = DenseNetArch(dense_blocks=recipe['dense_blocks'], growth_rate=recipe['growth_rate'],
                            compression_factor=recipe['compression_factor'], n_classes=recipe['n_classes'],
                            dropout_rate=recipe['dropout_rate'])

    
    assert len(densenet.layers) == 9

    assert isinstance(densenet.layers[0], tf.keras.layers.Conv2D)
    assert isinstance(densenet.layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(densenet.layers[2], tf.keras.layers.Activation)
    assert isinstance(densenet.layers[3], tf.keras.layers.MaxPool2D)

    #Dense blocks
    assert len(densenet.layers[4].layers) == (len(recipe['dense_blocks']) * 2)
    print(densenet.layers[4].layers)
    for block_id, block in enumerate(densenet.layers[4].layers):
        if block_id % 2 == 0: #DenseBlock
            assert isinstance(block, DenseBlock)
            assert len(block.layers[0].layers) == recipe['dense_blocks'][int(block_id / 2)]
        else: #TransitionBlock     
            assert isinstance(block, TransitionBlock)


    
def test_DenseNet_Interface_build_from_recipe(recipe):
    densenet = DenseNetInterface._build_from_recipe(recipe)

    assert densenet.optimizer.recipe_name == recipe['optimizer'].recipe_name
    assert densenet.optimizer.instance.__class__ == recipe['optimizer'].instance.__class__
    assert densenet.optimizer.args == recipe['optimizer'].args

    assert densenet.loss_function.recipe_name == recipe['loss_function'].recipe_name
    assert densenet.loss_function.instance.__class__ == recipe['loss_function'].instance.__class__
    assert densenet.loss_function.args == recipe['loss_function'].args

    assert densenet.metrics[0].recipe_name == recipe['metrics'][0].recipe_name
    assert densenet.metrics[0].instance.__class__ == recipe['metrics'][0].instance.__class__
    assert densenet.metrics[0].args == recipe['metrics'][0].args

    assert densenet.dense_blocks == recipe['dense_blocks']
    assert densenet.growth_rate == recipe['growth_rate']
    assert densenet.compression_factor ==  recipe['compression_factor']    
    assert densenet.dropout_rate ==  recipe['dropout_rate']    


def test_extract_recipe_dict(recipe, recipe_dict):
    densenet = DenseNetInterface._build_from_recipe(recipe)
    written_recipe = densenet._extract_recipe_dict()

    assert written_recipe == recipe_dict

def test_read_recipe_file(recipe, recipe_dict):
    path_to_recipe_file = os.path.join(path_to_tmp, "test_densenet_recipe.json")
    densenet = DenseNetInterface._build_from_recipe(recipe)
    written_recipe = densenet._extract_recipe_dict()
    densenet.save_recipe_file(path_to_recipe_file)

    #Read recipe as a recipe dict
    read_recipe = densenet._read_recipe_file(path_to_recipe_file,return_recipe_compat=False)
    assert read_recipe == recipe_dict

    #Read recipe as a recipe dict with RecipeCompat objects
    read_recipe = densenet._read_recipe_file(path_to_recipe_file,return_recipe_compat=True)
    assert read_recipe['optimizer'].recipe_name ==recipe['optimizer'].recipe_name
    assert read_recipe['optimizer'].instance.__class__ == recipe['optimizer'].instance.__class__
    assert read_recipe['optimizer'].args == recipe['optimizer'].args

    assert read_recipe['loss_function'].recipe_name == recipe['loss_function'].recipe_name
    assert read_recipe['loss_function'].instance.__class__ == recipe['loss_function'].instance.__class__
    assert read_recipe['loss_function'].args == recipe['loss_function'].args
    
    assert read_recipe['metrics'][0].recipe_name == recipe['metrics'][0].recipe_name
    assert read_recipe['metrics'][0].instance.__class__ == recipe['metrics'][0].instance.__class__
    assert read_recipe['metrics'][0].args == recipe['metrics'][0].args

    assert read_recipe['dense_blocks'] == recipe['dense_blocks']
    assert read_recipe['growth_rate'] == recipe['growth_rate']
    assert read_recipe['compression_factor'] ==  recipe['compression_factor']    
    assert read_recipe['dropout_rate'] ==  recipe['dropout_rate']  


def test_train_DenseNet(sample_data):
    data, labels = sample_data
    densenet = DenseNetInterface() #default densenet
    train_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)
    val_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)

    densenet.train_generator = train_generator
    densenet.val_generator = val_generator

    densenet.train_loop(2, checkpoint_freq=0)


