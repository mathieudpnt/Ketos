import pytest
import numpy as np
import tensorflow as tf
from ketos.neural_networks.dev_utils.nn_interface import RecipeCompat
from ketos.neural_networks.densenet import ConvBlock, DenseBlock, TransitionBlock, DenseNetArch, DenseNetInterface
from ketos.neural_networks.dev_utils.losses import FScoreLoss
#from ketos.neural_networks.dev_utils.metrics import Precision, Recall, Accuracy, FScore
import os
import tables
import json

current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


@pytest.fixture
def recipe_dict():
    recipe = {'dense_blocks':[ 6, 12, 24, 16],
                    'growth_rate':32,
                    'compression_factor':0.5,
                    'n_classes':2,
                    'dropout_rate':0.2,
                    'optimizer': {'recipe_name':'Adam', 'parameters': {'learning_rate':0.005}},
                    'loss_function': {'recipe_name':'CategoricalCrossEntropy', 'parameters':{}},  
                    'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]
        
    }
    return recipe

@pytest.fixture
def recipe():
    recipe = {'dense_blocks':[ 6, 12, 24, 16],
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
        



